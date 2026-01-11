#!/usr/bin/env python3
"""
interview_processor.py - Unified interview processing tool for Klavis.

A production-ready tool for processing procurement stakeholder interviews
through transcription (Whisper) and insight extraction (Ollama).

Modes:
    --watch     Monitor folder continuously for new files
    --file      Process a single specific file
    --batch     Process all files in raw/ folder once

Setup Instructions:
    # 1. Install Ollama
    brew install ollama

    # 2. Pull the model
    ollama pull mistral

    # 3. Install Python dependencies
    pip install openai-whisper watchdog

    # 4. Create directories (auto-created on first run)
    mkdir -p ~/klavis-interviews/{raw,transcripts,insights,archive}

    # 5. Test installation
    python interview_processor.py --test

Usage:
    python interview_processor.py --watch
    python interview_processor.py --file ~/Downloads/interview.m4a
    python interview_processor.py --batch --model small
    python interview_processor.py --status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, List, Dict, Any

# =============================================================================
# Configuration
# =============================================================================

WHISPER_MODEL = "medium"  # Options: tiny, base, small, medium, large
MIN_FILE_SIZE = 100000    # 100KB minimum (skip test recordings)
WATCH_DELAY = 5           # Seconds to wait after new file detected
SUPPORTED_FORMATS = ['.m4a', '.wav', '.mp3', '.mp4']
OLLAMA_MODEL = "mistral"
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB log rotation
LOG_BACKUP_COUNT = 3

# Directory paths
BASE_DIR = Path.home() / "klavis-interviews"
RAW_DIR = BASE_DIR / "raw"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
INSIGHTS_DIR = BASE_DIR / "insights"
ARCHIVE_DIR = BASE_DIR / "archive"
FAILED_DIR = BASE_DIR / "failed"
LOG_FILE = BASE_DIR / "processing.log"
FAILED_LOG = BASE_DIR / "failed_files.json"

# =============================================================================
# Ollama Extraction Prompt
# =============================================================================

EXTRACTION_PROMPT = """You are analyzing a procurement stakeholder interview transcript. Extract the following in structured markdown format:

## PAIN POINTS
- [List specific problems mentioned with context]

## CURRENT WORKAROUNDS
- [How they handle issues today]

## RISK PATTERNS
- [Recurring failures or vulnerabilities observed]

## OBJECTIONS/CONCERNS
- [Hesitations about potential solutions]

## DECISION CRITERIA
- [What would make them buy/pilot a solution]

## FOLLOW-UP QUESTIONS
- [5-7 specific questions to ask in next conversation]

## KEY QUOTES
- "[Verbatim quotes worth preserving]"

## EXECUTIVE SUMMARY
- [2-3 sentence summary of the conversation]

TRANSCRIPT:
{transcript_text}"""

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    success: bool
    filepath: Path
    error_message: Optional[str] = None
    transcript_path: Optional[Path] = None
    insights_path: Optional[Path] = None
    word_count: int = 0
    section_count: int = 0
    duration_seconds: float = 0.0
    processing_time: float = 0.0


@dataclass
class SessionStats:
    """Statistics for a processing session."""
    files_processed: int = 0
    files_failed: int = 0
    total_time: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def average_time(self) -> float:
        if self.files_processed == 0:
            return 0.0
        return self.total_time / self.files_processed

    def add_result(self, result: ProcessingResult) -> None:
        if result.success:
            self.files_processed += 1
            self.total_time += result.processing_time
        else:
            self.files_failed += 1
            if result.error_message:
                self.errors.append(f"{result.filepath.name}: {result.error_message}")


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with rotation and console output."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("interview_processor")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler with rotation
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_SIZE,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Global logger instance
logger: Optional[logging.Logger] = None

# =============================================================================
# Helper Functions
# =============================================================================

def timestamp() -> str:
    """Return formatted timestamp for display."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m {secs}s"


def format_time_mmss(seconds: float) -> str:
    """Format duration in MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def log_progress(emoji: str, message: str, file_num: int = 0, total: int = 0) -> None:
    """Log a progress message with timestamp and emoji."""
    progress = f" ({file_num} of {total})" if total > 0 else ""
    logger.info(f"[{timestamp()}] {emoji} {message}{progress}")


def create_directories() -> None:
    """Create all required directories if they don't exist."""
    for directory in [RAW_DIR, TRANSCRIPTS_DIR, INSIGHTS_DIR, ARCHIVE_DIR, FAILED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_sections(text: str) -> int:
    """Count markdown sections (## headers) in text."""
    return text.count("\n## ")


# =============================================================================
# Validation Functions
# =============================================================================

def validate_file(filepath: Path) -> tuple[bool, str]:
    """
    Validate the input audio file.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not filepath.exists():
        return False, f"File does not exist: {filepath}"

    if filepath.suffix.lower() not in SUPPORTED_FORMATS:
        return False, (
            f"Unsupported format: {filepath.suffix}. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    file_size = filepath.stat().st_size
    if file_size < MIN_FILE_SIZE:
        return False, (
            f"File too small ({file_size/1024:.1f}KB). "
            f"Minimum: {MIN_FILE_SIZE/1024:.0f}KB"
        )

    return True, ""


def check_ollama_running() -> tuple[bool, str]:
    """Check if Ollama is running and the model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return False, "Ollama is not running"

        if OLLAMA_MODEL not in result.stdout:
            return False, f"Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}"

        return True, ""
    except FileNotFoundError:
        return False, "Ollama not installed. Install with: brew install ollama"
    except subprocess.TimeoutExpired:
        return False, "Ollama not responding (timeout)"


def check_disk_space(min_gb: float = 1.0) -> tuple[bool, str]:
    """Check if sufficient disk space is available."""
    try:
        stat = os.statvfs(BASE_DIR)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        if free_gb < min_gb:
            return False, f"Low disk space: {free_gb:.1f}GB free (need {min_gb}GB)"
        return True, ""
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True, ""  # Don't block on disk check failure


def check_whisper_available() -> tuple[bool, str]:
    """Check if Whisper is available."""
    try:
        import whisper
        return True, ""
    except ImportError:
        return False, "Whisper not installed. Run: pip install openai-whisper"


# =============================================================================
# Audio Functions
# =============================================================================

def get_audio_duration(filepath: Path) -> Optional[float]:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-show_entries",
                "format=duration", "-of", "json", str(filepath)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as e:
        logger.debug(f"Could not get audio duration: {e}")
        return None


# =============================================================================
# Core Processing Functions
# =============================================================================

def transcribe_audio(filepath: Path, model: str = WHISPER_MODEL) -> tuple[Optional[str], Optional[float]]:
    """
    Transcribe audio file using Whisper.

    Returns:
        Tuple of (transcript_text, duration_seconds) or (None, None) on failure
    """
    try:
        import whisper
    except ImportError:
        logger.error("Whisper not installed")
        return None, None

    log_progress("🎙️ ", f"Transcribing with whisper-{model}...")
    logger.debug(f"Loading Whisper model: {model}")

    try:
        whisper_model = whisper.load_model(model)
        result = whisper_model.transcribe(str(filepath))

        transcript = result.get("text", "").strip()

        # Get duration from segments or ffprobe
        duration = None
        if "segments" in result and result["segments"]:
            duration = result["segments"][-1].get("end")
        if duration is None:
            duration = get_audio_duration(filepath)

        logger.debug(f"Transcription complete: {len(transcript)} chars, {count_words(transcript)} words")
        return transcript, duration

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None, None


def extract_insights(transcript: str) -> Optional[str]:
    """
    Send transcript to Ollama for insight extraction.

    Returns:
        Extracted insights as markdown, or None on failure
    """
    log_progress("🧠", f"Extracting insights with {OLLAMA_MODEL}...")

    prompt = EXTRACTION_PROMPT.format(transcript_text=transcript)

    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            logger.error(f"Ollama error: {result.stderr}")
            return None

        insights = result.stdout.strip()
        logger.debug(f"Extraction complete: {len(insights)} chars, {count_sections(insights)} sections")
        return insights

    except subprocess.TimeoutExpired:
        logger.error("Ollama extraction timed out (5 minutes)")
        return None
    except Exception as e:
        logger.error(f"Ollama extraction failed: {e}")
        return None


def save_transcript(filepath: Path, transcript: str, duration: Optional[float]) -> Optional[Path]:
    """Save transcript to markdown file with YAML frontmatter."""
    filename = filepath.stem
    output_path = TRANSCRIPTS_DIR / f"{filename}.md"

    duration_str = format_time_mmss(duration) if duration else "Unknown"
    word_count = count_words(transcript)

    content = f"""---
filename: {filepath.name}
processed_date: {datetime.now().isoformat()}
duration: {duration_str}
word_count: {word_count}
model: whisper-{WHISPER_MODEL}
---

# Transcript: {filename}

{transcript}
"""

    try:
        output_path.write_text(content, encoding="utf-8")
        log_progress("✅", f"Transcript saved ({word_count} words)")
        logger.debug(f"Transcript saved to: {output_path}")
        return output_path
    except PermissionError:
        logger.error(f"Permission denied writing to: {output_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to save transcript: {e}")
        return None


def save_insights(filepath: Path, insights: str) -> Optional[Path]:
    """Save insights to markdown file with YAML frontmatter."""
    filename = filepath.stem
    output_path = INSIGHTS_DIR / f"{filename}-insights.md"

    section_count = count_sections(insights)

    content = f"""---
source: {filepath.name}
processed_date: {datetime.now().isoformat()}
llm_model: {OLLAMA_MODEL}
section_count: {section_count}
---

# Interview Insights: {filename}

{insights}
"""

    try:
        output_path.write_text(content, encoding="utf-8")
        log_progress("✅", f"Insights saved ({section_count} sections)")
        logger.debug(f"Insights saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save insights: {e}")
        return None


def move_to_archive(filepath: Path) -> bool:
    """Move processed file to archive directory."""
    archive_path = ARCHIVE_DIR / filepath.name

    # Handle duplicate filenames
    counter = 1
    while archive_path.exists():
        archive_path = ARCHIVE_DIR / f"{filepath.stem}_{counter}{filepath.suffix}"
        counter += 1

    try:
        shutil.move(str(filepath), str(archive_path))
        log_progress("📦", "Moved to archive")
        logger.debug(f"Archived to: {archive_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to archive: {e}")
        return False


def record_failed_file(filepath: Path, error: str) -> None:
    """Record a failed file for later retry."""
    failed_data = {}
    if FAILED_LOG.exists():
        try:
            failed_data = json.loads(FAILED_LOG.read_text())
        except json.JSONDecodeError:
            pass

    failed_data[str(filepath)] = {
        "error": error,
        "timestamp": datetime.now().isoformat(),
        "attempts": failed_data.get(str(filepath), {}).get("attempts", 0) + 1
    }

    FAILED_LOG.write_text(json.dumps(failed_data, indent=2))


def process_single_file(
    filepath: Path,
    model: str = WHISPER_MODEL,
    skip_insights: bool = False,
    keep_original: bool = False,
    file_num: int = 0,
    total: int = 0
) -> ProcessingResult:
    """
    Process a single audio file through the full pipeline.

    Returns:
        ProcessingResult with success status and metadata
    """
    start_time = time.time()
    suffix = f" ({file_num} of {total})" if total > 0 else ""
    log_progress("📁", f"Processing: {filepath.name}{suffix}")

    # Validate file
    is_valid, error = validate_file(filepath)
    if not is_valid:
        logger.error(error)
        record_failed_file(filepath, error)
        return ProcessingResult(
            success=False,
            filepath=filepath,
            error_message=error
        )

    # Transcribe
    transcript, duration = transcribe_audio(filepath, model)
    if transcript is None:
        error = "Transcription failed"
        record_failed_file(filepath, error)
        return ProcessingResult(
            success=False,
            filepath=filepath,
            error_message=error
        )

    # Save transcript
    transcript_path = save_transcript(filepath, transcript, duration)
    if transcript_path is None:
        error = "Failed to save transcript"
        record_failed_file(filepath, error)
        return ProcessingResult(
            success=False,
            filepath=filepath,
            error_message=error
        )

    word_count = count_words(transcript)
    insights_path = None
    section_count = 0

    # Extract insights (unless skipped)
    if not skip_insights:
        insights = extract_insights(transcript)
        if insights is None:
            logger.warning("Insight extraction failed. Transcript saved, file not archived.")
            print("⚠️  Insight extraction failed. Transcript saved, file NOT archived.")
            record_failed_file(filepath, "Insight extraction failed")
            return ProcessingResult(
                success=False,
                filepath=filepath,
                error_message="Insight extraction failed",
                transcript_path=transcript_path,
                word_count=word_count,
                duration_seconds=duration or 0
            )

        insights_path = save_insights(filepath, insights)
        if insights_path is None:
            record_failed_file(filepath, "Failed to save insights")
            return ProcessingResult(
                success=False,
                filepath=filepath,
                error_message="Failed to save insights",
                transcript_path=transcript_path,
                word_count=word_count,
                duration_seconds=duration or 0
            )
        section_count = count_sections(insights)

    # Archive original (unless keeping)
    if not keep_original:
        move_to_archive(filepath)

    processing_time = time.time() - start_time
    log_progress("✨", f"Complete! ({format_duration(processing_time)})")

    return ProcessingResult(
        success=True,
        filepath=filepath,
        transcript_path=transcript_path,
        insights_path=insights_path,
        word_count=word_count,
        section_count=section_count,
        duration_seconds=duration or 0,
        processing_time=processing_time
    )


# =============================================================================
# Mode Handlers
# =============================================================================

def single_file_mode(
    filepath: Path,
    model: str = WHISPER_MODEL,
    skip_insights: bool = False,
    keep_original: bool = False
) -> bool:
    """Process a single file."""
    logger.info(f"\n{'='*60}")
    logger.info("Single File Processing Mode")
    logger.info(f"{'='*60}\n")

    result = process_single_file(
        filepath,
        model=model,
        skip_insights=skip_insights,
        keep_original=keep_original
    )

    return result.success


def batch_mode(
    model: str = WHISPER_MODEL,
    skip_insights: bool = False,
    keep_original: bool = False,
    dry_run: bool = False
) -> bool:
    """Process all files in the raw directory."""
    logger.info(f"\n{'='*60}")
    logger.info("Batch Processing Mode")
    logger.info(f"{'='*60}\n")

    # Find all files
    files = []
    for ext in SUPPORTED_FORMATS:
        files.extend(RAW_DIR.glob(f"*{ext}"))
        files.extend(RAW_DIR.glob(f"*{ext.upper()}"))

    files = sorted(set(files))

    if not files:
        logger.info("No files found in raw/ directory")
        return True

    logger.info(f"Found {len(files)} file(s) to process\n")

    if dry_run:
        logger.info("DRY RUN - Would process:")
        for f in files:
            size_kb = f.stat().st_size / 1024
            logger.info(f"  • {f.name} ({size_kb:.1f}KB)")
        return True

    stats = SessionStats()

    for i, filepath in enumerate(files, 1):
        result = process_single_file(
            filepath,
            model=model,
            skip_insights=skip_insights,
            keep_original=keep_original,
            file_num=i,
            total=len(files)
        )
        stats.add_result(result)

        # Small delay between files
        if i < len(files):
            time.sleep(1)

    # Print summary
    print(f"\n{'='*60}")
    print("Batch Processing Complete")
    print(f"{'='*60}")
    print(f"✅ Processed: {stats.files_processed}")
    print(f"❌ Failed: {stats.files_failed}")
    print(f"⏱️  Total time: {format_duration(stats.total_time)}")
    if stats.files_processed > 0:
        print(f"📈 Average per file: {format_duration(stats.average_time)}")

    if stats.errors:
        print("\nFailures:")
        for error in stats.errors:
            print(f"  • {error}")

    return stats.files_failed == 0


def watch_mode(
    model: str = WHISPER_MODEL,
    skip_insights: bool = False,
    keep_original: bool = False
) -> None:
    """Monitor the raw directory for new files."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    except ImportError:
        logger.error("watchdog not installed. Run: pip install watchdog")
        sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info("Watch Mode - Monitoring for new files")
    logger.info(f"{'='*60}")
    logger.info(f"Watching: {RAW_DIR}")
    logger.info(f"Model: whisper-{model}")
    logger.info("Press Ctrl+C to stop\n")

    stats = SessionStats()
    processed_files: set[str] = set()

    class AudioFileHandler(FileSystemEventHandler):
        def on_created(self, event: FileCreatedEvent) -> None:
            if event.is_directory:
                return

            filepath = Path(event.src_path)

            # Check if it's a supported format
            if filepath.suffix.lower() not in SUPPORTED_FORMATS:
                return

            # Avoid reprocessing
            if str(filepath) in processed_files:
                return

            logger.info(f"\n[{timestamp()}] 🆕 New file detected: {filepath.name}")
            logger.info(f"[{timestamp()}] ⏳ Waiting {WATCH_DELAY}s for file to finish writing...")

            # Wait for file to finish writing
            time.sleep(WATCH_DELAY)

            # Verify file still exists and is stable
            if not filepath.exists():
                logger.warning(f"File disappeared: {filepath.name}")
                return

            # Check file size is stable
            initial_size = filepath.stat().st_size
            time.sleep(1)
            if filepath.stat().st_size != initial_size:
                logger.info("File still being written, waiting longer...")
                time.sleep(WATCH_DELAY)

            processed_files.add(str(filepath))

            result = process_single_file(
                filepath,
                model=model,
                skip_insights=skip_insights,
                keep_original=keep_original
            )
            stats.add_result(result)

    observer = Observer()
    handler = AudioFileHandler()
    observer.schedule(handler, str(RAW_DIR), recursive=False)

    def signal_handler(sig, frame):
        logger.info("\n\nShutting down...")
        observer.stop()

        # Print session statistics
        print(f"\n[{timestamp()}] 📊 Session Statistics:")
        print(f"  ✅ Files processed: {stats.files_processed}")
        print(f"  ⏱️  Total time: {format_duration(stats.total_time)}")
        if stats.files_processed > 0:
            print(f"  📈 Average per file: {format_duration(stats.average_time)}")
        print(f"  ❌ Errors: {stats.files_failed}")

        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


# =============================================================================
# Utility Commands
# =============================================================================

def show_status() -> None:
    """Show current status of all directories."""
    print(f"\n{'='*60}")
    print("Klavis Interview Processor - Status")
    print(f"{'='*60}\n")

    def count_files(directory: Path, extensions: list) -> int:
        count = 0
        if directory.exists():
            for ext in extensions:
                count += len(list(directory.glob(f"*{ext}")))
                count += len(list(directory.glob(f"*{ext.upper()}")))
        return count

    raw_count = count_files(RAW_DIR, SUPPORTED_FORMATS)
    transcript_count = len(list(TRANSCRIPTS_DIR.glob("*.md"))) if TRANSCRIPTS_DIR.exists() else 0
    insights_count = len(list(INSIGHTS_DIR.glob("*-insights.md"))) if INSIGHTS_DIR.exists() else 0
    archive_count = count_files(ARCHIVE_DIR, SUPPORTED_FORMATS)

    print(f"📁 Raw files pending:    {raw_count}")
    print(f"📝 Transcripts created:  {transcript_count}")
    print(f"💡 Insights generated:   {insights_count}")
    print(f"📦 Files archived:       {archive_count}")

    # Check for failed files
    if FAILED_LOG.exists():
        try:
            failed_data = json.loads(FAILED_LOG.read_text())
            if failed_data:
                print(f"\n⚠️  Failed files: {len(failed_data)}")
                for filepath, info in failed_data.items():
                    print(f"   • {Path(filepath).name}: {info['error']}")
        except json.JSONDecodeError:
            pass

    # Check dependencies
    print(f"\n{'='*40}")
    print("Dependency Status:")
    print(f"{'='*40}")

    whisper_ok, whisper_err = check_whisper_available()
    print(f"{'✅' if whisper_ok else '❌'} Whisper: {'OK' if whisper_ok else whisper_err}")

    ollama_ok, ollama_err = check_ollama_running()
    print(f"{'✅' if ollama_ok else '❌'} Ollama: {'OK' if ollama_ok else ollama_err}")

    disk_ok, disk_err = check_disk_space()
    print(f"{'✅' if disk_ok else '⚠️ '} Disk: {'OK' if disk_ok else disk_err}")


def retry_failed(
    model: str = WHISPER_MODEL,
    skip_insights: bool = False,
    keep_original: bool = False
) -> bool:
    """Retry processing of previously failed files."""
    if not FAILED_LOG.exists():
        print("No failed files to retry")
        return True

    try:
        failed_data = json.loads(FAILED_LOG.read_text())
    except json.JSONDecodeError:
        print("Could not read failed files log")
        return False

    if not failed_data:
        print("No failed files to retry")
        return True

    print(f"\n{'='*60}")
    print(f"Retrying {len(failed_data)} failed file(s)")
    print(f"{'='*60}\n")

    stats = SessionStats()
    successful = []

    for filepath_str in list(failed_data.keys()):
        filepath = Path(filepath_str)

        if not filepath.exists():
            print(f"⚠️  File no longer exists: {filepath.name}")
            del failed_data[filepath_str]
            continue

        result = process_single_file(
            filepath,
            model=model,
            skip_insights=skip_insights,
            keep_original=keep_original
        )
        stats.add_result(result)

        if result.success:
            successful.append(filepath_str)

    # Remove successful files from failed log
    for filepath_str in successful:
        if filepath_str in failed_data:
            del failed_data[filepath_str]

    FAILED_LOG.write_text(json.dumps(failed_data, indent=2))

    print(f"\n✅ Retried successfully: {len(successful)}")
    print(f"❌ Still failing: {len(failed_data)}")

    return len(failed_data) == 0


def run_test() -> bool:
    """Run a test to verify the setup is working."""
    print(f"\n{'='*60}")
    print("Running Installation Test")
    print(f"{'='*60}\n")

    # Check dependencies
    print("Checking dependencies...")

    whisper_ok, whisper_err = check_whisper_available()
    print(f"  {'✅' if whisper_ok else '❌'} Whisper: {'OK' if whisper_ok else whisper_err}")

    ollama_ok, ollama_err = check_ollama_running()
    print(f"  {'✅' if ollama_ok else '❌'} Ollama: {'OK' if ollama_ok else ollama_err}")

    if not whisper_ok or not ollama_ok:
        print("\n❌ Dependencies not met. Please install missing components.")
        return False

    # Create a test audio file using say command (macOS)
    print("\nCreating test audio file...")

    test_text = """
    This is a test interview for the Klavis procurement analysis system.
    The main pain point we discussed was manual invoice processing taking too long.
    Currently, we handle this by having three people work overtime each month.
    I would consider a solution if it could reduce processing time by fifty percent.
    """

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
        test_file = Path(f.name)

    try:
        # Use macOS 'say' command to generate audio
        result = subprocess.run(
            ["say", "-o", str(test_file), "--data-format=alac", test_text],
            capture_output=True,
            timeout=30
        )

        if result.returncode != 0:
            print("❌ Could not create test audio (say command failed)")
            print("   This test requires macOS")
            return False

        # Check file size
        if test_file.stat().st_size < MIN_FILE_SIZE:
            print(f"⚠️  Test file too small ({test_file.stat().st_size} bytes)")
            print("   Skipping full test, but dependencies are OK")
            test_file.unlink()
            return True

        print(f"  ✅ Test file created: {test_file.stat().st_size / 1024:.1f}KB")

        # Process the test file
        print("\nProcessing test file...")
        result = process_single_file(
            test_file,
            model="tiny",  # Use tiny model for speed
            skip_insights=False,
            keep_original=True  # Don't archive test file
        )

        # Cleanup
        if test_file.exists():
            test_file.unlink()

        if result.success:
            print("\n✅ Test passed! Installation is working correctly.")

            # Clean up test outputs
            if result.transcript_path and result.transcript_path.exists():
                result.transcript_path.unlink()
            if result.insights_path and result.insights_path.exists():
                result.insights_path.unlink()

            return True
        else:
            print(f"\n❌ Test failed: {result.error_message}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Test audio creation timed out")
        return False
    except FileNotFoundError:
        print("❌ 'say' command not found (requires macOS)")
        print("   Dependencies are OK, but cannot run full test on this system")
        return True
    finally:
        if test_file.exists():
            test_file.unlink()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> int:
    """Main entry point for CLI."""
    global logger

    parser = argparse.ArgumentParser(
        prog="interview_processor",
        description="Klavis Interview Processor - Process procurement stakeholder interviews",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --watch                    Monitor folder continuously
  %(prog)s --file interview.m4a       Process single file
  %(prog)s --batch                    Process all files in raw/
  %(prog)s --batch --dry-run          Show what would be processed
  %(prog)s --status                   Show current status
  %(prog)s --retry-failed             Retry previously failed files
  %(prog)s --test                     Test installation

Supported formats: .m4a, .wav, .mp3, .mp4
        """
    )

    # Mode arguments (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Monitor folder continuously for new files"
    )
    mode_group.add_argument(
        "-f", "--file",
        type=Path,
        metavar="FILE",
        help="Process a specific file"
    )
    mode_group.add_argument(
        "-b", "--batch",
        action="store_true",
        help="Process all files in raw/ folder once"
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show current status of directories"
    )
    mode_group.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry processing of previously failed files"
    )
    mode_group.add_argument(
        "--test",
        action="store_true",
        help="Run a test to verify installation"
    )

    # Options
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help=f"Whisper model to use (default: {WHISPER_MODEL})"
    )
    parser.add_argument(
        "--skip-insights",
        action="store_true",
        help="Transcribe only, skip LLM extraction"
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Don't move to archive after processing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without processing"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug output"
    )

    args = parser.parse_args()

    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    # Setup logging
    logger = setup_logging(args.verbose)

    # Create directories
    create_directories()

    # Handle utility commands first
    if args.status:
        show_status()
        return 0

    if args.test:
        return 0 if run_test() else 1

    # Check dependencies for processing modes
    if not args.skip_insights:
        ollama_ok, ollama_err = check_ollama_running()
        if not ollama_ok:
            print(f"\n❌ {ollama_err}")
            print("\nTo start Ollama:")
            print("  1. Open a new terminal")
            print("  2. Run: ollama serve")
            print(f"  3. Then: ollama pull {OLLAMA_MODEL}")
            print("\nOr use --skip-insights to transcribe without extraction")
            return 1

    whisper_ok, whisper_err = check_whisper_available()
    if not whisper_ok:
        print(f"\n❌ {whisper_err}")
        return 1

    disk_ok, disk_err = check_disk_space()
    if not disk_ok:
        print(f"\n⚠️  {disk_err}")

    # Execute the selected mode
    if args.watch:
        watch_mode(
            model=args.model,
            skip_insights=args.skip_insights,
            keep_original=args.keep_original
        )
        return 0

    elif args.file:
        filepath = args.file.expanduser().resolve()
        success = single_file_mode(
            filepath,
            model=args.model,
            skip_insights=args.skip_insights,
            keep_original=args.keep_original
        )
        return 0 if success else 1

    elif args.batch:
        success = batch_mode(
            model=args.model,
            skip_insights=args.skip_insights,
            keep_original=args.keep_original,
            dry_run=args.dry_run
        )
        return 0 if success else 1

    elif args.retry_failed:
        success = retry_failed(
            model=args.model,
            skip_insights=args.skip_insights,
            keep_original=args.keep_original
        )
        return 0 if success else 1

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
