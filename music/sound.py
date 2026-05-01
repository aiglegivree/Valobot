"""
Simple speaker test.

This script plays piano.wav from this folder by default. It can also play
another audio file, or create a short WAV beep.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


DEFAULT_FREQUENCY_HZ = 880
DEFAULT_DURATION_SECONDS = 0.5
DEFAULT_VOLUME = 0.5
SAMPLE_RATE = 44_100
DEFAULT_AUDIO_FILE = Path(__file__).with_name("piano.wav")


def debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[sound.py] {message}")


def create_tone_wav(
    path: Path,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    volume: float = DEFAULT_VOLUME,
) -> None:
    """Create a mono WAV file containing a sine wave tone."""
    volume = max(0.0, min(volume, 1.0))
    sample_count = int(SAMPLE_RATE * duration_seconds)
    amplitude = int(32767 * volume)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

        for index in range(sample_count):
            sample = amplitude * math.sin(2 * math.pi * frequency_hz * index / SAMPLE_RATE)
            wav_file.writeframes(struct.pack("<h", int(sample)))


def describe_wav(path: Path, debug: bool) -> None:
    if path.suffix.lower() != ".wav":
        return

    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            duration = frame_count / frame_rate if frame_rate else 0.0
            debug_log(
                debug,
                (
                    f"WAV details: channels={wav_file.getnchannels()}, "
                    f"sample_width={wav_file.getsampwidth()} bytes, "
                    f"sample_rate={frame_rate} Hz, frames={frame_count}, "
                    f"duration={duration:.2f}s"
                ),
            )
    except Exception as error:
        debug_log(debug, f"Could not inspect WAV metadata: {error}")


def play_audio(path: Path, debug: bool = False) -> None:
    """Play an audio file using a system audio player."""
    players = [
        ("mpg123", ["-q", str(path)]),
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]),
        ("vlc", ["--intf", "dummy", "--play-and-exit", str(path)]),
        ("cvlc", ["--play-and-exit", str(path)]),
        ("mpv", ["--no-video", "--really-quiet", str(path)]),
        ("aplay", [str(path)]),
        ("paplay", [str(path)]),
        ("pw-play", [str(path)]),
        ("afplay", [str(path)]),
    ]

    debug_log(debug, f"Target file: {path}")
    debug_log(debug, f"Platform os.name={os.name}")
    debug_log(debug, f"File exists: {path.exists()}")
    if path.exists():
        debug_log(debug, f"File size: {path.stat().st_size} bytes")
    describe_wav(path, debug)

    for command, args in players:
        resolved = shutil.which(command)
        debug_log(debug, f"Checking backend '{command}': {resolved or 'not found'}")
        if resolved:
            cmdline = [command, *args]
            debug_log(debug, f"Running backend command: {cmdline}")
            try:
                subprocess.run(cmdline, check=True)
            except subprocess.CalledProcessError as error:
                debug_log(debug, f"Backend '{command}' failed with exit code {error.returncode}")
                raise RuntimeError(f"Audio backend '{command}' failed with exit code {error.returncode}") from error
            debug_log(debug, f"Backend '{command}' completed successfully")
            return

    if os.name == "nt":
        if path.suffix.lower() != ".wav":
            debug_log(debug, "Using os.startfile for non-WAV playback on Windows")
            os.startfile(path)
            return

        import winsound
        debug_log(debug, "Using winsound.PlaySound on Windows for WAV playback")
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        debug_log(debug, "winsound.PlaySound completed")
        return

    raise RuntimeError(
        "No audio player found. Install one supported backend such as "
        "ffplay, mpv, vlc, or aplay (alsa-utils)."
    )


def play_file(path: str | Path, debug: bool = False) -> None:
    """Play an existing audio file, such as an MP3 or WAV."""
    audio_path = Path(path).expanduser()
    debug_log(debug, f"Resolved audio path: {audio_path}")

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    play_audio(audio_path, debug=debug)


def beep(
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    volume: float = DEFAULT_VOLUME,
    debug: bool = False,
) -> None:
    """Generate and play a beep on the speaker."""
    debug_log(
        debug,
        (
            f"Generating beep: frequency={frequency_hz} Hz, "
            f"duration={duration_seconds}s, volume={volume}"
        ),
    )

    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wav_path = Path(temp_path)
    debug_log(debug, f"Temporary beep path: {wav_path}")

    try:
        create_tone_wav(wav_path, frequency_hz, duration_seconds, volume)
        play_audio(wav_path, debug=debug)
    finally:
        try:
            wav_path.unlink(missing_ok=True)
            debug_log(debug, f"Deleted temporary beep file: {wav_path}")
        except Exception as error:
            debug_log(debug, f"Could not delete temporary beep file: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play an audio file or test beep on the speaker.")
    parser.add_argument("file", nargs="?", help="Audio file to play. Default: piano.wav")
    parser.add_argument("--beep", action="store_true", help="Play a generated test beep instead.")
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY_HZ)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--volume", type=float, default=DEFAULT_VOLUME)
    parser.add_argument("--debug", action="store_true", help="Print backend discovery and playback checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.beep:
            beep(args.frequency, args.duration, args.volume, debug=args.debug)
        elif args.file:
            play_file(args.file, debug=args.debug)
        else:
            play_file(DEFAULT_AUDIO_FILE, debug=args.debug)
    except Exception as error:
        print(f"Could not play sound: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
