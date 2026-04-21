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


def play_audio(path: Path) -> None:
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

    for command, args in players:
        if shutil.which(command):
            subprocess.run([command, *args], check=True)
            return

    if os.name == "nt":
        if path.suffix.lower() != ".wav":
            os.startfile(path)
            return

        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return

    raise RuntimeError(
        "No audio player found. On Raspberry Pi, try installing one: "
        "sudo apt install mpg123"
    )


def play_file(path: str | Path) -> None:
    """Play an existing audio file, such as an MP3 or WAV."""
    audio_path = Path(path).expanduser()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    play_audio(audio_path)


def beep(
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    volume: float = DEFAULT_VOLUME,
) -> None:
    """Generate and play a beep on the speaker."""
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "beep.wav"
        create_tone_wav(wav_path, frequency_hz, duration_seconds, volume)
        play_audio(wav_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play an audio file or test beep on the speaker.")
    parser.add_argument("file", nargs="?", help="Audio file to play. Default: piano.wav")
    parser.add_argument("--beep", action="store_true", help="Play a generated test beep instead.")
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY_HZ)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--volume", type=float, default=DEFAULT_VOLUME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.beep:
            beep(args.frequency, args.duration, args.volume)
        elif args.file:
            play_file(args.file)
        else:
            play_file(DEFAULT_AUDIO_FILE)
    except Exception as error:
        print(f"Could not play sound: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
