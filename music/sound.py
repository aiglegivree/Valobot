"""
Simple speaker test.

This script creates a short WAV beep and plays it with the first audio
command available on the computer/Raspberry Pi.
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


def play_wav(path: Path) -> None:
    """Play a WAV file using a system audio player."""
    players = [
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
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return

    raise RuntimeError(
        "No audio player found. On Raspberry Pi, try installing ALSA utilities: "
        "sudo apt install alsa-utils"
    )


def beep(
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    volume: float = DEFAULT_VOLUME,
) -> None:
    """Generate and play a beep on the speaker."""
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "beep.wav"
        create_tone_wav(wav_path, frequency_hz, duration_seconds, volume)
        play_wav(wav_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a test sound on the speaker.")
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY_HZ)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--volume", type=float, default=DEFAULT_VOLUME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        beep(args.frequency, args.duration, args.volume)
    except Exception as error:
        print(f"Could not play sound: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
