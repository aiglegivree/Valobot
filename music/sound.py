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


def run_command(cmdline: list[str], debug: bool, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    debug_log(debug, f"Running command: {cmdline}")
    return subprocess.run(
        cmdline,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


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


def get_players(path: Path) -> list[tuple[str, list[str]]]:
    if os.name == "nt":
        return []

    if os.name == "posix":
        return [
            ("paplay", [str(path)]),
            ("pw-play", [str(path)]),
            ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]),
            ("mpv", ["--no-video", "--really-quiet", str(path)]),
            ("vlc", ["--intf", "dummy", "--play-and-exit", str(path)]),
            ("cvlc", ["--play-and-exit", str(path)]),
            ("aplay", [str(path)]),
            ("mpg123", ["-q", str(path)]),
            ("afplay", [str(path)]),
        ]

    return []


def get_audio_debug_snapshot(debug: bool) -> None:
    if os.name != "posix":
        return

    checks = [
        ["pactl", "list", "short", "sinks"],
        ["pactl", "get-default-sink"],
        ["wpctl", "status"],
    ]
    for cmdline in checks:
        command = cmdline[0]
        if not shutil.which(command):
            debug_log(debug, f"Skipping unavailable diagnostic command: {command}")
            continue
        try:
            result = run_command(cmdline, debug=False)
            output = result.stdout.strip()
            if output:
                debug_log(debug, f"{' '.join(cmdline)} output:\n{output}")
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or "").strip()
            debug_log(debug, f"Diagnostic command failed: {' '.join(cmdline)} :: {stderr}")


def play_audio(path: Path, debug: bool = False, backend: str | None = None, sink: str | None = None) -> None:
    """Play an audio file using a system audio player."""
    players = get_players(path)

    debug_log(debug, f"Target file: {path}")
    debug_log(debug, f"Platform os.name={os.name}")
    debug_log(debug, f"File exists: {path.exists()}")
    if path.exists():
        debug_log(debug, f"File size: {path.stat().st_size} bytes")
    describe_wav(path, debug)
    get_audio_debug_snapshot(debug)

    extra_env = os.environ.copy()
    if sink:
        extra_env["PULSE_SINK"] = sink
        debug_log(debug, f"Using requested sink via PULSE_SINK={sink}")

    for command, args in players:
        if backend is not None and command != backend:
            continue
        resolved = shutil.which(command)
        debug_log(debug, f"Checking backend '{command}': {resolved or 'not found'}")
        if resolved:
            cmdline = [command, *args]
            try:
                result = run_command(cmdline, debug=debug, env=extra_env)
            except subprocess.CalledProcessError as error:
                debug_log(debug, f"Backend '{command}' failed with exit code {error.returncode}")
                if error.stdout:
                    debug_log(debug, f"{command} stdout:\n{error.stdout.strip()}")
                if error.stderr:
                    debug_log(debug, f"{command} stderr:\n{error.stderr.strip()}")
                raise RuntimeError(f"Audio backend '{command}' failed with exit code {error.returncode}") from error
            if result.stdout.strip():
                debug_log(debug, f"{command} stdout:\n{result.stdout.strip()}")
            if result.stderr.strip():
                debug_log(debug, f"{command} stderr:\n{result.stderr.strip()}")
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
        "No compatible audio backend was usable. Try --backend paplay, "
        "--backend pw-play, or install one supported player."
    )


def play_file(
    path: str | Path,
    debug: bool = False,
    backend: str | None = None,
    sink: str | None = None,
) -> None:
    """Play an existing audio file, such as an MP3 or WAV."""
    audio_path = Path(path).expanduser()
    debug_log(debug, f"Resolved audio path: {audio_path}")

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    play_audio(audio_path, debug=debug, backend=backend, sink=sink)


def beep(
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    volume: float = DEFAULT_VOLUME,
    debug: bool = False,
    backend: str | None = None,
    sink: str | None = None,
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
        play_audio(wav_path, debug=debug, backend=backend, sink=sink)
    finally:
        try:
            wav_path.unlink(missing_ok=True)
            debug_log(debug, f"Deleted temporary beep file: {wav_path}")
        except Exception as error:
            debug_log(debug, f"Could not delete temporary beep file: {error}")


def list_sinks(debug: bool = False) -> int:
    if os.name != "posix":
        print("Sink listing is only supported on Linux/PipeWire/PulseAudio systems.")
        return 1

    if shutil.which("pactl"):
        result = run_command(["pactl", "list", "short", "sinks"], debug=debug)
        print(result.stdout.rstrip())
        return 0

    if shutil.which("wpctl"):
        result = run_command(["wpctl", "status"], debug=debug)
        print(result.stdout.rstrip())
        return 0

    print("Neither pactl nor wpctl is available.")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play an audio file or test beep on the speaker.")
    parser.add_argument("file", nargs="?", help="Audio file to play. Default: piano.wav")
    parser.add_argument("--beep", action="store_true", help="Play a generated test beep instead.")
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY_HZ)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--volume", type=float, default=DEFAULT_VOLUME)
    parser.add_argument("--backend", choices=["paplay", "pw-play", "ffplay", "mpv", "vlc", "cvlc", "aplay", "mpg123", "afplay"])
    parser.add_argument("--sink", help="Optional Pulse/PipeWire sink name or id, for example 94 or bluez_output....")
    parser.add_argument("--list-sinks", action="store_true", help="Print available Pulse/PipeWire sinks and exit.")
    parser.add_argument("--debug", action="store_true", help="Print backend discovery and playback checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.list_sinks:
            return list_sinks(debug=args.debug)
        if args.beep:
            beep(
                args.frequency,
                args.duration,
                args.volume,
                debug=args.debug,
                backend=args.backend,
                sink=args.sink,
            )
        elif args.file:
            play_file(args.file, debug=args.debug, backend=args.backend, sink=args.sink)
        else:
            play_file(DEFAULT_AUDIO_FILE, debug=args.debug, backend=args.backend, sink=args.sink)
    except Exception as error:
        print(f"Could not play sound: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
