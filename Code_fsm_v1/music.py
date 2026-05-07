from __future__ import annotations

from pathlib import Path
import subprocess
import sys


class MusicPlayer:
    def __init__(
        self,
        *,
        enabled: bool,
        backend: str | None = None,
        sink: str | None = None,
        debug: bool = False,
    ) -> None:
        self.enabled = enabled
        self.backend = backend
        self.sink = sink
        self.debug = debug
        self._process: subprocess.Popen[bytes] | None = None
        self._sound_script = Path(__file__).resolve().parents[1] / "music" / "sound.py"
        self._music_dir = self._sound_script.parent

    def _resolve_audio_path(self, filename: str | list[str] | tuple[str, ...]) -> Path | None:
        candidates = [filename] if isinstance(filename, str) else list(filename)
        for candidate in candidates:
            audio_path = self._music_dir / candidate
            if audio_path.exists():
                return audio_path

        if candidates:
            missing_list = ", ".join(str(self._music_dir / candidate) for candidate in candidates)
            print(f"[fsm music] Missing audio file(s): {missing_list}")
        return None

    def play(self, filename: str | list[str] | tuple[str, ...]) -> None:
        if not self.enabled:
            return

        audio_path = self._resolve_audio_path(filename)
        if audio_path is None:
            return

        self.stop()

        cmd = [sys.executable, str(self._sound_script), str(audio_path)]
        if self.backend:
            cmd.extend(["--backend", self.backend])
        if self.sink:
            cmd.extend(["--sink", self.sink])
        if self.debug:
            cmd.append("--debug")

        stdout = None if self.debug else subprocess.DEVNULL
        stderr = None if self.debug else subprocess.DEVNULL

        try:
            self._process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        except Exception as error:
            print(f"[fsm music] Could not start audio '{audio_path.name}': {error}")
            self._process = None

    def stop(self) -> None:
        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.0)
        self._process = None
