from __future__ import annotations

import os
from pathlib import Path
import signal
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
            popen_kwargs: dict[str, object] = {
                "stdout": stdout,
                "stderr": stderr,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            self._process = subprocess.Popen(cmd, **popen_kwargs)
        except Exception as error:
            print(f"[fsm music] Could not start audio '{audio_path.name}': {error}")
            self._process = None

    def stop(self) -> None:
        if self._process is None:
            return

        if self._process.poll() is None:
            try:
                if os.name == "nt":
                    self._process.terminate()
                else:
                    os.killpg(self._process.pid, signal.SIGTERM)
            except Exception:
                try:
                    self._process.terminate()
                except Exception:
                    pass

            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "nt":
                        self._process.kill()
                    else:
                        os.killpg(self._process.pid, signal.SIGKILL)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                self._process.wait(timeout=1.0)
        self._process = None
