from __future__ import annotations

import argparse
import atexit
import signal
import time

from corridor_config import AURA_FARM_DELAY, USE_LEDS
from fsm_config import (
    END_MUSIC_FALLBACK,
    END_MUSIC_FILE,
    MUSIC_BACKEND,
    MUSIC_DEBUG,
    MUSIC_SINK,
    USE_MUSIC,
)
from leds import LED
from motors import MotorDriver
from music import MusicPlayer


class AuraFarmLoop:
    def __init__(
        self,
        *,
        step_delay: float = AURA_FARM_DELAY,
        tick_sec: float = 0.01,
        use_leds: bool = USE_LEDS,
        use_music: bool = USE_MUSIC,
    ) -> None:
        self.step_delay = max(0.001, float(step_delay))
        self.tick_sec = max(0.001, float(tick_sec))
        self._running = False
        self._music_selection = (END_MUSIC_FILE, END_MUSIC_FALLBACK)

        self.leds = LED() if use_leds else None
        self.motors = MotorDriver()
        self.music_player = MusicPlayer(
            enabled=use_music,
            backend=MUSIC_BACKEND,
            sink=MUSIC_SINK,
            debug=MUSIC_DEBUG,
        )

    def _ensure_end_music_looping(self) -> None:
        process = self.music_player._process
        if process is None or process.poll() is not None:
            self.music_player.play(self._music_selection)

    def start(self) -> None:
        self._running = True
        self.motors.stop()
        if self.leds is not None:
            self.leds.aura_farm_start()
        self._ensure_end_music_looping()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.cleanup()

    def run_forever(self) -> None:
        self.start()
        while self._running:
            self.motors.stop()
            if self.leds is not None:
                self.leds.aura_farm_step(delay=self.step_delay)
            self._ensure_end_music_looping()
            time.sleep(self.tick_sec)

    def cleanup(self) -> None:
        try:
            self.motors.stop()
        except Exception:
            pass
        try:
            self.music_player.stop()
        except Exception:
            pass
        try:
            if self.leds is not None:
                self.leds.aura_farm_stop()
        except Exception:
            pass
        try:
            self.motors.cleanup()
        except Exception:
            pass
        try:
            if self.leds is not None:
                self.leds.cleanup()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone aura farm mode with looping end music.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=AURA_FARM_DELAY,
        help="Delay between LED aura steps in seconds.",
    )
    parser.add_argument(
        "--tick-sec",
        type=float,
        default=0.01,
        help="Main loop sleep in seconds.",
    )
    parser.add_argument(
        "--no-leds",
        action="store_true",
        help="Disable the LED aura animation.",
    )
    parser.add_argument(
        "--no-music",
        action="store_true",
        help="Disable looping end music.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = AuraFarmLoop(
        step_delay=args.step_delay,
        tick_sec=args.tick_sec,
        use_leds=USE_LEDS and not args.no_leds,
        use_music=USE_MUSIC and not args.no_music,
    )

    def _handle_shutdown(*_args) -> None:
        runner.stop()

    atexit.register(runner.cleanup)
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        runner.run_forever()
    finally:
        runner.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
