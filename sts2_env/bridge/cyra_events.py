"""Tell Cyra about the run, without ever making her a dependency of it.

cyra_game owns the broker connection, the salience rules and the wording, so this
imports that code rather than reimplementing any of it. What lives here is the
seam: finding cyra_game, and getting an async publish out of a synchronous game
loop without the game loop ever waiting on a message broker.

Two rules this file exists to keep:

1. **Publishing never blocks play.** The agent loop is synchronous and the game is
   blocked on its answer while we think. A publish that waited on RabbitMQ would
   add broker latency to every decision, and a broker that hung would hang the
   game. So messages go onto a queue and a background thread does the awaiting.

2. **Cyra being absent is normal.** Training and simulator evaluation must not
   need a broker, a cyra checkout, or aio_pika. Every failure here degrades to a
   no-op with one log line, because a missing Cyra should cost you commentary,
   never the run you were measuring.

Point CYRA_GAME_PATH at the cyra_game directory if the repos are not siblings.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Siblings by default: .../development/sts2-rl-agent and .../development/cyra
_DEFAULT_CYRA_GAME = Path(__file__).resolve().parents[2].parent / "cyra" / "cyra_game"

_QUEUE_MAX = 64
_SHUTDOWN = object()


def _find_cyra_game() -> Path | None:
    raw = os.environ.get("CYRA_GAME_PATH")
    candidate = Path(raw).expanduser() if raw else _DEFAULT_CYRA_GAME
    return candidate if (candidate / "events.py").is_file() else None


class CyraPublisher:
    """Fire-and-forget publishing to cyra_brain. Safe to construct always.

    `enabled` is False when cyra_game cannot be found or imported, and every
    method stays callable so no caller needs to branch on it.
    """

    def __init__(self, enabled: bool = True):
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._publish_payload = None
        self.enabled = False
        if enabled:
            self._try_enable()

    def _try_enable(self) -> None:
        path = _find_cyra_game()
        if path is None:
            logger.info(
                "cyra_game not found (set CYRA_GAME_PATH); running without Cyra.")
            return

        # cyra_game uses flat imports ("import config", "from salience import
        # Tier"), so its own directory has to be on sys.path -- importing it as a
        # package would fail on those.
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

        try:
            import events  # type: ignore
        except Exception as exc:  # noqa: BLE001
            # aio_pika missing, or no .env. Not an error for a training run.
            logger.info("cyra_game found but not importable (%s: %s); "
                        "running without Cyra.", type(exc).__name__, exc)
            return

        self._publish_payload = events.publish_payload
        self.enabled = True
        self._thread = threading.Thread(
            target=self._run, name="cyra-publisher", daemon=True)
        self._thread.start()
        logger.info("Cyra events enabled (via %s)", path)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._drain())
        finally:
            loop.close()

    async def _drain(self) -> None:
        while True:
            payload = await asyncio.get_event_loop().run_in_executor(
                None, self._queue.get)
            if payload is _SHUTDOWN:
                return
            try:
                await self._publish_payload(payload, "cyra.game.event")
            except Exception as exc:  # noqa: BLE001
                # publish_payload already swallows broker errors; this is the
                # backstop for anything it does not.
                logger.debug("Cyra publish failed (%s): %s", type(exc).__name__, exc)

    def publish(self, payload: dict) -> None:
        """Queue one event. Returns immediately, always."""
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            # Dropping is correct: stale commentary is worse than none, and the
            # run must not slow down to let Cyra catch up.
            logger.debug("Cyra queue full, dropping: %s", payload.get("text"))

    def close(self) -> None:
        if self.enabled and self._thread is not None:
            try:
                self._queue.put_nowait(_SHUTDOWN)
            except queue.Full:
                pass
