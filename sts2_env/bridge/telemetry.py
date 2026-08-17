"""Fire-and-forget milestone telemetry to RabbitMQ, in the proposal's schema.

The companion ``cyra_events`` publisher tells Cyra about a run for commentary;
this one tells ANY downstream service (stream overlay, dashboards, a second
brain) the structural facts of the run: it started, it ended, it died, it beat
a boss. Same discipline, different audience.

The rules are cyra_events' rules, because they were learned the same way:

1. **Publishing never blocks play.** Every event goes onto a bounded queue; a
   background thread owns the broker connection and does the awaiting. A hung
   broker slows nothing, and a full queue drops the event rather than the run.

2. **The broker being absent is normal.** Offline sweeps and simulator runs
   must not need RabbitMQ. Every failure degrades to one log line and a no-op
   publisher, because missing telemetry should cost a dashboard, never a
   measurement.

Events carry the envelope the proposal fixes -- event_type, run_id,
policy_version, git_sha, timestamp -- plus their own fields. game_version is
"unknown" until the mod reports it; the field exists now so downstream
consumers never need a schema change to start receiving it.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import queue
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Overridable without a flag for the same reason CYRA_GAME_PATH is:
#: deployment wiring should not require editing run scripts. The default is
#: the cyra stack's broker on this machine -- same broker, different audience.
DEFAULT_URL = os.environ.get("STS2_TELEMETRY_URL", "amqp://cyra:changeme@127.0.0.1:5672/")
DEFAULT_EXCHANGE = os.environ.get("STS2_TELEMETRY_EXCHANGE", "sts2.events")

_QUEUE_MAX = 256
_SHUTDOWN = object()


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TelemetryPublisher:
    """Publishes run-milestone events to a topic exchange. Safe to construct
    always: without aio_pika or without a reachable broker it disables itself
    and stays callable, every method a no-op."""

    def __init__(self, *, url: str = DEFAULT_URL, exchange: str = DEFAULT_EXCHANGE,
                 enabled: bool = True):
        self._url = url
        self._exchange = exchange
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self.enabled = False
        self.dropped = 0
        if enabled:
            self._try_enable()

    def _try_enable(self) -> None:
        try:
            import aio_pika  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            logger.info("aio_pika unavailable (%s: %s); telemetry disabled.",
                        type(exc).__name__, exc)
            return
        self._thread = threading.Thread(
            target=self._run, name="sts2-telemetry", daemon=True)
        self._thread.start()
        # Enabled optimistically: the background thread owns the broker
        # connection and may discover only there that it is down. Events queued
        # before connect are still correct to send, and dropping them on a dead
        # broker is cyra's exact degradation, not a crash.
        self.enabled = True
        logger.info("telemetry enabled (exchange %s)", self._exchange)

    # -- background broker loop -------------------------------------------

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._drain())
        finally:
            loop.close()

    async def _drain(self) -> None:
        import aio_pika
        import aio_pika.abc

        connection: aio_pika.abc.AbstractRobustConnection | None = None
        channel: aio_pika.abc.AbstractChannel | None = None
        exchange: aio_pika.abc.AbstractExchange | None = None

        while True:
            payload, routing_key = await asyncio.get_event_loop().run_in_executor(
                None, self._queue.get)
            if payload is _SHUTDOWN:
                break
            try:
                if connection is None:
                    connection = await aio_pika.connect_robust(self._url)
                    channel = await connection.channel()
                    exchange = await channel.declare_exchange(
                        self._exchange, aio_pika.ExchangeType.TOPIC, durable=True)
                body = _json_dumps(payload).encode("utf-8")
                await exchange.publish(
                    aio_pika.Message(
                        body=body,
                        content_type="application/json",
                        delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
                    ),
                    routing_key=routing_key,
                )
            except Exception as exc:  # noqa: BLE001
                # Reset the connection; the NEXT event retries. Telemetry that
                # raises into the game loop is worse than telemetry that skips.
                connection = channel = exchange = None
                logger.debug("telemetry publish failed (%s: %s); event dropped.",
                             type(exc).__name__, exc)
        if connection is not None:
            try:
                await connection.close()
            except Exception:  # noqa: BLE001
                pass

    # -- the synchronous seam the game loop sees ---------------------------

    def observe(self, record: dict[str, Any]) -> None:
        """Map one journal record onto zero or more telemetry events."""
        if not self.enabled:
            return
        for event in map_journal_record(record):
            self._enqueue(event)

    def emit(self, event_type: str, **fields: Any) -> None:
        """Publish one event directly (policy_version_loaded, update_detected)."""
        if not self.enabled:
            return
        self._enqueue(_envelope(event_type, **fields))

    def _enqueue(self, event: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((event, f"sts2.run.{event['event_type']}"))
        except queue.Full:
            self.dropped += 1
            logger.debug("telemetry queue full; dropped %s", event.get("event_type"))

    def close(self) -> None:
        if self.enabled and self._thread is not None:
            try:
                self._queue.put_nowait((_SHUTDOWN, ""))
            except queue.Full:
                pass


def _json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, default=str)


def _envelope(event_type: str, **fields: Any) -> dict[str, Any]:
    event = {"event_type": event_type, "timestamp": _utcnow()}
    event.update(fields)
    return event


def map_journal_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The structural events one journal record implies.

    Pure on purpose: the same record can feed other consumers, and the mapping
    is unit-testable without a broker. `run` and `session` give every event a
    run_id that matches the journal, so a downstream service can replay a
    dashboard against the journal that produced it.
    """
    event_type = record.get("event")
    run_id = f"{record.get('session', 'unknown')}_{record.get('run', 0)}"
    base = {
        "run_id": run_id,
        "policy_version": record.get("policy_version", ""),
        "git_sha": record.get("git_sha", ""),
    }
    events: list[dict[str, Any]] = []

    if event_type == "run_start":
        events.append(_envelope("run_start", **base,
                                character=record.get("character"),
                                ascension=record.get("ascension", 0)))
    elif event_type == "policy_version_loaded":
        # policy_version and git_sha arrive in `base`; only the extra field
        # is named here (passing policy_version again would be a duplicate
        # keyword).
        events.append(_envelope("policy_version_loaded", **base,
                                source=record.get("source")))
    elif event_type == "act_clear":
        events.append(_envelope("boss_beaten", **base,
                                act=record.get("act_from"),
                                floor=record.get("floor")))
        events.append(_envelope("act_entered", **base, act=record.get("act_to")))
    elif event_type == "combat_end":
        room = record.get("room_type")
        survived = (record.get("hp_after") or 0) > 0
        if room == "Elite" and survived:
            enemies = [e.get("id") for e in (record.get("enemies") or [])]
            events.append(_envelope("elite_beaten", **base,
                                    floor=record.get("floor"), enemies=enemies))
    elif event_type == "potion_used":
        events.append(_envelope("potion_used", **base,
                                potion=record.get("potion"),
                                floor=record.get("floor")))
    elif event_type == "run_end":
        # The journal's own verdicts: `run_hp <= 0` is a death, `act_cleared`
        # is the win. Field names are the journal's, not the proposal's
        # draft -- mapping from fields that do not exist would emit silently
        # empty telemetry, which is the failure this module exists to avoid.
        run_hp = record.get("run_hp")
        died = isinstance(run_hp, int) and run_hp <= 0
        events.append(_envelope(
            "run_end", **base,
            result=record.get("result"),
            outcome="died" if died else (
                "cleared" if record.get("act_cleared") else "terminated"),
            floor_reached=record.get("floor"),
            act=record.get("act"),
            hp_remaining=run_hp,
            death_enemy_id=record.get("death_enemy_id"),
        ))
        if died:
            events.append(_envelope(
                "died", **base,
                floor_reached=record.get("floor"),
                act=record.get("act"),
                death_cause=record.get("death_enemy_id"),
                room_type=record.get("room_type"),
            ))
    return events
