"""Send each run's transcript to a local model and collect its review.

    .venv/bin/python scripts/review_runs.py --tag boss_telemetry

Talks to anything with an OpenAI-compatible `/v1/chat/completions`: llama-server,
Ollama, vLLM, LM Studio. Nothing about this is Qwen-specific; point `--base-url`
at whatever is serving.

RESUMABLE, BECAUSE 100 RUNS IS AN HOUR
--------------------------------------
Every reply lands on disk as it arrives and an existing `run_NNN.json` is
skipped, so a Ctrl-C, an OOM or a server restart costs the run in flight and
nothing else. Re-running the command continues where it stopped.

THE REPLY HAS TO PARSE, AND THAT IS RETRIED
-------------------------------------------
A small model wanders off the JSON contract fairly often. A reply that does not
parse is retried with the parse error fed back to it, up to `--retries`, and
only then written to `failed/` -- kept rather than dropped, because a directory
of malformed replies is how you find out the prompt is wrong rather than the
model being stupid.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
Judge the reviews. `aggregate_run_reports.py` ranks and cross-checks them
against the journal; this file only gets the text and validates its shape. A
script that both collects opinions and decides which are true is one that can
quietly agree with itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Appended to the generated game reference. The reference explains the game;
#: this states the job and the output contract.
TASK = """

---

# Your task

You are given one complete run: every fight turn by turn, every card played,
the lines the agent considered and REJECTED with their scores, and every
reward, shop, rest and map choice.

Find where the run was decided. Say what should have been done instead, and be
concrete about which floor and which turn.

Reply with a single JSON object and nothing else. The placeholders show the
SHAPE; fill them from this run and nothing else:

```json
{
  "run": <number>,
  "outcome": "<how the run ended>",
  "summary": "<one or two sentences on what decided this run>",
  "mistakes": [
    {"floor": <n>, "turn": <n>,
     "did": "<what she actually played, from the transcript>",
     "better": "<what to play instead, and why>",
     "cost_hp": <n>,
     "kind": "<one of the kinds below>",
     "confidence": "high|medium|low"}
  ],
  "good_plays": ["<floor N: something this run genuinely did well>"]
}
```

`kind` is one of: combat, card_reward, map, shop, rest, potion.
`confidence` is one of: high, medium, low.

Every mistake MUST carry a `floor`, and a combat mistake a `turn` -- a claim
without a location cannot be checked and will be discarded, as is any claim on
a floor the run never reached. Quote only cards the transcript shows her
playing on that turn.

**Do not review only the fights.** Card rewards, shop purchases, rest sites and
map choices are all in the transcript and all decide runs -- a deck that cannot
kill the boss was assembled in the corridor. If the deckbuilding and routing
were sound, say so; if a reward was taken that never got played, or a rest spent
upgrading at low HP, that is a mistake worth more than most combat lines. If the run was
already lost by the time of a mistake, say so in `summary` rather than listing
ten consequences of one earlier error. An empty `mistakes` list is a valid and
useful answer."""

DEFAULT_CONTEXT = REPO / "output" / "review_context.md"


def _system_prompt(path: Path) -> str:
    """The generated game reference, plus the task.

    Identical on every call on purpose: llama.cpp caches a matching prompt
    prefix, so 7k tokens of card and monster reference are evaluated once and
    reused for the other 99 runs. The alternative -- a short system prompt and
    a per-run card list -- pays unique tokens on every single call AND tells
    the model less.
    """
    if not path.exists():
        raise SystemExit(
            f"missing {path}. Generate it first:\n"
            f"  .venv/bin/python scripts/build_review_context.py\n"
            f"Without it an 8B reviews a game it invented -- the card names "
            f"overlap Slay the Spire 1 and the effects do not.")
    return path.read_text(encoding="utf-8") + TASK


def _post(url: str, payload: dict, timeout: float, api_key: str | None) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read().decode("utf-8"))


def _extract(reply: str) -> dict | None:
    from scripts.aggregate_run_reports import _load_json
    return _load_json(reply)


def _valid(data: dict) -> str | None:
    """None if the shape is usable, else what is wrong with it."""
    if not isinstance(data, dict):
        return "top level is not an object"
    if "mistakes" not in data:
        return "missing the required key 'mistakes' (use [] if there are none)"
    if not isinstance(data["mistakes"], list):
        return "'mistakes' must be a list"
    for m in data["mistakes"]:
        if not isinstance(m, dict):
            return "each entry in 'mistakes' must be an object"
        if m.get("floor") is None:
            return "every mistake needs a 'floor'; a claim without a location cannot be checked"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1",
                    help="llama-server default. Ollama is http://127.0.0.1:11434/v1")
    ap.add_argument("--model", default="local", help="ignored by llama-server")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="review only the first N, for a trial")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="low on purpose: this is analysis, not prose")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--redo", action="store_true", help="re-review runs already done")
    ap.add_argument("--context", default=None,
                    help="generated game reference used as the system prompt; "
                         "defaults to output/review_context.md")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))

    tdir = Path(args.transcripts or f"output/transcripts/{args.tag}")
    odir = Path(args.out or f"output/reports/{args.tag}")
    faildir = odir / "failed"
    odir.mkdir(parents=True, exist_ok=True)
    faildir.mkdir(parents=True, exist_ok=True)

    files = sorted(tdir.glob("run_*.md"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"no transcripts in {tdir}. Run:\n"
              f"  .venv/bin/python scripts/export_run_transcripts.py --tag {args.tag}")
        return 1

    system = _system_prompt(Path(args.context) if args.context else DEFAULT_CONTEXT)
    url = args.base_url.rstrip("/") + "/chat/completions"
    print(f"{len(files)} transcripts -> {url}")
    print(f"system prompt {len(system) / 4 / 1000:.1f}k tokens "
          f"(cached after the first call)\n")

    done = failed = skipped = 0
    started = time.monotonic()
    for i, path in enumerate(files, 1):
        target = odir / (path.stem + ".json")
        if target.exists() and not args.redo:
            skipped += 1
            continue

        transcript = path.read_text(encoding="utf-8")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": transcript},
        ]

        parsed, last_error, reply = None, None, ""
        for attempt in range(args.retries + 1):
            try:
                resp = _post(url, {
                    "model": args.model, "messages": messages,
                    "temperature": args.temperature, "max_tokens": args.max_tokens,
                }, args.timeout, args.api_key)
                reply = resp["choices"][0]["message"]["content"]
            except (urllib.error.URLError, OSError, KeyError, TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"  [{i}/{len(files)}] {path.name}: {last_error}")
                break

            candidate = _extract(reply)
            problem = "the reply was not JSON" if candidate is None else _valid(candidate)
            if problem is None:
                parsed = candidate
                break
            last_error = problem
            # Feed the error back rather than just re-rolling: a small model
            # usually fixes a named contract violation on the second pass.
            messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content":
                    f"That reply could not be used: {problem}. "
                    f"Reply again with only the JSON object."},
            ]

        elapsed = time.monotonic() - started
        if parsed is None:
            failed += 1
            (faildir / (path.stem + ".txt")).write_text(
                f"# {last_error}\n\n{reply}", encoding="utf-8")
            print(f"  [{i}/{len(files)}] {path.name}: FAILED ({last_error})")
            continue

        parsed.setdefault("run", int(path.stem.split("_")[1]))
        target.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        done += 1
        rate = elapsed / max(1, done)
        left = (len(files) - i) * rate
        print(f"  [{i}/{len(files)}] {path.name}: "
              f"{len(parsed.get('mistakes') or [])} mistakes  "
              f"({rate:.0f}s/run, ~{left / 60:.0f} min left)")

    print(f"\n{done} reviewed, {skipped} already done, {failed} failed")
    if failed:
        print(f"  malformed replies kept in {faildir} -- read a couple before "
              f"blaming the model; usually it is the prompt.")
    print(f"\nNow:  .venv/bin/python scripts/aggregate_run_reports.py --tag {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
