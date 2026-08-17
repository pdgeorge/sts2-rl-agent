# Reviewing a session with a local model

Read all 100 runs of a session, one at a time, and turn them into one ranked
report: which mistakes recur, what they cost, and which individual runs are
worth opening yourself.

The pipeline is four steps and only the third needs a GPU.

```
build_review_context.py    ->  output/review_context.md   (once, ~7k tokens)
run100.sh <tag>            ->  journal + transcripts      (transcripts are automatic)
review_runs.py --tag <tag> ->  output/reports/<tag>/run_NNN.json
aggregate_run_reports.py   ->  the ranked report
```

## 0. The game reference, once

```bash
.venv/bin/python scripts/build_review_context.py
```

Writes `output/review_context.md` -- about 7k tokens: how act 1 works, how the
agent decides, how to read a transcript, and **the real behaviour of all 184
cards an Ironclad run can hold plus every enemy it can meet**, all generated
from the game's own source.

That file is the system prompt on every review call. It is deliberately large
and it is effectively free: llama.cpp caches an identical prompt prefix, so
7k tokens of reference are evaluated **once** and reused for the other 99 runs.
A short system prompt with a per-run card list would pay unique tokens on every
call and tell the model less.

Regenerate it after a game update; it is derived, not written.

---

## 1. Serving the model

Nothing in these scripts is Qwen-specific. `review_runs.py` speaks
OpenAI-compatible `/v1/chat/completions`, so llama-server, Ollama, vLLM and LM
Studio all work — point `--base-url` at whichever is running.

### Bare metal, and on this box that is not a close call

`nvcc` (CUDA 13.3), `cmake` and the driver are already installed, so the build
is the whole cost. Docker would additionally need its GPU path repaired: as of
2026-08-17 `docker run --gpus all` fails here with

```
failed to fulfil mount request: open /usr/lib/libnvidia-gtk3.so.610.57.04:
no such file or directory
```

— the container toolkit's library list is stale against driver 610.57.04. That
is fixable (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`, then
`--device nvidia.com/gpu=all`) but it is a yak-shave in front of a job that has
no reason to be containerised. Bare metal also makes `-ngl`, context size and
KV quantisation trivial to retune, which is exactly what you will be doing.

```bash
# Build. ~5 minutes.
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build ~/llama.cpp/build --config Release -j

# The model: one file, 4.7 GB. The mmproj-*.gguf files in that repo are the
# vision projector and are NOT needed -- this job is pure text, and skipping
# them saves VRAM.
hf download unsloth/Qwen3-VL-8B-Instruct-GGUF \
    --include "Qwen3-VL-8B-Instruct-Q4_K_M.gguf" \
    --local-dir ~/models/qwen3-vl-8b

# Serve.
~/llama.cpp/build/bin/llama-server \
    --model ~/models/qwen3-vl-8b/Qwen3-VL-8B-Instruct-Q4_K_M.gguf \
    --ctx-size 40960 \
    --n-gpu-layers 99 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --host 127.0.0.1 --port 8080
```

`-DCMAKE_CUDA_ARCHITECTURES=89` is the 4070 SUPER (Ada, sm_89). Naming it skips
building every other architecture and cuts the build time by most of itself.

**The context size is not a guess, and 16384 is too small.** Measured against a
real session:

| | tokens |
|---|---|
| system prompt (the generated reference) | 7.0k |
| transcript, mean | 8.6k |
| transcript, **largest** | **23.5k** |
| reply | up to 2.0k |
| **worst case, with one retry** | **~34.7k** |

The longest transcripts are the deepest runs, which are the ones most worth
reviewing, so truncating them is the opposite of what you want. 40960 covers
the worst case with headroom.

`--cache-type-k/v q8_0` is what makes that fit. At 40k context an f16 KV cache
is ~5.9 GB, which on top of a 4.7 GB model leaves almost nothing of the 12 GB;
quantised to q8_0 it is ~3 GB and the whole thing sits around 7.7 GB. Watch
`nvidia-smi` on the first call and drop `--ctx-size` if it is tight.

### Ollama, if you would rather

```bash
ollama serve                       # then, in another shell:
printf 'FROM ~/models/qwen3-vl-8b/Qwen3-VL-8B-Instruct-Q4_K_M.gguf\nPARAMETER num_ctx 40960\n' > /tmp/Modelfile
ollama create qwen3-review -f /tmp/Modelfile
```

then `--base-url http://127.0.0.1:11434/v1 --model qwen3-review`.

Note the cyra compose already defines an `ollama` service with the GPU
reserved, but it is deliberately not published to the host, so it needs a
`ports:` entry — and it would hit the same broken container-GPU path described
above.

### A note on the model choice

`Qwen3-VL-8B` is the vision-language variant. It handles text fine, but for a
pure text-reasoning job the plain **`Qwen3-8B`** is usually the stronger pick
and the same size. Worth trying both on five runs before committing to a
hundred — the trial below costs about three minutes.

---

## 2. Trial it on five runs before spending an hour

```bash
.venv/bin/python scripts/review_runs.py --tag <tag> --limit 5
```

Then actually read `output/reports/<tag>/run_001.json`. The three things that
decide whether the full pass is worth running:

- **Does it hold the JSON contract?** Malformed replies are retried with the
  error fed back, then kept in `reports/<tag>/failed/`. A few failures are
  normal; half of them means the prompt needs work, not the model.
- **Can it tell 0.002 from 0.1?** The transcript states outright that a
  0.004 gap is a coin flip. If it flags near-ties as blunders, its claims will
  swamp the aggregate with noise.
- **Does it use the card reference?** Every transcript opens with the real
  behaviour of every card and enemy in that run, generated from the decompile.
  An 8B has never seen this game — the names overlap Slay the Spire 1 and the
  effects do not — so a review that describes a card wrongly is reviewing a
  game it invented, and nothing downstream of that is worth reading.

If the small model cannot manage the judgement call, narrow the ask rather than
abandoning it. Something mechanical — *"list every turn where a passed line
scored within 0.05 of the played one and contained a kill"* — is well within a
7B and still surfaces the pattern.

## 3. The full pass

```bash
.venv/bin/python scripts/review_runs.py --tag <tag>
```

Resumable: each reply is written as it arrives and an existing `run_NNN.json` is
skipped, so a Ctrl-C or a server restart costs only the run in flight. Re-run
the same command to continue.

## 4. The report

```bash
.venv/bin/python scripts/aggregate_run_reports.py --tag <tag>
```

Three sections, and they answer different questions:

- **RECURRING** — ranked by `claims x claimed HP`, so a small mistake made
  constantly outranks a dramatic one made twice. This is where a change comes
  from. A claim in 40 runs is worth a paired A/B; the same claim in 2 runs is a
  reviewer noticing something once.
- **MOST EGREGIOUS** — which transcripts to open yourself.
- **GOT FURTHEST** — the runs that went deepest, and what the reviewer thought
  they did right.

Claims are discarded if they carry no floor, or land on a floor the run never
reached (checked against the journal's own `run_end`). Nothing else is
filtered. In particular a reviewer saying *the evaluator preferred the worse
line* is **not** treated as an error: prediction 10 measured that more search
does not help, so if anything is left in the fight it is in the scoring, and
that disagreement is the most interesting thing a review can produce.

---

## What this is for, and what it is not

It generates **leads**, not conclusions. `SCOREBOARD.md`'s rule is unchanged: a
lead becomes a prediction written down before the change is built, and then a
live measurement of 100+ runs decides it. Six false positives came out of this
project's own harnesses in a single day, and a model asked "what went wrong"
will always find something.

The aggregate exists precisely so the question becomes *"does this happen in
forty runs"* rather than *"does this sound plausible"*.

## While a session is running

Don't. The live searcher is **time-budgeted** — measured on a real boss turn it
explores 609 of ~1447 nodes inside its 3 seconds — so anything competing for
CPU or GPU makes the agent under-search and changes the number the session is
measuring. Review yesterday's session, not today's.
