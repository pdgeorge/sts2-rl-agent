# Reviewing a session with a local model

Read all 100 runs of a session, one at a time, and turn them into one ranked
report: which mistakes recur, what they cost, and which individual runs are
worth opening yourself.

The pipeline is four steps and only the third needs a GPU.

```
run100.sh <tag>            ->  journal + transcripts   (transcripts are automatic)
review_runs.py --tag <tag> ->  output/reports/<tag>/run_NNN.json
aggregate_run_reports.py   ->  the ranked report
```

---

## 1. Serving the model

Nothing in these scripts is Qwen-specific. `review_runs.py` speaks
OpenAI-compatible `/v1/chat/completions`, so llama-server, Ollama, vLLM and LM
Studio all work — point `--base-url` at whichever is running.

### llama.cpp (most direct for a GGUF off Hugging Face)

```bash
# One-off: build it. ~5 minutes on this box.
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON
cmake --build ~/llama.cpp/build --config Release -j

# The model. Q4_K_M is ~5 GB and leaves room for a long context on 12 GB.
hf download unsloth/Qwen3-VL-8B-Instruct-GGUF \
    --include "*Q4_K_M*" --local-dir ~/models/qwen3-vl-8b

# Serve it. -ngl 99 puts every layer on the 4070.
~/llama.cpp/build/bin/llama-server \
    -m ~/models/qwen3-vl-8b/*Q4_K_M*.gguf \
    -c 16384 -ngl 99 --host 127.0.0.1 --port 8080
```

`-c 16384` matters. A transcript is ~9k tokens and the reply needs room; 16k
leaves headroom for the retry exchange, which appends the failed reply and an
error message to the conversation.

### Ollama, if you would rather

```bash
ollama serve                       # then, in another shell:
printf 'FROM ~/models/qwen3-vl-8b/Qwen3-VL-8B-Instruct-Q4_K_M.gguf\nPARAMETER num_ctx 16384\n' > /tmp/Modelfile
ollama create qwen3-review -f /tmp/Modelfile
```

then `--base-url http://127.0.0.1:11434/v1 --model qwen3-review`.

Note the cyra compose already defines an `ollama` service with the GPU
reserved, but it is deliberately not published to the host, so it needs a
`ports:` entry before this can reach it.

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
