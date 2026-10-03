# localllmtest

**Find out whether a local LLM is actually good enough for your real work, before you rely on it.**

`model_eval.py` runs a suite of 13 practical tests against one or more models served by [Ollama](https://ollama.com), scores each one automatically, and writes a side by side comparison report. The tests are modelled on everyday knowledge work: writing in a house style, rewriting resume bullets without inventing facts, steering committee updates, invoice maths, structured JSON for automation pipelines, Python coding, translation, long documents, hallucination traps, over refusal and raw speed on your own hardware.

Public leaderboards tell you how a model does on benchmarks. This tells you how it does on *your* jobs, on *your* machine.

Built by [Company31](https://company31.com).

---

## Contents

* [Quick start](#quick-start)
* [Requirements](#requirements)
* [Installing Ollama and models](#installing-ollama-and-models)
* [Running the suite](#running-the-suite)
* [The tests](#the-tests)
* [Scoring and verdicts](#scoring-and-verdicts)
* [Reading the report](#reading-the-report)
* [Choosing models for your hardware](#choosing-models-for-your-hardware)
* [Uncensored (abliterated) models](#uncensored-abliterated-models)
* [Customising the suite](#customising-the-suite)
* [Troubleshooting](#troubleshooting)
* [Limitations and safety notes](#limitations-and-safety-notes)

---

## Quick start

```powershell
# 1. Ollama running, models pulled
ollama pull qwen3:8b
ollama pull gemma4:12b

# 2. Run the suite
py model_eval.py qwen3:8b gemma4:12b

# 3. Open the report it prints, e.g. results\20261003_1830\report.md
```

On macOS or Linux use `python3` instead of `py`.

---

## Requirements

| What | Version | Notes |
|---|---|---|
| Python | 3.9 or newer | Standard library only. Nothing to `pip install`. |
| Ollama | Recent release | Gemma 4 models need Ollama 0.22 or newer. |
| A GPU | Optional | Works on CPU, just slowly. 8 GB of VRAM is enough for 7B to 12B models. |

The script talks to Ollama's local HTTP API (default `http://localhost:11434`). It works on Windows, macOS and Linux.

---

## Installing Ollama and models

### Windows

1. Download and run the installer from [ollama.com/download](https://ollama.com/download). It runs as a tray app.
2. Keep your NVIDIA driver up to date; Ollama uses CUDA automatically.
3. Check the install:

   ```powershell
   ollama --version
   ```

4. Optional: store models on another drive by setting the user environment variable `OLLAMA_MODELS`, for example `D:\ollama`, then restart Ollama.

### Python on Windows

If `py` is not recognised:

```powershell
winget install Python.Python.3.14 --source winget
```

Close and reopen your terminal, then check with `py --version`.

### Pulling models

```powershell
ollama pull qwen3:8b
ollama list          # what is installed
ollama ps            # what is loaded, and whether it is on GPU or CPU
ollama rm <model>    # free disk space
```

---

## Running the suite

```text
py model_eval.py [models ...] [--only KEYS] [--host URL] [--out DIR] [--think on|off] [--list]
```

| Option | What it does |
|---|---|
| `models` | One or more Ollama model tags. Each is tested in turn and compared in the report. |
| `--only` | Comma separated test keys, to run a subset. Example: `--only coding,coding_fix,speed` |
| `--host` | Ollama URL. Defaults to the `OLLAMA_HOST` environment variable, then `http://localhost:11434`. |
| `--out` | Folder for results. Default `results`. A timestamped subfolder is created per run. |
| `--think` | Force thinking mode `on` or `off` for reasoning models that support it. Default leaves the model's own setting. |
| `--list` | Print the test keys and exit. |

### Examples

```powershell
# Compare three models on everything
py model_eval.py gemma4:12b qwen3:8b huihui_ai/gemma-4-abliterated:12b

# Quick coding and speed check of one model
py model_eval.py qwen3:8b --only coding,coding_fix,speed

# Test a model served by Ollama on another machine
py model_eval.py qwen3:8b --host http://192.168.1.50:11434

# Reasoning model with thinking switched off (faster, often better at strict formats)
py model_eval.py qwen3:8b --think off
```

Before anything runs, the script checks that Ollama is reachable and that every model you named is pulled, so a typo fails in one second rather than after an hour.

### How long it takes

Roughly 10 to 30 minutes per model on an RTX 4060 class GPU, depending on model size and how much it rambles. Models that spill out of VRAM into system RAM are much slower. Start a multi model run and come back to it.

---

## The tests

Every test sends a realistic prompt, then grades the answer with automatic checks. A test's score is the percentage of its checks that passed.

| Key | Test | What is checked |
|---|---|---|
| `linkedin` | LinkedIn post in a defined personal style | 130 to 280 words, at least 5 emojis, a short hook as the first line, ends with a question, **no dashes** used as punctuation, **Australian spelling** |
| `resume` | Rewrite a paragraph into 3 resume bullets | Exactly 3 bullets, **no invented numbers**, key facts kept, no dashes, Australian spelling |
| `steerco` | Messy program notes into a steering committee update | Under 170 words, has a RAG status, status is Amber or Red (the project is over budget and behind), mentions the forecast overrun, states the decision needed, flags the key person risk, style rules |
| `invoice` | Australian tax invoice maths | Valid JSON with correct subtotal, 10% GST and total to the cent |
| `actions_json` | Meeting notes into action items | Valid JSON list, exactly 4 actions, correct owners, correct ISO due dates, a group decision not misfiled as an action |
| `coding` | Write a Python function from a spec | The generated code is **executed against hidden unit tests** (Australian mobile number normaliser, 9 cases including invalid inputs) |
| `coding_fix` | Find and fix bugs in a function | The fixed code is **executed against hidden tests**, plus both bugs are explained |
| `agent_json` | Strict JSON for an automation pipeline | Raw JSON only (no fences or chatter), all schema keys, correct ISO 8601 time with the Sydney offset, valid enum value, tag rules, integer type |
| `languages` | Translate a business message to Italian and Mexican Spanish | Valid JSON, both languages present and recognisable, time preserved, no English left over |
| `long_context` | Find one fact buried in a ~12k token document | The whole prompt was actually read (not silently truncated), correct date and approver found |
| `hallucination` | Ask about an agreement that does not exist | Admits it does not know or doubts it exists, and does not list invented provisions |
| `refusal` | Five legitimate but edgy requests | Answers each without refusing: a comedic roast, a phishing example for staff training, a novel villain monologue, scam tactics to warn elderly parents, brutally honest career feedback |
| `speed` | A 300 word explanation | Generation speed of at least 15 tok/s (comfortable) and 8 tok/s (usable), first token in under 5 seconds |

### Why these tests

* **Style rules** (no dashes, Australian English) are where small models slip first. They follow the gist but not the constraint. If a model cannot follow a stated rule, every draft needs fixing by hand.
* **No fabrication** checks matter most for anything that goes out under your name. The resume test fails a model that adds a number that was not in the source.
* **JSON tests** decide whether a model can sit inside an agent or automation without breaking the parser.
* **Coding tests run the code.** Plausible looking code that does not work scores zero.
* **The long context test** catches a common silent failure: Ollama truncating the prompt to the default context window so the model never sees the answer.
* **Hallucination and refusal** are two sides of trust. A good model says "I don't know" to fake facts, and does not refuse ordinary work because it contains a sharp word.

---

## Scoring and verdicts

Each model gets a **weighted overall score**. Tests that matter most for day to day use count for more:

| Weight | Tests |
|---|---|
| 1.5 | `linkedin`, `resume`, `steerco`, `coding`, `coding_fix`, `hallucination` |
| 1.0 | `invoice`, `actions_json`, `agent_json`, `languages`, `long_context`, `refusal`, `speed` |

| Overall | Verdict | Meaning |
|---|---|---|
| 85% and above | **Daily driver** | Trust it for most routine work |
| 70 to 84% | **Good with review** | Useful, but check its output |
| 50 to 69% | **Niche use only** | Pick the specific tests it passed and use it only for those |
| Below 50% | **Not good enough** | Stick with a hosted model |

When you run only a subset with `--only`, the overall score is weighted across just those tests.

---

## Reading the report

Each run writes to `results\<YYYYMMDD_HHMM>\`:

| File | Contents |
|---|---|
| `report.md` | Summary table comparing every model on every test, median speed, weighted score and verdict, then a section per model with every check marked PASS or FAIL, timing stats and the full output |
| `results.json` | The same data in machine readable form, for your own charts or tracking models over time |

To read the report nicely in VS Code: open the folder, open `report.md`, press **Ctrl + Shift + V**.

**Automatic checks are not the whole story.** They catch format, facts, maths, broken code and refusals. They cannot judge whether a LinkedIn post is actually engaging or whether a status update reads well to a steering committee. Each test in the report has a `your rating: __/5` slot so you can add a human judgement alongside the score.

---

## Choosing models for your hardware

The deciding factor is **VRAM**. A model that fits entirely on the GPU is fast. One that spills into system RAM slows down sharply. Check with `ollama ps`: you want `100% GPU`.

| GPU memory | Comfortable size | Examples to try |
|---|---|---|
| 8 GB | 7B to 12B at Q4 | `qwen3:8b`, `gemma4:12b` (right at the limit) |
| 12 to 16 GB | 14B to 20B | `gemma4:12b` with long context, 14B class models |
| 24 GB | 27B to 32B | Larger Qwen and Gemma models |

**Plenty of system RAM helps with mixture of experts (MoE) models.** Models like `gemma4:26b` (about 4B active parameters) or `qwen3-coder:30b` (about 3B active) only use a fraction of their weights per token, so they stay usable even when partly offloaded to RAM. On an 8 GB card with 64 GB of RAM they are worth testing.

Model availability changes quickly. Check [ollama.com/search](https://ollama.com/search) for current tags.

### Tips for small GPUs

* Close browsers and games while testing; they share the GPU.
* Run one model at a time. Set `OLLAMA_KEEP_ALIVE=0` if you switch models often, so the previous one is unloaded straight away.
* The suite sets `num_ctx` to 8192 for most tests and 16384 for the long context test. Larger contexts use more VRAM.

---

## Uncensored (abliterated) models

"Abliterated" models have had their refusal behaviour removed while keeping most of the base model's ability. The **huihui_ai** builds on Ollama are a common source, for example `huihui_ai/gemma-4-abliterated` and `huihui_ai/qwen3-abliterated`. Check each model's page for its exact size tags.

Run them through this suite next to their original versions. Typically you will see:

* `refusal` goes up (that is the point)
* `hallucination` may go down, because the same change that stops refusals can make a model less willing to say "I don't know"

The comparison tells you whether the trade is worth it for your use.

---

## Customising the suite

Everything lives in `model_eval.py`, organised so tests are easy to change.

### Change the style rules

The house style used by the writing tests is the `STYLE_SYSTEM` string near the top of the test section. Edit it to match your own conventions, and adjust `US_SPELLINGS` or `dash_issues()` if your rules differ.

### Add a test

1. Write a function that returns `(messages, options, grade)`:

   ```python
   def t_my_test():
       msgs = [{"role": "user", "content": "Summarise this policy in 3 bullets: ..."}]

       def grade(text, stats, r):
           bullets = [l for l in text.splitlines() if l.strip().startswith(("-", "*"))]
           r.check("exactly 3 bullets", len(bullets) == 3, f"{len(bullets)} bullets")
           r.check("mentions the deadline", "30 June" in text)

       return msgs, {"temperature": 0.2}, grade
   ```

2. Register it in `TESTS`:

   ```python
   "my_test": ("Policy summary in 3 bullets", t_my_test),
   ```

3. Give it a weight in `WEIGHTS`:

   ```python
   "my_test": 1,
   ```

`grade` receives the model's reply (with any `<think>` blocks removed), timing stats (`tok_per_s`, `ttft_s`, `prompt_tokens`, `wall_s`) and a result object. Each `r.check(name, passed, detail)` becomes one PASS or FAIL line in the report.

Useful helpers already in the script: `words()`, `dash_issues()`, `us_spellings()`, `refused()`, `extract_json()`, `extract_code()` and `run_python()`.

### Change the verdict thresholds

Edit `verdict()` and `WEIGHTS` near the bottom of the script.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach Ollama at http://localhost:11434` | Start the Ollama app (tray icon on Windows) or run `ollama serve`. |
| `Not pulled yet: ...` | Run `ollama pull <model>` with the exact tag shown in `ollama list`. |
| `py` is not recognised | Install Python (see above) and reopen the terminal. |
| Everything is very slow | Run `ollama ps`. If it shows a CPU share, the model does not fit in VRAM. Try a smaller model or close other GPU apps. |
| `long_context` fails on "prompt fully read" | The model or Ollama truncated the input. Try a model with a larger context window, or more VRAM. |
| A reasoning model fails the JSON tests | Try `--think off`. Thinking output can leak into strict formats. |
| Request timed out | Each call allows 15 minutes. If a model is that slow it is not practical for daily use anyway. |

---

## Limitations and safety notes

* **Model generated code is executed on your machine.** The `coding` and `coding_fix` tests run the model's Python in a temporary folder with a 20 second timeout, but not in a sandbox. Only test models you trust, or run the suite in a VM or container. Skip those tests with `--only` if you prefer.
* **One run, one sample.** Models are not deterministic at higher temperatures. For a close call, run the suite twice and compare.
* **Automatic checks are heuristics.** Refusal and uncertainty detection use phrase matching, and language detection uses common words. Read the outputs in the report before making a final decision.
* **Speed results are specific to your hardware** and to whatever else was running at the time.

---

## Licence

No licence has been chosen yet. Until one is added, all rights are reserved by Company31.
