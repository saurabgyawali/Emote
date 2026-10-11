<div align="center">

# Emote 🎙️

> Reading how something is said, not just what was said — on an edge device, fully offline.

![Platform](https://img.shields.io/badge/platform-Jetson%20Orin%20Nano-76B900?logo=nvidia&logoColor=white)
![Language](https://img.shields.io/badge/language-Python%203.10-3776AB?logo=python&logoColor=white)
![ASR](https://img.shields.io/badge/ASR-faster--whisper-000000?logo=openai&logoColor=white)
![Backend](https://img.shields.io/badge/backend-CTranslate2%20CUDA-76B900?logo=nvidia&logoColor=white)
![Offline](https://img.shields.io/badge/network-none%20required-success)
![Status](https://img.shields.io/badge/status-Alpha-orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Emote** captures one speaker on a USB microphone, transcribes them locally on a Jetson Orin Nano,
and turns every utterance into a table of acoustic and timing features — the raw material for
speech emotion recognition, with no audio ever leaving the device.

**Pipeline:** microphone → voice activity detection → local Whisper → per-utterance features → model

**Emotion model (stage 3):** the training pipeline is in [`pipeline/`](pipeline/README.md) — start with **[NEXT_STEPS.md](NEXT_STEPS.md)** on a GPU machine.

[Features](#-features) • [How It Works](#-how-it-works) • [Getting Started](#-getting-started) • [Roadmap](#-roadmap) • [Contributing](#-contributing)

</div>

---

## 🧭 Table of Contents

- [Features](#-features)
- [Why Emote?](#-why-emote)
- [How It Works](#-how-it-works)
- [What a Session Produces](#-what-a-session-produces)
- [Feature Set](#-feature-set)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Modes of Operation](#-modes-of-operation)
- [Moving to the Jetson](#-moving-to-the-jetson)
- [Configuration](#-configuration)
- [Diagnostics](#-diagnostics)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Known Issues](#-known-issues)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## ✨ Features

### Stage 1 — live transcriber (`transcriber/`)

- 🎤 **Robust microphone capture**
  - The device is chosen by **name substring** (`"USB"`), not index, because USB indices change between boots.
  - Sample rate is probed: 16 kHz first (no resampling), then the device's own default.
  - Stream-safe resampling keeps filter state across blocks, so utterances have no seam artifacts.

- 🗣 **Utterance segmentation, not fixed windows**
  - WebRTC VAD on 30 ms frames decides where speech starts and stops.
  - Configurable start/stop thresholds, pre-roll, minimum length, and a hard chunking limit for monologues.
  - Coughs, clicks, and short blips are discarded before they ever reach the model.

- ⚡ **Local Whisper on the GPU**
  - `faster-whisper` on **CTranslate2 built from source with CUDA** (SM 8.7 / Orin), because the stock aarch64 wheels are CPU-only.
  - Automatic CPU fallback with a lighter model when no CUDA device is present.
  - Word-level timestamps and confidences, needed for the timing features downstream.

- 🧹 **Hallucination filter**
  - Whisper's classic silence artifacts ("Thank you.", subtitle credits) are dropped by log-probability, no-speech probability, and repetition rules.
  - Every dropped utterance is logged with its reason, so thresholds can be tuned from real data.

- 🩺 **Built to be debugged remotely**
  - DEBUG run log per session, a heartbeat every 5 s (mic level, queue depth, VAD speech %, overflows), native-crash stacks via `faulthandler`, and a one-command diagnostics bundle.
  - `install.sh` and `run.sh --doctor` print a full environment report before anything else runs.

### Stage 2 — feature extraction (`emotion_analysis/`)

- 📊 **62 features per utterance** across prosody, energy, spectral shape, timbre, timing, and ASR quality.
- ♻️ **Idempotent**: re-running appends only new utterances; `--force` recomputes everything.
- 🧪 **Unsupervised baseline**: k-means over the standardised feature table with a silhouette score and a PCA scatter — a sanity check that vocal styles separate *at all* before any label is collected.
- 📁 **CSV + Parquet** output, so the table drops straight into pandas, scikit-learn, or a notebook.

---

## 🎯 Why Emote?

Speech emotion work usually starts with a corpus someone else recorded, in a studio, in a language and setting that does not match the deployment.

Common problems:
- ❌ Cloud ASR means every second of a person's voice leaves the room.
- ❌ Fixed-length windows cut mid-word and blur the prosody that carries the emotion.
- ❌ Transcript-only pipelines throw away pitch, loudness, and pause structure — where most of the affect actually lives.
- ❌ Edge deployment is an afterthought, so nothing survives the move from a workstation to a 7 GB board.

**Emote** is built the other way around: edge-first, offline, utterance-aligned, and it keeps the audio alongside the text so features can always be recomputed from the original signal.

---

## 🔄 How It Works

```
USB mic -> sounddevice callback (copies blocks only) -> audio_queue
  -> audio worker: stream-safe resample to 16 kHz -> WebRTC VAD (30 ms frames) -> utterance builder
  -> utterance_queue -> transcription worker: faster-whisper -> hallucination filter
  -> sessions/<timestamp>/
```

1. **Capture** — the PortAudio callback does nothing but copy blocks into a queue, so the audio thread never blocks and never drops frames while the GPU is busy.
2. **Segment** — the audio worker resamples to 16 kHz and runs VAD frame by frame, emitting one utterance when trailing silence exceeds the threshold.
3. **Transcribe** — a separate worker thread runs Whisper on the finished utterance, with word timestamps, while capture continues uninterrupted.
4. **Filter and persist** — surviving utterances are written to disk with their audio clip, timings, and confidences.
5. **Extract** — offline, on any machine, `extract_features.py` turns each clip into a feature row.

---

## 📦 What a Session Produces

Every run creates `transcriber/sessions/<timestamp>/`:

| File | Content |
|------|---------|
| `transcript.txt` | Plain running transcript, one line per utterance. |
| `utterances.jsonl` | One JSON object per utterance: text, start/end times, word timings, confidences, filter verdict. |
| `clips/utt_NNNN.wav` | The exact 16 kHz audio Whisper heard, per utterance. |
| `session.wav` | The full session at the microphone's native rate, so features can be recomputed from the original signal. |
| `session.json` | Session metadata: device, rates, model, config snapshot, counts, durations. |
| `run.log` | A copy of the DEBUG log for that run. |

---

## 🎼 Feature Set

`emotion_analysis/scripts/extract_features.py` → `results/features.csv` (+ `.parquet`), one row per kept utterance.

| Group | Columns | Why it matters for emotion |
|-------|---------|----------------------------|
| **prosody** | `f0_mean/std/min/max/range/slope`, `voiced_fraction` | Arousal: higher, more variable pitch in anger/joy/fear; flat, low pitch in sadness. |
| **energy** | `rms_mean/std/max`, `dynamic_range_db` | Loudness and its variability track arousal. |
| **spectral** | `spectral_centroid/bandwidth/rolloff/flatness`, `zcr` | Voice brightness and tension — harsh versus breathy. |
| **timbre** | `mfcc1..13_mean/std` | The standard input for classical speech-emotion classifiers. |
| **timing** | `speech_rate_wps`, `articulation_rate_wps`, `syllable_rate_proxy`, `pause_mean/max/fraction`, `duration_s` | Fast with few pauses = excited or angry; slow with many pauses = sad or tired. |
| **asr** | `avg_logprob`, `no_speech_prob`, `word_prob_mean`, `word_count`, `text` | Quality control, plus text for lexical emotion cues. |

Pitch uses `librosa.pyin` over 60–500 Hz. Everything is computed on the same 16 kHz clip the model transcribed.

---

## 🧱 Tech Stack

- **Language:** Python 3.10
- **ASR:** `faster-whisper` (`small.en` by default, `base.en` as CPU fallback)
- **Inference backend:** CTranslate2 v4.5.0, built from source with `-DWITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`
- **Audio I/O:** `sounddevice` / PortAudio, `soundfile`
- **Voice activity detection:** `webrtcvad`
- **Signal processing:** `numpy`, `scipy`, `librosa`
- **Modelling:** `pandas`, `scikit-learn`, `matplotlib`
- **Hardware:** NVIDIA Jetson Orin Nano (JetPack 6, L4T R36.4.7, CUDA 12.6)
- **Network:** none at runtime — install is the only step that needs the internet

---

## 🗂 Project Structure

```text
Emote/
├── transcriber/                    # Stage 1 — runs on the Jetson
│   ├── config.toml                 # The one file you edit; every knob documented inline
│   ├── install.sh                  # Idempotent installer: apt, venv, pip, CUDA CTranslate2, model, tests, doctor
│   ├── run.sh                      # Launcher: --list-devices, --doctor, --wav, --name, --verbose
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── build_ctranslate2_cuda.sh    # CUDA CTranslate2 build, reuses ../ct2 and its wheel cache
│   │   ├── collect_diagnostics.sh       # One tar.gz with logs + system/audio/CUDA facts
│   │   └── download_model.py            # Whisper model fetch
│   ├── src/transcriber/
│   │   ├── main.py                 # Pipeline, threads, heartbeat, run summary
│   │   ├── audio_capture.py        # Device resolution, rate probing, capture stream
│   │   ├── resampler.py            # Stream-safe resampling (filter state kept across blocks)
│   │   ├── vad_gate.py             # WebRTC VAD framing
│   │   ├── utterance_builder.py    # Speech/silence state machine, pre-roll, chunking
│   │   ├── whisper_backend.py      # faster-whisper wrapper, CUDA detection, CPU fallback
│   │   ├── session.py              # Session folder writer
│   │   ├── context_window.py       # Rolling context across utterances
│   │   ├── logging_utils.py        # DEBUG run log, crash logs, exception hooks
│   │   ├── doctor.py               # Environment health report
│   │   └── config.py / configure.py     # Config loading and the interactive wizard
│   ├── tests/                      # Unit tests (config, resampler, utterance builder, helpers)
│   ├── models/                     # Downloaded Whisper weights (not in git)
│   ├── logs/                       # run_/install_/ct2_build_/crash_ logs (not in git)
│   └── sessions/                   # Session output (not in git)
│
├── emotion_analysis/               # Stage 2 — runs anywhere
│   ├── scripts/
│   │   ├── extract_features.py     # Sessions -> results/features.csv
│   │   ├── summarize_sessions.py   # Per-session summary table
│   │   └── train_baseline.py       # Unsupervised k-means baseline + PCA plot
│   ├── tests/
│   └── requirements.txt
│
├── ct2/                            # CUDA CTranslate2 build root (source tree not in git)
│   ├── CTranslate2/                # v4.5.0 source + aarch64 build tree (~11 GB)
│   ├── wheels/                     # Cached built wheels — turns a 45 min build into seconds
│   └── README.md
│
├── go.sh                           # The one command: update -> check -> record -> bundle logs
├── run_pipeline.sh                 # Transcribe -> extract features -> train baseline, one command
├── pack.sh                         # Clean tarball for copying to the Jetson
├── CLAUDE.md                       # Engineering context, verified facts, debugging playbook
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Jetson Orin Nano** with JetPack 6 (L4T R36.x), CUDA 12.x, Python 3.10 — for the live transcriber.
- Any Linux machine with Python 3.10+ — for feature extraction.
- A **USB microphone**.
- Internet access for the install only.

### First run on a new machine

```bash
cd Emote
bash go.sh install      # venv, requirements, CUDA CTranslate2, model, tests, health report
bash go.sh mic          # confirm the microphone is loud enough
bash go.sh              # record
```

That is the whole setup. Details of every flag are in
[Installation modes](#installation-modes) below.

---

## 🎛 Modes of operation

Everything is driven by two scripts. `go.sh` (in `Emote/`) is the one to remember — it chains
the steps in the right order. `run.sh` (in `Emote/transcriber/`) is the transcriber itself,
and every `go.sh` step is a thin wrapper around it.

### The default: one command, the whole loop

```bash
cd Emote
bash go.sh
```

In order: **update** the code → **check** the machine → **record** a session → **bundle** the
logs. If the check fails it asks before recording. If there is no virtualenv it offers to
install one. If the session crashes it still bundles the logs. At the end it prints the path
of the single file to carry back.

Arguments pass through to the recorder, so `bash go.sh --quiet --name interview1` records a
labelled session with transcript-only output.

### `go.sh` step by step

Run any step alone when you do not want the whole loop:

| Command | What it does | When you want it |
|---|---|---|
| `bash go.sh update` | `git pull --ff-only`, or explains how to copy the code across when there is no remote. | Before a session, to be sure you are running the newest code. |
| `bash go.sh install` | Runs `install.sh`: venv, pip, CUDA CTranslate2, model, tests, health report. Idempotent. | First run on a machine, or after the venv is lost. |
| `bash go.sh check` | Health report only (`run.sh --doctor`). | "Is this machine ready?" Confirms CUDA, model, mic. |
| `bash go.sh mic` | 10-second level meter with a verdict. | **Nothing gets transcribed.** Almost always capture gain. |
| `bash go.sh record` | A recording session only. | You already checked, just want to record. |
| `bash go.sh logs` | Builds the diagnostics tarball only. Add `--with-audio` to include the recordings. | After a crash, or to re-send evidence. |
| `bash go.sh features` | `run_pipeline.sh --no-record`: extracts features from sessions already on disk. | After recording, to get `features.csv`. |
| `bash go.sh copy` | Builds `../Emote-to-copy/` — code only, ~240 KB. `--tar` makes a tarball with the model instead. | Moving code to the Jetson by hand. |

### Console modes while recording

The same session, four ways to watch it. All four write the identical full DEBUG
log to `transcriber/logs/<day>/<hour>/run_<timestamp>.log` — the mode only changes what you see.

| Mode | Command | On screen |
|---|---|---|
| **Default** | `bash run.sh` | The mascot screen: a small listening character with a live status line (face, mic level in dB, heard/dropped counts, clock), transcript lines scrolling above it, dropped lines in dim grey, warnings in yellow. Plain text, no status line, when output is not a terminal or `NO_COLOR` is set. |
| **Log** | `bash run.sh --log` | INFO log: device chosen, audio plan, Whisper plan, a heartbeat every 5 s, one line per utterance. |
| **Quiet** | `bash run.sh --quiet` | One line, then only transcript lines: `[00:00:02.210] Hello there.` Nothing else. |
| **Verbose** | `bash run.sh --verbose` | Everything, including DEBUG: every device, every sample rate tried, every VAD transition. |

Use the default to talk to it, quiet to copy text out, log or verbose when something is wrong
and you want to watch it happen. All work through `go.sh` (`bash go.sh --log`).

### Stopping a session

**Esc**, **Enter**, **q**, or **Ctrl+C** — all four stop cleanly. Esc works as a single
keypress when you are at a terminal; in a pipe or under `nohup`, Enter is needed.

Stopping is always safe. Every utterance is written to disk the moment it is transcribed, the
session folder is finalised on the way out, queued utterances are drained first, and `go.sh`
bundles the logs afterwards either way. There is no "too early".

### Input modes

| Input | Command | Notes |
|---|---|---|
| **Microphone** | `bash run.sh` | The real path. Needs PortAudio and a USB mic. |
| **Audio file** | `bash run.sh --wav clip.wav --name test` | Same pipeline, no microphone. This is how the pipeline is exercised on a machine without PortAudio. |

### Inspection modes (these exit immediately, they do not record)

| Command | Shows |
|---|---|
| `bash run.sh --doctor` | Python, packages, ctranslate2 CUDA device count, config validity, model presence, every input device, which one `audio.microphone` resolves to, and an OK / PROBLEMS verdict. Also written to the run log. |
| `bash run.sh --list-devices` | Just the input devices, with indices. Use it to fill in `audio.microphone`. |
| `bash run.sh --mic-check` | Records for 10 s (`--mic-check 30` for longer) with a live level bar, then reports peak and RMS in dBFS, what percentage of frames the VAD counted as speech, and a verdict: too quiet, clipping, VAD barely triggering, or usable. |

`--mic-check` is the one to reach for when a session produces little or no transcript. A peak
below 0.02 means the VAD chops your speech into bursts shorter than `vad.min_speech_ms`, and
they are thrown away before Whisper sees them — you speak for two minutes and get five lines.
Raise the capture gain with `alsamixer -c 0` (F4 for capture, arrow up, Esc) or
`amixer -c 0 sset Mic 80% cap`, then check again. If the hardware gain is already at maximum,
set `[audio] input_gain` in `config.toml` to the value `--mic-check` suggests; it is applied in
software before the VAD, and is computed from the speech RMS with headroom so it cannot clip.

The run summary reports the same thing after the fact:

```
  discarded          : 13 short speech burst(s), 5.4s total, below min_speech_ms=300
  input level        : peak 0.0094 (-40 dBFS), loudest rms 0.0042 (-48 dBFS)
                       TOO QUIET -- raise the capture gain, or set audio.input_gain = 9
```

### Installation modes

```bash
cd Emote/transcriber
bash install.sh                 # everything, idempotent, safe to re-run
bash install.sh --interactive   # ...plus a wizard for mic / device / model
```

Seven steps, each skipped when already satisfied: apt packages → virtualenv → pip requirements
→ **CUDA CTranslate2** → Whisper model → unit tests → health report. Tee'd to
`logs/<day>/<hour>/install_<timestamp>.log`.

| Flag | Effect |
|---|---|
| `--only-cuda` | Just the CUDA CTranslate2 step. |
| `--only-model` | Just the Whisper model download. |
| `--skip-cuda-build` | Stay on the CPU wheel. |
| `--ct2-jobs 2` | Fewer parallel compile jobs, if the build is OOM-killed. |
| `--venv DIR` | Use or create a virtualenv elsewhere (network shares reject venv symlinks). |
| `--debug` | Trace every shell command into the log. |

The CUDA step is smart about what already exists, in this order: a cached wheel in
`ct2/wheels/` (seconds) → the C++ library already installed in `/usr/local` (a few minutes,
Python binding only) → a reusable build tree → a full compile (30–45 minutes).

### Pipeline modes (both stages at once)

```bash
cd Emote
bash run_pipeline.sh                  # record -> features -> baseline training in the background
bash run_pipeline.sh --wav clip.wav   # same, from a file
bash run_pipeline.sh --no-record      # skip recording; use the sessions already on disk
bash run_pipeline.sh --skip-train     # stop after features.csv
bash run_pipeline.sh --foreground-train   # wait for training instead of backgrounding it
```

`ANALYSIS_PYTHON=/path/to/python` overrides the interpreter used for stage 2.

### Stage 2 on its own

```bash
cd Emote/emotion_analysis
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/extract_features.py          # -> results/features.csv (appends new utterances only)
.venv/bin/python scripts/extract_features.py --force  # recompute everything
.venv/bin/python scripts/summarize_sessions.py        # per-session table
.venv/bin/python scripts/train_baseline.py            # unsupervised k-means + PCA plot
```

### Where everything lands

Logs are filed by the day and hour they were written, because a flat folder became
unreadable after a single morning of debugging:

```
transcriber/logs/
├── 2026-09-20/
│   ├── 07/
│   │   └── run_20260920_074039.log
│   └── 09/
│       ├── run_20260920_092054.log
│       ├── install_20260920_092010.log
│       ├── ct2_build_20260920_092012.log
│       └── crash_20260920_092054.log     (only after a native crash)
├── diagnostics/
│   └── 2026-09-20/
│       └── 09/
│           └── diagnostics_jetson-desktop_20260920_092103.tar.gz
├── latest.log            → the newest run log
├── install_latest.log    → the newest install log
└── transcriber.log       INFO history across every run
```

| Path | What | In git? |
|---|---|---|
| `transcriber/sessions/<timestamp>/` | transcript, utterances.jsonl, clips, session.wav, session.json, run.log | No |
| `transcriber/logs/<day>/<hour>/run_<ts>.log` | Full DEBUG log of one run | No |
| `transcriber/logs/diagnostics/<day>/<hour>/*.tar.gz` | The bundles to carry back | No |
| `transcriber/logs/<day>/<hour>/install_<ts>.log`, `ct2_build_<ts>.log` | Install and CUDA build output | No |
| `transcriber/logs/latest.log`, `transcriber.log` | Newest run log pointer, and the INFO history | No |
| `emotion_analysis/results/features.csv` | Feature table | No |
| `emotion_analysis/results/model/` | Baseline model, clusters, PCA plot, STATUS.json | No |
| `../Emote-to-copy/` | Code to carry to the Jetson | No (outside the repo) |

Nothing produced by a run is ever committed: recordings and transcripts stay out of git by
design.

### Tests

```bash
cd Emote/transcriber      && .venv/bin/python -m pytest -q tests
cd Emote/emotion_analysis && .venv/bin/python -m pytest -q tests
```

---

## 📦 Moving to the Jetson

The server and the Jetson are not networked to each other, so code crosses by hand. One
command builds exactly what to carry:

```bash
cd Emote
bash go.sh copy                 # -> ../Emote-to-copy/  (code only, ~220 KB, 42 files)
bash go.sh copy --tar           # -> ../Emote-<date>.tar.gz including the model
```

### The round trip

| # | Where | Action |
|---|---|---|
| 1 | Server | `bash go.sh copy` → `../Emote-to-copy/` |
| 2 | Your computer | Download that folder |
| 3 | Jetson | Drop it **on top of** `~/Developer/RASL/Emote`, merge/replace |
| 4 | Jetson | `cd ~/Developer/RASL/Emote && bash go.sh` |
| 5 | Your computer | Take the one tarball it names at the end |
| 6 | Server | Drop it in `transcriber/logs/diagnostics/<day>/<hour>/` |

If the Jetson ever gets this repo directly (it does have internet — the installer reaches
PyPI), `bash go.sh update` will `git pull` instead and steps 1–3 disappear.

Drop `Emote-to-copy/` **on top of** the Jetson's `Emote` folder and choose Merge/Replace.
Never delete that folder first: the model, the venv, the CUDA build and past logs live there
and are deliberately not in the copy.

### Bringing results back

`bash go.sh` ends by printing the path of a single file:

```
transcriber/diagnostics_<hostname>_<timestamp>.tar.gz
```

It is written to `transcriber/logs/diagnostics/<day>/<hour>/`. Drag that one file back into the same folder on the
server. It contains the run logs, the health report, session transcripts and metadata, and
the system, audio, CUDA and kernel facts. Audio is excluded unless you run
`bash go.sh logs --with-audio`. `transcriber/logs/` is gitignored, so nothing dropped there
is ever committed.

Lower-level packing:

```bash
bash pack.sh                    # -> ../Emote-<date>.tar.gz, clean
bash pack.sh --with-model       # ...including the Whisper weights, so the Jetson need not download them
bash pack.sh --with-ct2-src     # ...including the 11 GB CTranslate2 source tree (rarely wanted)
```

`pack.sh` excludes virtualenvs, caches, logs, sessions, and `ct2/CTranslate2`.
It **keeps** `ct2/wheels/`, so a previously built aarch64 wheel travels with the folder and the CUDA step finishes in seconds instead of 45 minutes.

> **Never copy a `.venv` between machines.** `install.sh` creates or reuses one on the target; an x86 venv on aarch64 is moved aside automatically, but it wastes a copy.

---

## ⚙️ Configuration

`transcriber/config.toml` is the single hand-edited file. Every knob is commented in place. The ones that matter most:

| Key | Default | Meaning |
|-----|---------|---------|
| `audio.microphone` | `"USB"` | Name substring (preferred) or device index. |
| `audio.capture_sample_rate` | `"auto"` | Try 16 kHz first, then the device default. |
| `audio.block_duration_ms` | `100` | Larger blocks = fewer input overflows. |
| `audio.input_gain` | `1.0` | Software gain applied before VAD and Whisper. Only for a mic whose hardware gain is already maxed; `--mic-check` suggests a value. |
| `vad.aggressiveness` | `2` | 0 (lax) … 3 (strict). |
| `vad.silence_end_ms` | `700` | Trailing silence that ends an utterance. |
| `vad.max_utterance_s` | `15` | Long monologues are chunked here. |
| `model.name` | `"small.en"` | `tiny.en` … `medium.en`, `distil-large-v3`. |
| `model.device` | `"auto"` | CUDA when the CUDA CTranslate2 build is present, else CPU. |
| `model.word_timestamps` | `true` | Required for the speaking-rate features. |
| `model.require_cuda` | `false` | `true` refuses to run on CPU. |

---

## 🔍 Diagnostics

```bash
bash scripts/collect_diagnostics.sh                # one tar.gz: logs, session metadata, system/audio/CUDA facts
bash scripts/collect_diagnostics.sh --with-audio   # also the recordings (large; it is your voice)
bash scripts/collect_diagnostics.sh --dry-run      # list what would be collected, write nothing
```

The bundle adds what the logs alone cannot show: USB devices, who is holding the microphone,
memory pressure, kernel messages, and which `libctranslate2` the Python package actually loads.

| Log | Content |
|-----|---------|
| `logs/<day>/<hour>/run_<ts>.log` | DEBUG trail of one run: environment, full config, every device and rate tried, heartbeats, per-utterance Whisper output, final summary. |
| `logs/latest.log` | Points at the newest run log. |
| `logs/transcriber.log` | INFO history across all runs. |
| `logs/<day>/<hour>/install_<ts>.log` | Full output of every `install.sh` run. |
| `logs/<day>/<hour>/ct2_build_<ts>.log` | Full CTranslate2 compile output. |
| `logs/<day>/<hour>/crash_<ts>.log` | Only present after a native crash; contains the thread stacks. |
| `logs/diagnostics/<day>/<hour>/*.tar.gz` | The bundles to carry back; previous bundles are never packed into a new one. |

---

## 🛣 Roadmap

- [x] Live microphone → transcript pipeline on the Jetson.
- [x] CUDA CTranslate2 build, self-contained inside the project.
- [x] Per-utterance feature extraction (62 features).
- [x] Unsupervised baseline with silhouette scoring.
- [ ] 🏷 Collect labelled sessions — self-report per session or per utterance.
- [ ] 📈 Supervised baseline on the per-speaker-normalised feature table, replacing k-means.
- [x] 🧪 Training pipeline: WavLM audio + transcript + prosody towers, gated fusion, 10-experiment suite, analysis figures, ONNX export ([pipeline/](pipeline/README.md)); smoke-tested, awaiting a GPU.
- [ ] 🤖 Train it on RAVDESS, CREMA-D, TESS, MELD (+ IEMOCAP, MSP-Podcast once licensed) — [NEXT_STEPS.md](NEXT_STEPS.md).
- [x] 🔤 Fuse acoustic and lexical models using the transcript text (built; to be trained).
- [x] 🎯 Report categories as probabilities so overlapping states stay visible (`emotions.jsonl`).
- [x] 🗜 Compress and quantise the model for inference on the Orin Nano (int8 ONNX export, parity-checked).
- [ ] 📡 Live emotion readout during a session, not just offline.
- [ ] 👥 Multi-speaker diarisation.

---

## 🤝 Contributing

Contributions are welcome — especially around VAD tuning, hallucination filtering, and feature design.

1. **Fork** this repository.

2. Create a feature branch:
   ```bash
   git checkout -b feature/my-improvement
   ```

3. Make your changes and commit:
   ```bash
   git commit -m "Add my improvement"
   ```

4. Push your branch:
   ```bash
   git push origin feature/my-improvement
   ```

5. Open a **Pull Request** describing what you changed and why.

Shell scripts should pass `bash -n`, Python changes should keep the unit tests green, and anything touching the Jetson path should say explicitly what was verified on hardware and what was not.

---

## 🐛 Known Issues

- **No full Jetson run has been observed yet.** The installer completes end to end on the device, but the live microphone path, GPU transcription, and session output are still unverified on hardware.
- `sounddevice` cannot import without PortAudio, so the live path does not run on machines lacking it — use `run.sh --wav` there.
- The hallucination filter thresholds are untested against real noisy speech and will need tuning from collected sessions.
- The CUDA CTranslate2 build takes 30–45 minutes on an Orin Nano from cold, and can be OOM-killed at high job counts — use `--ct2-jobs 2`.
- **CUDA can run out of memory on the Orin.** The GPU shares system RAM, so a browser or IDE left open can leave too little for the model: `CUDA failed with error out of memory`. Close what you can, check `tegrastats`, or use a smaller `model.name`. The transcriber falls back to the CPU automatically.
- Virtualenv creation fails on some network shares (symlink I/O errors); create the venv on local disk.
- **There is no emotion model yet.** Stage 2 produces features and an unsupervised baseline; no labels have been collected.

---

## 🙏 Acknowledgments

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2) for fast local inference.
- [OpenAI Whisper](https://github.com/openai/whisper) for the underlying ASR models.
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) for frame-level speech detection.
- [librosa](https://librosa.org/) for the acoustic feature implementations.
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) for making local GPU inference on the edge practical.

---

## 📜 License

```text
MIT License

Copyright (c) 2026 Saurab Gyawali

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

**How it is said carries as much as what is said.**

[⬆ Back to Top](#emote-)

</div>
