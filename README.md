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

### Install on the Jetson

```bash
cd Emote/transcriber
bash install.sh                 # idempotent; safe to re-run, completed steps are skipped
bash install.sh --interactive   # ...plus a wizard for mic / device / model
```

`install.sh` runs seven steps, each skipped when already satisfied: apt packages → virtualenv →
pip requirements → **CUDA CTranslate2** → Whisper model → unit tests → health report.
Everything is tee'd to `logs/install_<timestamp>.log`.

Useful flags: `--skip-cuda-build`, `--only-cuda`, `--only-model`, `--ct2-jobs 2` (if the compile is OOM-killed), `--debug`.

### Run a live session

```bash
bash run.sh --list-devices      # find the mic; put a name substring in config.toml
bash run.sh --doctor            # environment health report
bash run.sh                     # talk; Enter / q / Ctrl+C stops and saves everything
bash run.sh --name interview    # label the session folder
```

### Run over a file (no microphone needed)

```bash
bash run.sh --wav some_speech.wav --name test
```

This is the path to use on a development machine: it exercises the whole pipeline without touching PortAudio.

### Extract features

```bash
cd Emote/emotion_analysis
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/extract_features.py        # -> results/features.csv, appends only new utterances
.venv/bin/python scripts/extract_features.py --force
.venv/bin/python scripts/summarize_sessions.py
.venv/bin/python scripts/train_baseline.py
```

### Everything at once

```bash
cd Emote
bash run_pipeline.sh            # transcriber -> features -> baseline training in the background
```

### Tests

```bash
cd Emote/transcriber      && .venv/bin/python -m pytest -q tests
cd Emote/emotion_analysis && .venv/bin/python -m pytest -q tests
```

---

## 📦 Moving to the Jetson

```bash
cd Emote
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
| `logs/run_<ts>.log` | DEBUG trail of one run: environment, full config, every device and rate tried, heartbeats, per-utterance Whisper output, final summary. |
| `logs/latest.log` | Points at the newest run log. |
| `logs/transcriber.log` | INFO history across all runs. |
| `logs/install_<ts>.log` | Full output of every `install.sh` run. |
| `logs/ct2_build_<ts>.log` | Full CTranslate2 compile output. |
| `logs/crash_<ts>.log` | Only present after a native crash; contains the thread stacks. |

---

## 🛣 Roadmap

- [x] Live microphone → transcript pipeline on the Jetson.
- [x] CUDA CTranslate2 build, self-contained inside the project.
- [x] Per-utterance feature extraction (62 features).
- [x] Unsupervised baseline with silhouette scoring.
- [ ] 🏷 Collect labelled sessions — self-report per session or per utterance.
- [ ] 📈 Supervised baseline on the per-speaker-normalised feature table, replacing k-means.
- [ ] 🤖 Fine-tune a pretrained speech emotion encoder (wav2vec2 / HuBERT) on the clips.
- [ ] 🔤 Fuse acoustic and lexical models using the transcript text.
- [ ] 🎯 Report categories as probabilities so overlapping states stay visible.
- [ ] 🗜 Compress and quantise the model for real-time inference on the Orin Nano.
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
