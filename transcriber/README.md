# transcriber

Offline English speech-to-text for a single speaker on a USB microphone,
built for a Jetson Orin Nano (JetPack 6, Python 3.10). Backend: faster-whisper
on CTranslate2 (CUDA when the source-built binding is present, CPU otherwise).

```
mic -> resampler (16 kHz, stream-safe) -> WebRTC VAD -> utterance builder
    -> faster-whisper -> sessions/<timestamp>/{transcript.txt, utterances.jsonl, clips/, session.wav, session.json}
```

## Install (Jetson, with internet)

```bash
bash install.sh                 # everything, non-interactive, safe to re-run
bash install.sh --interactive   # ...plus a wizard that edits config.toml (mic / device / model)
```

`install.sh` is idempotent: apt packages, the virtualenv (`.venv`, or the one
named in `.venv_path` / `--venv DIR`), pip requirements, the CUDA CTranslate2
build, the model download and the tests are each skipped when already done.
The CUDA step reuses `../ct2/` from a previous compile (and caches the built
wheel in `../ct2/wheels`), so re-installing into a new venv takes seconds, not
45 minutes. That build root lives inside `Emote/`; point it elsewhere with
`CT2_BUILD_ROOT=/some/dir bash install.sh` or the build script's `--build-root`. Useful flags: `--skip-cuda-build`,
`--only-model`, `--only-cuda`, `--ct2-jobs 2` (if the compile is OOM-killed).

## Run

```bash
bash run.sh                    # live session; Enter / q / Ctrl+C stops, everything is saved
bash run.sh --list-devices     # find the mic; put a name substring in config.toml
bash run.sh --doctor           # environment health report
bash run.sh --wav file.wav     # same pipeline over an audio file
bash run.sh --name interview   # label the session folder
```

## Configuration

`config.toml` is the single file to edit. Every knob is documented inline.
The ones that matter most:

| Key | Default | Notes |
|-----|---------|-------|
| `audio.microphone` | `"USB"` | name substring or index; substring survives USB re-enumeration |
| `model.name` | `small.en` | `base.en` is faster, `medium.en` better but slower |
| `model.device` | `auto` | `cuda` if the CUDA CTranslate2 build is present, else `cpu` |
| `vad.silence_end_ms` | 700 | how long a pause ends an utterance |
| `vad.max_utterance_s` | 15 | long monologues are chunked |
| `filter.*` | | drops Whisper hallucinations on noise ("Thank you.") |
| `output.session_wav_source` | `device` | native mic rate for later analysis |

## Session outputs

| File | Content |
|------|---------|
| `transcript.txt` | `[hh:mm:ss.mmm] text` per kept utterance |
| `utterances.jsonl` | one JSON per utterance: `start_s`, `end_s`, `text`, `words[]` with times and probabilities, `avg_logprob`, `no_speech_prob`, `kept`, `dropped_reason`, `clip` |
| `clips/utt_NNNN.wav` | 16 kHz mono, exactly what Whisper heard |
| `session.wav` | whole run, mono 16-bit, native mic rate |
| `session.json` | mic, rates, model actually used, config summary, counts |

All times are seconds from the start of `session.wav`.

## Troubleshooting

- `No input device name contains 'USB'`: the error lists every device; set
  `audio.microphone` to a substring of the right one, or its index.
- `PortAudio library not found`: `sudo apt-get install libportaudio2 portaudio19-dev`
  then `bash install.sh` again.
- `ctranslate2 CUDA devices 0` in `--doctor`: `bash install.sh --only-cuda`.
  `model.device = auto` keeps working on CPU meanwhile.
- Frequent `input overflow` warnings: set `audio.block_duration_ms = 0` or a
  larger `audio.latency`.

## Logs: where to look when something goes wrong

| File | Content |
|------|---------|
| `logs/run_<timestamp>.log` (also `sessions/<ts>/run.log`) | DEBUG trail of one run: environment and package versions, full config, every audio device found, every sample rate tried, mic stream parameters, a heartbeat every 5 s (mic level, queue depth, VAD speech %, overflows), one block per utterance with Whisper's segments and confidences, and a final summary. |
| `logs/latest.log` | Points at the newest run log. |
| `logs/transcriber.log` | INFO history across all runs. |
| `logs/install_<timestamp>.log` | Full output of every `install.sh` run, with a machine/CUDA/network header. `--debug` adds shell tracing. |
| `logs/ct2_build_<timestamp>.log` | Full CTranslate2 compile output. |
| `logs/crash_<timestamp>.log` | Only left behind when a native crash (segfault in PortAudio or CTranslate2) happened; contains the thread stacks. |

### Getting logs off the Jetson

```bash
bash scripts/collect_diagnostics.sh              # one tar.gz: logs, session metadata, system/audio/CUDA facts
bash scripts/collect_diagnostics.sh --with-audio # also the recordings (large; it is your voice)
```

It runs `--doctor` and `--list-devices` first, so their logs are included, and adds what the
logs alone cannot show: USB devices, who holds the microphone, memory pressure, kernel
messages, and which `libctranslate2` the Python package actually loads. Copy the tarball back
to the machine where the code is being fixed.

`bash run.sh --log` shows the INFO log on the console and `--verbose` the DEBUG lines (the default screen shows only the mascot, transcript and warnings). Warnings to
watch for in a run log: `SILENT?` (mic delivers zeros), `stream appears
stalled` (USB dropped), `input overflow`, `RTF > 1` (slower than real time,
usually CPU fallback), `DROPPED (...)` (hallucination filter).

## Development on a machine without a microphone

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
bash run.sh --wav some_speech.wav
```

## Personal vocabulary

Names and jargon Whisper should prefer go in `vocabulary.txt`, one per line, most important
first (only ~400 characters are used). `bash go.sh vocab` suggests words from your own
recorded sessions; `bash go.sh vocab --write` appends them. Keep it to words you actually
say: on a 2026-10-10 benchmark a list of the speakers' real names cut errors by 20-33%,
while unrelated words did nothing either way.
