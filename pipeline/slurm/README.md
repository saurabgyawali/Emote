# Emote on ACCRE

Written 2026-11-04. Modelled on DALE's `clip_clip/slurm/` (same allocation, same pitfalls). Tested on the lab box
with stub `sbatch`/`squeue`/`accre_storage` only: **nothing here has run on ACCRE yet.**

| Item | Value |
|---|---|
| Code on ACCRE | `~/Documents/Emote/` (copied from the lab box `Games/Emote-to-copy/`, built by `bash go.sh copy`) |
| Data, encoders, checkpoints | `~/emote_data/` (`$EMOTE_DATA`): `raw/`, `cache/`, `features/`, `manifests/`, `pretrained/`, `runs/<exp>/seed<k>/best.pt`, `deploy/` |
| Small results | `~/Documents/Emote/pipeline/results/` (JSON, CSV, per-run logs), `pipeline/figures/` |
| Python | Miniforge env `~/miniforge3/envs/emote` (conda-forge scientific stack + pip torch 2.6.0+cu124, the validated pair) |
| Allocation | `dsi_dgx_iacc` / `interactive_gpu` / `dgx_iacc`. Every job needs a typed GPU (CPU-only jobs are rejected: "Missing GRES flag"); headers ask for `nvidia_a100-sxm4-80gb:1`, override with `--gres=gpu:nvidia_a100-sxm4-40gb:1` (peak 5.3 GB, 7.1 GB WavLM-Large) |
| Slurm logs | `~/Documents/Emote/pipeline/logs/slurm/`, record of every submit in `pipeline/logs/submissions.log` |

## Commands (login node, from `~/Documents/Emote`)

| Step | Command | Job | Expect |
|---|---|---|---|
| 1. Setup | `bash pipeline/slurm/submit.sh setup --all-models` | 1 GPU, 32 CPU, 96 G, 10 h | env, downloads (~12 GB + 11 GB MELD), manifest, prosody, tests, GPU smoke test. 1-3 h |
| 2. Check | `bash pipeline/slurm/submit.sh status` ; `tail -n 40 pipeline/logs/slurm/setup_<job>.out` | - | manifest ~25k clips, `prosody: extracted`, `Setup complete` |
| 3. Quick look | `bash pipeline/slurm/submit.sh train --experiments E5_audio_text_prosody --seeds 0 --override train.epochs=2` | 1 task | `pipeline/results/E5_audio_text_prosody/seed0/train_summary.json`: minutes/epoch, peak memory |
| 4. Core suite | `bash pipeline/slurm/submit.sh train` | 18 tasks + post | ~6 at a time (group GPU cap), ~20-40 min each |
| 5. More | `bash pipeline/slurm/submit.sh train --groups extended large` | 12 tasks + post | E6/E7/X2 and WavLM-Large (3x slower) |
| 6. Results | `cat pipeline/results/RESULTS.md` ; `ls ~/emote_data/deploy/` | post job | tables, 11 figures, ONNX for the Jetson |

- `submit.sh list [selection]` prints the array index -> run table without submitting. Re-run one task: `submit.sh train --array=7 --force`.
- A resubmitted array skips finished runs (config fingerprint), so after failures just submit the same command again.
- The quick look's override changes its config fingerprint, so step 4 retrains E5 seed 0 properly (and replaces it).

## ESD (public download, signed licence first)

1. The PI signs the licence (linked from https://github.com/HLTSingapore/Emotional-Speech-Data) and emails the dated PDF to haizhou.li@nus.edu.sg.
   Use is allowed only after that (clause 5). Clause 6: models built on ESD must be made freely available and sent to the licensor.
2. Then `bash pipeline/slurm/submit.sh setup --all-models --esd` (downloads only the missing pieces; English speakers kept).
   If Google Drive refuses ("too many users"), download the zip in a browser, scp it to `~/emote_data/raw/esd/`, resubmit.
3. If the manifest was already built without ESD: `bash pipeline/slurm/submit.sh prepare`.

CMU-MOSEI is not used: CMU shares only processed features, not audio ("we are not allowed to share the raw videos",
CMU-MultimodalSDK README), and every model here starts from the waveform.

## Licensed corpora (after approval)

1. Download on ACCRE (login node) or the lab box, unpack to `~/emote_data/raw/iemocap/IEMOCAP_full_release/Session1..5` and
   `~/emote_data/raw/msp_podcast/{Audios,Labels/labels_consensus.csv,Transcripts}`.
2. `bash pipeline/slurm/submit.sh prepare` (manifest rebuilt, new prosody rows only).
3. Retrain **named** experiments only (every config fingerprint changes when the data does):
   `bash pipeline/slurm/submit.sh train --experiments E5_audio_text_prosody X1_acted_to_natural`.
- **Space:** MSP-Podcast is 100+ GB. With the open corpora and 30 runs (~70 G charged) home cannot also hold it
  (186 G quota, x1.16 panfs charge). Ask ACCRE or your PI for group storage the dgx nodes can read before downloading it.

## Pitfalls carried over from DALE

| Pitfall | Handled by |
|---|---|
| dgx nodes cannot write `/nobackup/userspace` | everything in home |
| `module purge` under `set -u` kills the job with an empty log | `env.sh` turns `-u` off around module/conda |
| an already-active env makes `conda activate` a no-op and the system python wins | `env.sh` deactivates first and checks the python path |
| `/cvmfs` wheels need GLIBC 2.35 (Rocky 9 has 2.34) | conda-forge stack; `PIP_CONFIG_FILE=/dev/null` |
| `#SBATCH` lines after the first command are ignored | headers first in every file |
| relative paths fail when submitted from the wrong folder | `submit.sh` cds to the Emote folder itself |
| a machine-wide `HF_HOME` sends models to another project | forced to `$EMOTE_DATA/pretrained`; train jobs run `HF_HUB_OFFLINE=1` |
