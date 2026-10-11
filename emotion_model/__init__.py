"""Emote stage 3: speech emotion recognition from how something is said (audio, prosody) and what is said (transcript).

Modules
-------
labels    unified emotion classes, per-dataset label sets, the manifest schema
audio     loading, loudness normalisation, cropping, training augmentation
prosody   the 43 hand-crafted pitch/energy/spectral features from emotion_analysis
data      manifest-backed Dataset, collate, balanced sampler
model     audio tower (WavLM/HuBERT/wav2vec2), text tower (RoBERTa), prosody tower, gated fusion, heads
losses    label-set-masked cross-entropy, concordance (CCC) loss
metrics   UA / WA / macro-F1 / confusion / CCC / calibration, per dataset
config    YAML config with defaults and dotted CLI overrides
"""

__version__ = "0.1.0"
