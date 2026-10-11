"""Three towers, gated fusion, two heads.

                       ┌─ audio tower ──── WavLM-Base-Plus -> learned mix of all layers -> attentive stats pool -> 256
    16 kHz waveform ───┤
                       └─ prosody tower ── 42 pitch/energy/spectral features (stage 2) -> MLP ----------------> 256
    Whisper transcript ── text tower ───── emotion-pretrained DistilRoBERTa -> mean pool ----------------------> 256
                                                                    │
                    gated fusion: softmax over the modalities present for each clip, with modality dropout
                                                                    │
                     categorical head (7 emotions, masked per corpus)   dimensional head (arousal/valence/dominance)

Why this shape
--------------
* Audio: self-supervised speech encoders carry emotion mostly in their middle layers,
  so the tower learns a softmax-weighted sum over every hidden layer (the SUPERB
  recipe) instead of taking the last one. WavLM-Base-Plus is the strongest base-size
  encoder on SUPERB emotion recognition and small enough (95 M) to quantise for the
  Jetson. The CNN front end always stays frozen; the top `finetune_layers` transformer
  layers train.
* Text: what is said matters as much as how (sarcasm aside). Acted corpora reuse a
  handful of emotion-neutral sentences, so text can only help on natural speech (MELD,
  MSP-Podcast, IEMOCAP); modality dropout stops the model leaning on it where it is
  uninformative and keeps it working when Whisper returns nothing.
* Prosody: 42 interpretable features, cheap on the device, robust when the neural
  towers meet an unfamiliar microphone.
* Gated fusion yields per-clip modality weights, logged for analysis, rather than an
  opaque concatenation.
"""

from __future__ import annotations

import torch
from torch import nn

from .labels import DIMENSIONS, EMOTIONS

TINY_AUDIO = "tiny-random-wavlm"
TINY_TEXT = "tiny-random-roberta"


# --- encoders --------------------------------------------------------------------
def build_audio_encoder(name: str, spec_augment: bool) -> nn.Module:
    if name == TINY_AUDIO:
        # Random weights, tiny dims: lets tests and the smoke run exercise the real code
        # path without downloading 380 MB.
        from transformers import WavLMConfig, WavLMModel

        cfg = WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                          conv_dim=(32,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                          num_buckets=32, max_bucket_distance=64, layerdrop=0.0,
                          mask_time_prob=0.05 if spec_augment else 0.0)
        model = WavLMModel(cfg)
    else:
        from transformers import AutoModel

        extra = {"layerdrop": 0.0}
        if not spec_augment:
            extra["mask_time_prob"] = 0.0
        model = AutoModel.from_pretrained(name, **extra)
    # No gradient checkpointing: with transformers 4.57 it re-enters itself inside WavLM and
    # memory grows to 44 GB instead of shrinking (2026-10-10). The full model peaks at 5.3 GB
    # at batch 16 anyway; small GPUs use a smaller batch with gradient accumulation.
    return model


class ByteTokenizer:
    """Stand-in tokenizer for the tiny text tower: UTF-8 bytes, RoBERTa-style specials (0 bos, 1 pad, 2 eos)."""

    vocab_size = 259

    def __call__(self, texts: list[str], max_length: int = 64, **_):
        ids = [[0] + [b + 3 for b in t.encode("utf-8")][: max_length - 2] + [2] for t in texts]
        width = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), width), 1, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


def build_text_encoder(name: str) -> tuple[nn.Module, object]:
    if name == TINY_TEXT:
        from transformers import RobertaConfig, RobertaModel

        cfg = RobertaConfig(vocab_size=ByteTokenizer.vocab_size, hidden_size=32, num_hidden_layers=2,
                            num_attention_heads=2, intermediate_size=64, max_position_embeddings=80,
                            pad_token_id=1, bos_token_id=0, eos_token_id=2)
        return RobertaModel(cfg, add_pooling_layer=False), ByteTokenizer()
    from transformers import AutoModel, AutoTokenizer

    return AutoModel.from_pretrained(name, add_pooling_layer=False), AutoTokenizer.from_pretrained(name)


def _set_trainable(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


# --- towers ----------------------------------------------------------------------
class AttentiveStatsPool(nn.Module):
    """Attention-weighted mean and standard deviation over valid frames."""

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(x).squeeze(-1).float().masked_fill(~mask, -1e4)
        w = torch.softmax(logits, dim=-1).unsqueeze(-1)
        xf = x.float()
        mean = (w * xf).sum(1)
        var = (w * (xf - mean.unsqueeze(1)) ** 2).sum(1)
        return torch.cat([mean, torch.sqrt(var.clamp_min(1e-6))], dim=-1)


class AudioTower(nn.Module):
    def __init__(self, encoder: str, embed_dim: int, finetune_layers: int, dropout: float,
                 spec_augment: bool = True) -> None:
        super().__init__()
        self.encoder = build_audio_encoder(encoder, spec_augment)
        cfg = self.encoder.config
        hidden = cfg.hidden_size
        # wav2vec2-base style models (group norm in the feature encoder) must not be given
        # an attention mask: their padding is meant to be zeros, not masked.
        self.use_mask = getattr(cfg, "feat_extract_norm", "layer") == "layer"
        self.finetune_layers = finetune_layers
        _set_trainable(self.encoder, False)
        if finetune_layers > 0:
            layers = self.encoder.encoder.layers
            for layer in layers[len(layers) - min(finetune_layers, len(layers)):]:
                _set_trainable(layer, True)
            norm = getattr(self.encoder.encoder, "layer_norm", None)
            if norm is not None and getattr(cfg, "do_stable_layer_norm", False):
                _set_trainable(norm, True)
        self.layer_logits = nn.Parameter(torch.zeros(cfg.num_hidden_layers + 1))
        self.pool = AttentiveStatsPool(hidden)
        self.proj = nn.Sequential(nn.Linear(2 * hidden, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout))

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, wave: torch.Tensor, wave_mask: torch.Tensor) -> torch.Tensor:
        kwargs = {"attention_mask": wave_mask} if self.use_mask else {}
        if self.finetune_layers == 0:
            with torch.no_grad():
                out = self.encoder(wave, output_hidden_states=True, **kwargs)
        else:
            out = self.encoder(wave, output_hidden_states=True, **kwargs)
        hs = torch.stack(out.hidden_states, dim=0)
        mixed = (self.layer_weights().view(-1, 1, 1, 1).to(hs.dtype) * hs).sum(0)
        lengths = self.encoder._get_feat_extract_output_lengths(wave_mask.sum(-1))
        frames = torch.arange(mixed.shape[1], device=mixed.device).unsqueeze(0)
        frame_mask = frames < lengths.unsqueeze(1)
        return self.proj(self.pool(mixed, frame_mask))


class TextTower(nn.Module):
    def __init__(self, encoder: str, embed_dim: int, finetune_layers: int, dropout: float) -> None:
        super().__init__()
        self.encoder, self.tokenizer = build_text_encoder(encoder)
        _set_trainable(self.encoder, False)
        if finetune_layers > 0:
            layers = self.encoder.encoder.layer
            for layer in layers[len(layers) - min(finetune_layers, len(layers)):]:
                _set_trainable(layer, True)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(hidden, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        m = attention_mask.unsqueeze(-1).float()
        return self.proj((h * m).sum(1) / m.sum(1).clamp_min(1.0))


class ProsodyTower(nn.Module):
    def __init__(self, n_features: int, embed_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- the model -------------------------------------------------------------------
class EmotionModel(nn.Module):
    MODALITIES = ("audio", "text", "prosody")

    def __init__(self, cfg: dict, n_prosody: int = 42) -> None:
        super().__init__()
        d = cfg["embed_dim"]
        drop = cfg["dropout"]
        towers: dict[str, nn.Module] = {}
        if cfg["audio"]["enabled"]:
            a = cfg["audio"]
            towers["audio"] = AudioTower(a["encoder"], d, a["finetune_layers"], drop, a["spec_augment"])
        if cfg["text"]["enabled"]:
            t = cfg["text"]
            towers["text"] = TextTower(t["encoder"], d, t["finetune_layers"], drop)
        if cfg["prosody"]["enabled"]:
            towers["prosody"] = ProsodyTower(n_prosody, d, cfg["prosody"]["hidden"], drop)
        if not towers:
            raise ValueError("model: enable at least one of audio, text, prosody")
        self.towers = nn.ModuleDict(towers)
        self.modalities = [m for m in self.MODALITIES if m in towers]
        self.gate = nn.ModuleDict({m: nn.Linear(d, 1) for m in self.modalities})
        self.fuse = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop))
        self.classifier = nn.Linear(d, len(EMOTIONS))
        self.regressor = nn.Linear(d, len(DIMENSIONS))
        self.modality_dropout = cfg["modality_dropout"]

    @property
    def tokenizer(self):
        return self.towers["text"].tokenizer if "text" in self.towers else None

    def forward(self, wave: torch.Tensor | None = None, wave_mask: torch.Tensor | None = None,
                input_ids: torch.Tensor | None = None, text_mask: torch.Tensor | None = None,
                has_text: torch.Tensor | None = None, prosody: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        embeds, present = [], []
        for m in self.modalities:
            if m == "audio":
                e = self.towers["audio"](wave, wave_mask)
                p = torch.ones(e.shape[0], dtype=torch.bool, device=e.device)
            elif m == "text":
                e = self.towers["text"](input_ids, text_mask)
                p = has_text.bool()
            else:
                e = self.towers["prosody"](prosody)
                p = torch.ones(e.shape[0], dtype=torch.bool, device=e.device)
            embeds.append(e.float())
            present.append(p)
        E = torch.stack(embeds, dim=1)                       # B, M, d
        P = torch.stack(present, dim=1)                      # B, M
        if self.training and self.modality_dropout > 0 and len(self.modalities) > 1:
            keep = torch.rand(P.shape, device=P.device) >= self.modality_dropout
            dropped = P & keep
            # never leave a clip with nothing: fall back to what it had
            P = torch.where(dropped.any(dim=1, keepdim=True), dropped, P)
        scores = torch.cat([self.gate[m](E[:, i]) for i, m in enumerate(self.modalities)], dim=1)
        alpha = torch.softmax(scores.masked_fill(~P, -1e4), dim=1)
        fused = (alpha.unsqueeze(-1) * E).sum(1)
        h = fused + self.fuse(fused)
        return {
            "logits": self.classifier(h),
            "dims": torch.sigmoid(self.regressor(h)),
            "alpha": alpha,
            "embedding": h,
        }

    def parameter_groups(self, lr_encoder: float, lr_head: float, weight_decay: float) -> list[dict]:
        """Pretrained encoder weights train slowly; everything new trains fast."""
        enc, head = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc if name.split(".")[2:3] == ["encoder"] else head).append(p)
        groups = [{"params": head, "lr": lr_head, "weight_decay": weight_decay}]
        if enc:
            groups.append({"params": enc, "lr": lr_encoder, "weight_decay": weight_decay})
        return groups


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
