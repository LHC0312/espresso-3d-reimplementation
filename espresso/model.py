"""Espresso-3D — unofficial re-implementation from the Affogato paper (Lee et al., arXiv:2506.12009).

The official code and weights were not public at the time of writing, so this module follows the paper's
Sec. 4 and Appendix B.2. Details the paper leaves open (decoder depth/width, projection MLPs, use of text
tokens as extra queries) are this implementation's choices.

Components (paper Sec. 4 + Appendix B.2):
  * 3D vision encoder : pretrained PartField (PVCNN + triplane transformer), frozen.
  * text encoder      : Recap-CLIP text tower (UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP), finetuned
                        with a small lr (following PointRefer).
  * heatmap decoder   : text embeddings are the queries of a transformer mask decoder. Point
                        features + Fourier positional encodings are keys/values. The attended
                        text query goes through a residual FFN and the heatmap is the similarity
                        between point embeddings and the conditioned text embedding.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from partfield.model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat
from partfield.model.triplane import TriplaneTransformer


# --------------------------------------------------------------------------- PartField
class _Cfg(dict):
    __getattr__ = dict.__getitem__


def _partfield_pvcnn_cfg():
    # configs/final/demo.yaml + partfield/config/defaults.py
    return _Cfg(point_encoder_type="pvcnn", use_point_scatter=True,
                z_triplane_channels=256, z_triplane_resolution=128,
                unet_cfg=_Cfg(depth=3, enabled=True, rolled=True, use_3d_aware=True,
                              start_hidden_channels=32, use_initial_conv=False))


class PartFieldEncoder(nn.Module):
    """Frozen PartField point-feature extractor: (B, N, 3) -> (B, N, 448)."""

    out_dim = 448

    def __init__(self, ckpt_path=None):
        super().__init__()
        self.pvcnn = TriPlanePC2Encoder(_partfield_pvcnn_cfg(), device="cuda",
                                        shape_min=-1, shape_length=2, use_2d_feat=False)
        self.triplane_transformer = TriplaneTransformer(
            input_dim=128 * 2, transformer_dim=1024, transformer_layers=6, transformer_heads=8,
            triplane_low_res=32, triplane_high_res=128, triplane_dim=512)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            own = {k: v for k, v in sd.items()
                   if k.startswith("pvcnn.") or k.startswith("triplane_transformer.")}
            missing, unexpected = self.load_state_dict(own, strict=False)
            assert not missing, f"PartField missing keys: {missing[:5]}"
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):  # always frozen / eval
        return super().train(False)

    @staticmethod
    def normalize(xyz):
        # PartField demo normalisation: bbox centre -> origin, longest side -> 1.8 ([-0.9, 0.9])
        bbmin, bbmax = xyz.min(1, keepdim=True).values, xyz.max(1, keepdim=True).values
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.9 / (bbmax - bbmin).max(-1, keepdim=True).values.clamp(min=1e-6)
        return (xyz - center) * scale

    @torch.no_grad()
    def forward(self, xyz):
        pc = self.normalize(xyz.float())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            planes = self.pvcnn(pc, pc)
            planes = self.triplane_transformer(planes)
            part_planes = planes[:, :, 64:]  # first 64 channels are the SDF planes
            feat = sample_triplane_feat(part_planes, pc)
        return feat.float()


# --------------------------------------------------------------------------- Recap-CLIP
class RecapCLIPText(nn.Module):
    """Recap-CLIP text tower. Returns pooled (B, 768) and token (B, L, 1024) features."""

    HUB = "hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP"

    def __init__(self, name=HUB):
        super().__init__()
        import open_clip
        clip, _, _ = open_clip.create_model_and_transforms(name)
        self.tokenizer = open_clip.get_tokenizer(name)
        # Recap-CLIP HTxt: BERT tokenizer, bidirectional (no causal mask) 24x1024 text transformer,
        # pool_type='last', projection 1024 -> 768
        self.token_embedding = clip.token_embedding
        self.positional_embedding = clip.positional_embedding
        self.transformer = clip.transformer
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection
        self.attn_mask = clip.attn_mask
        self.text_pool_type = clip.text_pool_type
        del clip
        self.width = self.token_embedding.weight.shape[1]
        self.embed_dim = self.text_projection.shape[1]

    def forward(self, texts, device):
        tok = self.tokenizer(texts).to(device)  # (B, 128), [CLS] ... pad id 0
        x = self.token_embedding(tok) + self.positional_embedding[: tok.shape[1]]
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        if self.text_pool_type == "last":
            pooled = x[:, -1]
        elif self.text_pool_type == "argmax":
            pooled = x[torch.arange(x.shape[0]), tok.argmax(-1)]
        else:
            pooled = x[:, 0]
        pooled = pooled @ self.text_projection
        pad = tok == 0
        return pooled, x, pad


# --------------------------------------------------------------------------- decoder
class FourierPE(nn.Module):
    def __init__(self, dim, n_freq=16, max_freq=64.0):
        super().__init__()
        self.register_buffer("freqs", torch.logspace(0, math.log2(max_freq), n_freq, base=2.0) * math.pi,
                             persistent=False)
        self.proj = nn.Sequential(nn.Linear(3 + 6 * n_freq, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, xyz):
        x = xyz.unsqueeze(-1) * self.freqs  # B N 3 F
        x = torch.cat([xyz, x.sin().flatten(-2), x.cos().flatten(-2)], -1)
        return self.proj(x)


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden=None, n_blocks=2):
        super().__init__()
        hidden = hidden or dim * 4
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            for _ in range(n_blocks))

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return x


class DecoderLayer(nn.Module):
    """Mask2Former-style query layer: text queries cross-attend to point features (keys/values = point
    features + Fourier PE), then self-attend among the query tokens of the same text, then FFN."""

    def __init__(self, dim, heads=8):
        super().__init__()
        self.n1, self.cross = nn.LayerNorm(dim), nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2, self.self_attn = nn.LayerNorm(dim), nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n3, self.ffn = nn.LayerNorm(dim), nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                                              nn.Linear(4 * dim, dim))

    def forward(self, q, q_pad, k, v, k_per_obj):
        """q: (P, L, D) queries of P texts; k, v: (O, N, D) per-object keys/values; k_per_obj pairs per object.
        Cross-attention is row-independent, so the k texts of one object are concatenated into one query
        sequence and attend to that object's points once (identical to per-pair attention, 5x cheaper)."""
        P, L, D = q.shape
        h = self.n1(q).reshape(P // k_per_obj, k_per_obj * L, D)
        q = q + self.cross(h, k, v, need_weights=False)[0].reshape(P, L, D)
        h = self.n2(q)
        q = q + self.self_attn(h, h, h, key_padding_mask=q_pad, need_weights=False)[0]
        return q + self.ffn(self.n3(q))


class Espresso3D(nn.Module):
    def __init__(self, partfield_ckpt=None, dim=256, n_layers=3, heads=8, text_encoder=None):
        super().__init__()
        self.vision = PartFieldEncoder(partfield_ckpt)
        self.text = text_encoder if text_encoder is not None else RecapCLIPText()
        self.point_in = nn.Sequential(nn.LayerNorm(PartFieldEncoder.out_dim),
                                      nn.Linear(PartFieldEncoder.out_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pos = FourierPE(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.tok_in = nn.Linear(self.text.width, dim)
        self.pool_in = nn.Linear(self.text.embed_dim, dim)
        self.layers = nn.ModuleList(DecoderLayer(dim, heads) for _ in range(n_layers))
        self.q_out = nn.Sequential(ResidualMLP(dim), nn.LayerNorm(dim))   # residual FFN on the text query
        self.p_out = nn.Sequential(ResidualMLP(dim, n_blocks=1), nn.LayerNorm(dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-2.0))

    def trainable_parameters(self):
        dec = [p for n, p in self.named_parameters()
               if p.requires_grad and not n.startswith("text.") and not n.startswith("vision.")]
        txt = [p for p in self.text.parameters() if p.requires_grad]
        return dec, txt

    def decoder_state_dict(self):
        """Everything except the frozen PartField weights (saved separately / re-downloaded)."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("vision.")}

    def forward(self, xyz, texts, obj_index=None, point_feat=None):
        """xyz: (B_obj, N, 3); texts: list of B_pair strings; obj_index: (B_pair,) object of each pair
        (contiguous, equal count per object, as built by collate_objects). Returns logits (B_pair, N)."""
        if point_feat is None:
            point_feat = self.vision(xyz)
        n_obj = xyz.shape[0]
        k = len(texts) // n_obj
        if obj_index is not None:
            assert k * n_obj == len(texts) and torch.equal(
                obj_index, torch.arange(n_obj, device=obj_index.device).repeat_interleave(k))
        pooled, tokens, pad = self.text(texts, xyz.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = self.point_in(point_feat)                       # (O, N, D) per object
            key = self.kv_norm(v + self.pos(PartFieldEncoder.normalize(xyz)))
            q = torch.cat([self.pool_in(pooled).unsqueeze(1), self.tok_in(tokens)], 1)
            q_pad = torch.cat([pad.new_zeros(pad.shape[0], 1), pad], 1)
            for layer in self.layers:
                q = layer(q, q_pad, key, key, k)
            t = F.normalize(self.q_out(q[:, 0]), dim=-1).view(n_obj, k, -1)   # conditioned text embedding
            p = F.normalize(self.p_out(v), dim=-1)                            # (O, N, D)
            logits = torch.einsum("ond,okd->okn", p, t).reshape(n_obj * k, -1)
        return logits.float() * self.logit_scale.exp() + self.logit_bias


# --------------------------------------------------------------------------- loading helpers
PARTFIELD_HF = ("mikaelaangel/partfield-ckpt", "model_objaverse.ckpt")  # official PartField (Objaverse) weights


def partfield_checkpoint(path=None):
    """Local PartField checkpoint path, downloading the official one from the Hugging Face Hub if needed."""
    if path:
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(*PARTFIELD_HF)


def read_weights(weights):
    """weights: local .pt / .safetensors file, or '<hf_repo_id>:<filename>' on the Hugging Face Hub."""
    import os
    if not os.path.exists(weights) and ":" in weights:
        from huggingface_hub import hf_hub_download
        repo, fname = weights.split(":", 1)
        weights = hf_hub_download(repo, fname)
    if weights.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(weights)
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    return sd.get("model", sd)


def load_espresso(weights, partfield_ckpt=None):
    """Build Espresso-3D with frozen PartField and load decoder + text-encoder weights."""
    model = Espresso3D(partfield_checkpoint(partfield_ckpt))
    sd = {k: v.float() for k, v in read_weights(weights).items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("vision.")]
    assert not missing and not unexpected, f"weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
    return model
