"""Predict an affordance heatmap for a point cloud and a natural-language query.

Example:
  python inference.py --points mug.npy --query "Point to the part you would hold to drink." \
      --weights LHC0312/espresso-3d-reimplementation:espresso3d_laso_seen_best.safetensors --out heatmap.npy

--points: .npy (N x >=3, first three columns are xyz) or a whitespace-separated text file (.txt / .xyz).
The output is an (N,) float array in [0, 1], one score per input point.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from espresso.model import load_espresso  # noqa: E402

DEFAULT_WEIGHTS = "LHC0312/espresso-3d-reimplementation:espresso3d_affogato_pretrain.safetensors"


def load_points(path):
    pts = np.load(path) if path.endswith(".npy") else np.loadtxt(path)
    return np.asarray(pts, dtype=np.float32)[:, :3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points", required=True)
    p.add_argument("--query", required=True, action="append", help="can be given several times")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--partfield_ckpt", default=None)
    p.add_argument("--out", default="heatmap.npy")
    a = p.parse_args()

    model = load_espresso(a.weights, a.partfield_ckpt).cuda().eval()
    xyz = torch.from_numpy(load_points(a.points))[None].cuda()          # (1, N, 3)
    with torch.no_grad():
        feat = model.vision(xyz)                                         # frozen PartField, once
        feat = feat.expand(len(a.query), -1, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xyz.expand(len(a.query), -1, -1), a.query, point_feat=feat)
    heat = logits.sigmoid().float().cpu().numpy()                       # (Q, N)
    np.save(a.out, heat[0] if len(a.query) == 1 else heat)
    for q, h in zip(a.query, heat):
        print(f"{q!r}: max={h.max():.3f}  points>0.5: {(h > 0.5).mean() * 100:.1f}%")
    print(f"saved {a.out} {heat.shape if len(a.query) > 1 else heat[0].shape}")


if __name__ == "__main__":
    main()
