"""Evaluate an Espresso-3D checkpoint on the LASO test split with the official metric definitions.

Reports overall aIoU / AUC / SIM / MAE (as in Affogato Table 2), per-object / per-affordance breakdowns,
and — for the Unseen setting — the subset of test samples whose (object, affordance) pair was held out
from training.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from espresso.data import LASO, LASO_AFFORDANCES, LASO_CLASSES, collate_laso  # noqa: E402
from espresso.metrics import batch_metrics, summarize  # noqa: E402
from espresso.model import load_espresso  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--partfield_ckpt", default=None, help="PartField checkpoint (default: download the official one)")
    p.add_argument("--setting", default="seen")
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--bs", type=int, default=32)
    a = p.parse_args()
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)

    model = load_espresso(a.ckpt, a.partfield_ckpt).to(dev).eval()

    ds = LASO(a.data_root, a.split, a.setting)
    dl = DataLoader(ds, a.bs, shuffle=False, num_workers=4, collate_fn=collate_laso)
    mets, cls, aff, held = [], [], [], []
    with torch.no_grad():
        for b in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b["xyz"].to(dev), b["texts"])
            pred = logits.sigmoid().float().cpu().numpy()
            gt = b["heatmap"].numpy()
            mets.append(batch_metrics(pred, gt))
            cls += b["cls"].tolist(); aff += b["aff"].tolist(); held += b["heldout"].tolist()
    m = np.concatenate(mets)
    cls, aff, held = np.array(cls), np.array(aff), np.array(held, dtype=bool)

    res = dict(ckpt=a.ckpt, setting=a.setting, split=a.split, overall=summarize(m))
    if held.any():
        res["heldout_pairs_only"] = summarize(m[held])
        res["seen_pairs_only"] = summarize(m[~held])
    res["per_object"] = {LASO_CLASSES[c]: summarize(m[cls == c]) for c in np.unique(cls)}
    res["per_affordance"] = {LASO_AFFORDANCES[f]: summarize(m[aff == f]) for f in np.unique(aff)}
    json.dump(res, open(os.path.join(a.out, f"laso_{a.setting}_{a.split}.json"), "w"), indent=1)
    o = res["overall"]
    print(f"[LASO {a.setting}/{a.split}] n={o['n']} aIoU={o['aIoU']:.1f} AUC={o['AUC']:.1f} "
          f"SIM={o['SIM']:.1f} MAE={o['MAE']:.3f}")
    if "heldout_pairs_only" in res:
        h = res["heldout_pairs_only"]
        print(f"   held-out pairs only n={h['n']} aIoU={h['aIoU']:.1f} AUC={h['AUC']:.1f} "
              f"SIM={h['SIM']:.1f} MAE={h['MAE']:.3f}")


if __name__ == "__main__":
    main()
