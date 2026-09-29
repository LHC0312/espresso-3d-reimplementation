"""Datasets: Affogato-750K (pretraining) and LASO (finetuning / evaluation)."""
import json
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- Affogato
# Each object dir: xyzc.npy (16384, 8) float16 = xyz + 5 soft heatmaps; queries.json (5 queries).


def affogato_split(obj_id, val_per_mille=7):
    """Deterministic object-level hold-out (≈0.7% of objects ≈ 1K objects / 5K pairs) for monitoring."""
    return "val" if zlib.crc32(obj_id.encode()) % 1000 < val_per_mille else "train"


class AffogatoObjects(Dataset):
    """One item = one object with all of its 5 query/heatmap pairs."""

    def __init__(self, root, split="train", index_file=None, n_points=16384, augment=False, limit=None):
        self.root, self.n_points, self.augment = root, n_points, augment
        index_file = index_file or os.path.join(root, "index.txt")
        if os.path.exists(index_file):
            ids = [l.strip() for l in open(index_file) if l.strip() and not l.strip().endswith(".txt")]
        else:
            ids = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.ids = [i for i in ids if affogato_split(i) == split]
        if limit:
            self.ids = self.ids[:limit]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = os.path.join(self.root, self.ids[i])
        a = np.load(os.path.join(d, "xyzc.npy")).astype(np.float32)
        with open(os.path.join(d, "queries.json")) as f:
            meta = json.load(f)[0]
        xyz, hm = a[:, :3], a[:, 3:8].T  # (N,3), (5,N)
        queries = list(meta["queries"])
        if self.n_points < len(xyz):
            sel = np.random.choice(len(xyz), self.n_points, replace=False)
            xyz, hm = xyz[sel], hm[:, sel]
        if self.augment:
            xyz = xyz * np.random.uniform(0.9, 1.1, size=(1, 3)).astype(np.float32)
        return dict(xyz=torch.from_numpy(xyz), heatmap=torch.from_numpy(np.clip(hm, 0, 1)),
                    queries=queries, cls=meta.get("class_name", ""), obj_id=self.ids[i])


def collate_objects(batch):
    """-> xyz (B_obj,N,3), heatmap (B_pair,N), texts [B_pair], obj_index (B_pair,)"""
    xyz = torch.stack([b["xyz"] for b in batch])
    hms, texts, idx, meta = [], [], [], []
    for j, b in enumerate(batch):
        for k, q in enumerate(b["queries"]):
            hms.append(b["heatmap"][k]); texts.append(q); idx.append(j)
            meta.append((b["obj_id"], b["cls"]))
    return dict(xyz=xyz, heatmap=torch.stack(hms), texts=texts, obj_index=torch.tensor(idx), meta=meta)


# --------------------------------------------------------------------------- LASO
LASO_CLASSES = ["Bag", "Bed", "Bowl", "Clock", "Dishwasher", "Display", "Door", "Earphone", "Faucet",
                "Hat", "StorageFurniture", "Keyboard", "Knife", "Laptop", "Microwave", "Mug",
                "Refrigerator", "Chair", "Scissors", "Table", "TrashCan", "Vase", "Bottle"]
LASO_AFFORDANCES = ["lay", "sit", "support", "grasp", "lift", "contain", "open", "wrap_grasp", "pour",
                    "move", "display", "push", "pull", "listen", "wear", "press", "cut", "stab"]
# LASO supplementary Table 6: (affordance -> object classes) removed from training in the Unseen setting.
LASO_UNSEEN_REMOVED = {"contain": ["microwave", "vase"], "cut": ["scissors"], "display": ["display"],
                       "grasp": ["mug", "scissors"], "move": ["table"], "open": ["microwave", "trashcan"],
                       "pour": ["mug", "trashcan"], "press": ["keyboard"], "stab": ["scissors"],
                       "support": ["chair"], "wrap_grasp": ["vase"]}


def laso_is_heldout(cls, aff):
    return cls.lower() in LASO_UNSEEN_REMOVED.get(aff, [])


def pc_normalize(pc):
    pc = pc - pc.mean(0)
    return pc / np.max(np.sqrt((pc ** 2).sum(1)))


class LASO(Dataset):
    """Mirrors the official LASO loader (data_utils/shapenetpart.py::AffordQ).

    setting='seen'  : official released split.
    setting='unseen': training set without the (affordance, object) pairs of LASO supp. Table 6
                      (16120 -> 11558 train instances, exactly as reported). Val/test are shared.
    """

    def __init__(self, root, split="train", setting="seen"):
        self.split = split
        self.cls2idx = {c.lower(): i for i, c in enumerate(LASO_CLASSES)}
        self.aff2idx = {a: i for i, a in enumerate(LASO_AFFORDANCES)}
        with open(os.path.join(root, f"anno_{split}.pkl"), "rb") as f:
            anno = pickle.load(f)
        with open(os.path.join(root, f"objects_{split}.pkl"), "rb") as f:
            self.objects = pickle.load(f)
        if setting == "unseen" and split == "train":
            anno = [a for a in anno if not laso_is_heldout(a["class"], a["affordance"])]
        self.set_anno(anno)
        df = pd.read_csv(os.path.join(root, "Affordance-Question.csv"))
        self.questions = {(r["Object"], r["Affordance"]): r for _, r in df.iterrows()}

    def set_anno(self, anno):
        self.anno = anno
        self.shape_ids = sorted({str(a["shape_id"]) for a in anno})
        self.shape_index = {sid: i for i, sid in enumerate(self.shape_ids)}

    def __len__(self):
        return len(self.anno)

    def shape_points(self, sid):
        return torch.from_numpy(pc_normalize(np.asarray(self.objects[sid], dtype=np.float32)))

    def question(self, cls, aff):
        qid = f"Question{np.random.randint(1, 15)}" if self.split == "train" else "Question0"
        return self.questions[(cls, aff)][qid]

    def __getitem__(self, i):
        a = self.anno[i]
        pts = pc_normalize(np.asarray(self.objects[str(a["shape_id"])], dtype=np.float32))
        return dict(xyz=torch.from_numpy(pts.astype(np.float32)),
                    heatmap=torch.from_numpy(np.asarray(a["mask"], dtype=np.float32)),
                    text=self.question(a["class"], a["affordance"]),
                    cls=self.cls2idx[a["class"]], aff=self.aff2idx[a["affordance"]],
                    heldout=laso_is_heldout(a["class"], a["affordance"]), shape_id=a["shape_id"],
                    shape_idx=self.shape_index[str(a["shape_id"])])


def collate_laso(batch):
    return dict(xyz=torch.stack([b["xyz"] for b in batch]), heatmap=torch.stack([b["heatmap"] for b in batch]),
                texts=[b["text"] for b in batch], obj_index=None,
                cls=torch.tensor([b["cls"] for b in batch]), aff=torch.tensor([b["aff"] for b in batch]),
                heldout=torch.tensor([b["heldout"] for b in batch]),
                shape_idx=torch.tensor([b["shape_idx"] for b in batch]),
                meta=[(b["shape_id"], LASO_CLASSES[b["cls"]] + "/" + LASO_AFFORDANCES[b["aff"]]) for b in batch])
