"""Train Espresso-3D on Affogato-750K (pretraining) or LASO (finetuning), with a wall-clock budget.

Outputs in --out:
  status.json          live progress (epoch/step/elapsed/eta/last loss/last val)
  train_log.jsonl      every --log_every steps
  val_log.jsonl        every validation
  ckpt/last.pt         full resumable state (overwritten)
  ckpt/backup_*.pt     fp16 weights every --backup_every epochs (the newest --keep_backups are kept)
  ckpt/best.pt         fp16 weights with the best validation aIoU
  ckpt/final.pt        fp16 weights at the end of training

The learning-rate schedule is warmup + cosine over max(step / total_steps, elapsed / time_budget),
so a run can be bounded either by --epochs or by a wall-clock --time_budget_min.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from espresso.data import LASO, AffogatoObjects, collate_laso, collate_objects  # noqa: E402
from espresso.metrics import batch_metrics_torch, bce_dice_loss, summarize  # noqa: E402
from espresso.model import Espresso3D, partfield_checkpoint, read_weights  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["affogato", "laso"], required=True)
    p.add_argument("--setting", choices=["seen", "unseen"], default="seen")
    p.add_argument("--data_root", required=True)
    p.add_argument("--partfield_ckpt", default=None, help="PartField checkpoint (default: download the official one)")
    p.add_argument("--init", default=None, help="weights to initialise decoder + text encoder from (.pt, .safetensors or <hf_repo>:<file>)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=50)
    p.add_argument("--time_budget_min", type=float, default=1e9)
    p.add_argument("--bs", type=int, default=64, help="pairs per step (affogato: rounded to objects x 5)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--text_lr", type=float, default=5e-6)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--freeze_text", action="store_true")
    p.add_argument("--n_points", type=int, default=16384)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=float, default=1.0, help="epochs")
    p.add_argument("--backup_every", type=float, default=5.0, help="epochs")
    p.add_argument("--val_limit", type=int, default=0, help="max val objects/samples (0 = all)")
    p.add_argument("--limit", type=int, default=0, help="debug: limit train items")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache_bs", type=int, default=16, help="batch size when caching LASO PartField features")
    p.add_argument("--keep_backups", type=int, default=0, help="keep only the N newest backups (0 = keep all)")
    return p.parse_args()


def jdump(path, obj, mode="a"):
    with open(path, mode) as f:
        f.write(json.dumps(obj) + ("\n" if mode == "a" else ""))


def build_data(a):
    if a.dataset == "affogato":
        tr = AffogatoObjects(a.data_root, "train", n_points=a.n_points, augment=True, limit=a.limit or None)
        va = AffogatoObjects(a.data_root, "val", n_points=a.n_points, limit=a.val_limit or None)
        obj_bs = max(1, a.bs // 5)
        mk = lambda ds, sh, bs: DataLoader(ds, bs, shuffle=sh, num_workers=a.workers, collate_fn=collate_objects,
                                           pin_memory=True, drop_last=sh, persistent_workers=a.workers > 0,
                                           prefetch_factor=4 if a.workers else None)
        return mk(tr, True, obj_bs), mk(va, False, obj_bs), len(tr)
    tr = LASO(a.data_root, "train", a.setting)
    va = LASO(a.data_root, "val", a.setting)
    if a.limit:
        tr.set_anno(tr.anno[: a.limit])
    if a.val_limit:
        va.set_anno(va.anno[: a.val_limit])
    mk = lambda ds, sh: DataLoader(ds, a.bs, shuffle=sh, num_workers=a.workers, collate_fn=collate_laso,
                                   pin_memory=True, drop_last=sh, persistent_workers=a.workers > 0)
    return mk(tr, True), mk(va, False), len(tr)


@torch.no_grad()
def build_feature_cache(model, ds, dev, bs=32):
    """Frozen PartField features for every LASO shape: (n_shapes, 2048, 448) fp16 on GPU.
    Exact (no augmentation in the official LASO protocol) and ~10x faster than recomputing per step."""
    feats = None
    for i in range(0, len(ds.shape_ids), bs):
        xyz = torch.stack([ds.shape_points(s) for s in ds.shape_ids[i:i + bs]]).to(dev)
        f = model.vision(xyz).half()
        if feats is None:
            feats = torch.empty(len(ds.shape_ids), *f.shape[1:], dtype=torch.float16, device=dev)
        feats[i:i + len(f)] = f
    return feats


def to_dev(b, dev):
    out = dict(b)
    out["xyz"] = b["xyz"].to(dev, non_blocking=True)
    out["heatmap"] = b["heatmap"].to(dev, non_blocking=True)
    if b.get("obj_index") is not None:
        out["obj_index"] = b["obj_index"].to(dev, non_blocking=True)
    return out


@torch.no_grad()
def validate(model, loader, dev, cache=None):
    model.eval()
    mets, losses = [], []
    for b in loader:
        b = to_dev(b, dev)
        pf = cache[b["shape_idx"].to(dev)].float() if cache is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["xyz"], b["texts"], b["obj_index"], point_feat=pf)
        loss, _, _ = bce_dice_loss(logits, b["heatmap"])
        losses.append(loss.item())
        pred = logits.sigmoid()
        mets.append(batch_metrics_torch(pred, b["heatmap"]).cpu())
    model.train()
    if not mets:
        return None
    res = summarize(torch.cat(mets).numpy())
    res["loss"] = float(np.mean(losses))
    return res


def main():
    a = get_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    os.makedirs(os.path.join(a.out, "ckpt"), exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    train_loader, val_loader, n_train = build_data(a)
    steps_per_epoch = len(train_loader)
    total_steps = int(math.ceil(a.epochs * steps_per_epoch))
    print(f"[data] train items={n_train} steps/epoch={steps_per_epoch} total_steps(max)={total_steps}", flush=True)

    model = Espresso3D(partfield_checkpoint(a.partfield_ckpt)).to(dev)
    if a.freeze_text:
        model.text.requires_grad_(False)
    if a.init:
        sd = read_weights(a.init)  # .pt / .safetensors / "<hf_repo>:<file>"
        missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        print(f"[init] from {a.init}; unexpected={len(unexpected)}; "
              f"missing(non-vision)={[k for k in missing if not k.startswith('vision.')][:5]}", flush=True)
    dec, txt = model.trainable_parameters()
    groups = [{"params": dec, "lr": a.lr}]
    if txt:
        groups.append({"params": txt, "lr": a.text_lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=a.wd)
    base_lrs = [g["lr"] for g in groups]
    print(f"[model] decoder params={sum(p.numel() for p in dec)/1e6:.2f}M "
          f"text params(trainable)={sum(p.numel() for p in txt)/1e6:.1f}M", flush=True)

    caches = {}
    if a.dataset == "laso":
        tc = time.time()
        caches = {k: build_feature_cache(model, dl.dataset, dev, a.cache_bs) for k, dl in
                  (("train", train_loader), ("val", val_loader))}
        print(f"[cache] PartField features cached for {sum(len(c) for c in caches.values())} shapes "
              f"in {time.time()-tc:.0f}s", flush=True)

    step, epoch, elapsed_prev, best = 0, 0, 0.0, -1.0
    last_path = os.path.join(a.out, "ckpt", "last.pt")
    if a.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["opt"])
        step, epoch, elapsed_prev, best = st["step"], st["epoch"], st["elapsed"], st.get("best", -1.0)
        print(f"[resume] step={step} epoch={epoch} elapsed={elapsed_prev/60:.1f}min", flush=True)

    budget = a.time_budget_min * 60
    t0 = time.time()
    elapsed = lambda: elapsed_prev + time.time() - t0

    def progress():
        return max(step / max(total_steps, 1), elapsed() / budget)

    def set_lr():
        w = min(1.0, (step + 1) / a.warmup)
        c = 0.5 * (1 + math.cos(math.pi * min(progress(), 1.0)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * w * (0.01 + 0.99 * c)

    def save(path, full, extra=None):
        sd = model.decoder_state_dict()
        if full:
            obj = dict(model=sd, opt=opt.state_dict(), step=step, epoch=epoch, elapsed=elapsed(), best=best)
        else:
            obj = dict(model={k: v.half() if v.is_floating_point() else v for k, v in sd.items()},
                       step=step, epoch=epoch)
        obj.update(extra or {})
        torch.save(obj, path + ".tmp")
        os.replace(path + ".tmp", path)

    val_every = max(1, int(round(a.val_every * steps_per_epoch)))
    backup_every = max(1, int(round(a.backup_every * steps_per_epoch)))
    run_loss, run_n, done, last_val = 0.0, 0, False, None
    tstep = time.time()
    model.train()
    while not done:
        for b in train_loader:
            b = to_dev(b, dev)
            set_lr()
            pf = caches["train"][b["shape_idx"].to(dev)].float() if caches else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b["xyz"], b["texts"], b["obj_index"], point_feat=pf)
            loss, bce, dice = bce_dice_loss(logits, b["heatmap"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(dec + txt, 1.0)
            opt.step()
            step += 1
            run_loss += loss.item(); run_n += 1
            ep_float = step / steps_per_epoch

            if step % a.log_every == 0:
                dt = (time.time() - tstep) / a.log_every
                tstep = time.time()
                rec = dict(step=step, epoch=round(ep_float, 4), loss=run_loss / run_n, bce=bce.item(),
                           dice=dice.item(), gnorm=float(gn), lr=opt.param_groups[0]["lr"],
                           s_per_step=dt, elapsed_min=elapsed() / 60, progress=progress())
                jdump(os.path.join(a.out, "train_log.jsonl"), rec)
                run_loss, run_n = 0.0, 0
                eta = (1 - progress()) / max(progress(), 1e-9) * elapsed() / 60
                status = dict(rec, eta_min=eta, steps_per_epoch=steps_per_epoch, last_val=last_val,
                              best_val_aIoU=best, time=time.strftime("%H:%M:%S"))
                jdump(os.path.join(a.out, "status.json"), status, mode="w")
                print(f"[train] ep {ep_float:.3f} step {step} loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                      f"{dt:.2f}s/it elapsed {rec['elapsed_min']:.1f}m eta {eta:.1f}m", flush=True)

            finished = progress() >= 1.0
            if step % val_every == 0 or finished:
                last_val = validate(model, val_loader, dev, cache=caches.get("val")) or dict(aIoU=-1.0)
                last_val.update(step=step, epoch=round(ep_float, 4), elapsed_min=elapsed() / 60)
                jdump(os.path.join(a.out, "val_log.jsonl"), last_val)
                print(f"[val] step {step} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                      for k, v in last_val.items()), flush=True)
                if last_val["aIoU"] > best:
                    best = last_val["aIoU"]
                    save(os.path.join(a.out, "ckpt", "best.pt"), full=False, extra=dict(val=last_val))
                save(last_path, full=True)
            if step % backup_every == 0 or finished:
                save(os.path.join(a.out, "ckpt", f"backup_ep{ep_float:07.3f}.pt"), full=False)
                if a.keep_backups > 0:
                    for old in sorted(glob.glob(os.path.join(a.out, "ckpt", "backup_ep*.pt")))[:-a.keep_backups]:
                        os.remove(old)
                save(last_path, full=True)
            if finished:
                done = True
                break
        epoch += 1
    save(os.path.join(a.out, "ckpt", "final.pt"), full=False)
    jdump(os.path.join(a.out, "status.json"), dict(done=True, step=step, elapsed_min=elapsed() / 60,
                                                   best_val_aIoU=best, last_val=last_val), mode="w")
    print(f"[done] step={step} elapsed={elapsed()/60:.1f}min best_val_aIoU={best:.2f}", flush=True)


if __name__ == "__main__":
    main()
