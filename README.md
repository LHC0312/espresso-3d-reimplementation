# Espresso-3D — unofficial re-implementation

An **unofficial** re-implementation of **Espresso-3D**, the 3D affordance-grounding model from
[*Affogato: Open-Vocabulary Affordance Grounding with Automated Data Generation at Scale*](https://arxiv.org/abs/2506.12009)
(Lee, Park, Park, Kang, Cho — POSTECH). It is pretrained on **Affogato-750K** and evaluated on **LASO**.

> **Not affiliated with the authors.** When this was written (Sep 2026) the official code
> (`junha-l/affordance_segmentation`) and Espresso weights were not public, so the model was rebuilt from the
> paper's Sec. 4 and Appendix B.2. Numbers below come from this implementation, not from the official one.

- Weights: [huggingface.co/LHC0312/espresso-3d-reimplementation](https://huggingface.co/LHC0312/espresso-3d-reimplementation)
- Given a point cloud and a free-form query ("Point to the part you would hold to carry it."), the model
  predicts a per-point affordance heatmap in [0, 1].

## Model

```
points (N×3) ──► PartField (frozen) ──► 448-d feature per point ──► MLP ─► v (N×256)
                                                                           │ + Fourier PE(xyz)
                                                                           ▼
query text ──► Recap-CLIP text tower ──► [sentence emb + 128 token embs] ─► 129 queries (256-d)
               (finetuned, lr 5e-6)                                        │
                                           3 × [cross-attn to points → self-attn → FFN]
                                                                           ▼
                             heatmap_i = σ( s · cos( MLP(v_i), ResFFN(query_0) ) + b )
```

| Part | Paper | This implementation |
|---|---|---|
| 3D encoder | PartField (PVCNN + triplane transformer), official checkpoint, frozen (B.2) | same — [`mikaelaangel/partfield-ckpt`](https://huggingface.co/mikaelaangel/partfield-ckpt), frozen |
| Text encoder | Recap-CLIP, finetuned "following PointRefer" | `UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP` text tower, lr 5e-6 |
| Decoder | text embeddings as cross-attention queries over point features + Fourier PE, residual FFN, heatmap = similarity | 3 layers, width 256, 8 heads; sentence embedding + token embeddings as queries (depth, width and projection MLPs are not specified in the paper) |
| Loss | BCE + Dice, equal weights (as LASO) | same |
| Points | 16,384 (Affogato), 2,048 (LASO) | same |

Point features do not depend on the query, so all queries of one object share one PartField pass and one
set of point embeddings. For LASO the frozen PartField features are cached once on the GPU (LASO uses no
augmentation, so this is exact).

## Results

Released checkpoints were trained on **one RTX PRO 6000 (96 GB)**: Affogato pretraining for 185 minutes
(**3.2 epochs**; the paper uses 50 epochs on 8× RTX A6000), then LASO finetuning for up to 45 minutes
(Seen 34 epochs, Unseen 44 epochs; paper: 50). "best" is the checkpoint with the highest **LASO validation**
aIoU; "final" is the last one (the official LASO code evaluates the last model).

**LASO test (2,416 pairs)** — metrics and protocol follow the official LASO evaluation (aIoU over 20
thresholds, ROC-AUC, SIM, MAE).

| Seen | aIoU ↑ | AUC ↑ | SIM ↑ | MAE ↓ |
|---|---|---|---|---|
| Paper · Espresso-3D + Affogato-750K pretrain | **21.9** | **85.9** | **63.7** | 0.116 |
| Paper · Espresso-3D (no pretrain) | 20.4 | 86.0 | 63.3 | 0.102 |
| Paper · PointRefer | 20.8 | 87.3 | 62.9 | **0.093** |
| **Ours · best (epoch 13)** | 19.5 | 84.1 | 60.2 | 0.099 |
| Ours · final (epoch 34) | 18.9 | 82.7 | 59.0 | 0.104 |

| Unseen | aIoU ↑ | AUC ↑ | SIM ↑ | MAE ↓ |
|---|---|---|---|---|
| Paper · Espresso-3D + Affogato-750K pretrain | **20.8** | **82.9** | **61.4** | 0.122 |
| Paper · Espresso-3D (no pretrain) | 18.7 | 80.0 | 60.0 | **0.101** |
| Paper · PointRefer | 14.6 | 80.2 | 50.7 | 0.119 |
| **Ours · best (epoch 9)** | 17.6 | 81.7 | 59.1 | 0.104 |
| Ours · final (epoch 44) | 16.2 | 77.7 | 54.0 | 0.116 |

Unseen, restricted to the 766 test pairs whose (object, affordance) pair is held out from training:
best 12.1 / 72.0 / 55.8 / 0.125, final 7.0 / 63.1 / 44.7 / 0.149 — longer LASO finetuning overfits the seen
pairs and loses the transfer to held-out pairs.

**Zero-shot (Affogato pretraining only, no LASO training):** aIoU 7.2 · AUC 63.4 · SIM 41.5 · MAE 0.157.
LASO's questions and binary human masks differ from Affogato's queries and soft heatmaps; the paper reports
LASO only after finetuning.

**Affogato validation** (our own object-level hold-out: 1,050 objects / 5,250 pairs; the paper's 5K
human-verified test list is not part of the public dataset): aIoU 13.98 · AUC 77.0 · SIM 48.4 · MAE 0.116
(paper, official test split: 13.6 · 79.0 · 42.9 · 0.111 — not directly comparable).

PartField's scatter operations and bf16 autocast are not bit-deterministic on GPU, so repeated evaluations
differ slightly (logits by ~0.05; metrics typically by a few tenths).

### LASO Unseen split

The public LASO release only contains the Seen split ([issue](https://github.com/yl3800/LASO/issues/7)).
The Unseen training set is rebuilt by removing the (affordance → objects) pairs of LASO supplementary Table 6
(`espresso/data.py: LASO_UNSEEN_REMOVED`), which reproduces the paper's count exactly (16,120 → 11,558
training pairs). As in LASO, Seen and Unseen share the validation and test sets.

## Usage

```bash
pip install -r requirements.txt     # PyTorch with CUDA
```

Inference (weights and the PartField checkpoint are downloaded from the Hugging Face Hub):

```bash
python inference.py --points object.npy \
  --query "Point to the part you would hold to carry it." \
  --weights LHC0312/espresso-3d-reimplementation:espresso3d_laso_seen_best.safetensors \
  --out heatmap.npy
```

```python
import torch
from espresso import load_espresso

model = load_espresso("LHC0312/espresso-3d-reimplementation:espresso3d_affogato_pretrain.safetensors").cuda().eval()
xyz = torch.randn(1, 16384, 3).cuda()                     # one point cloud (any N)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    heat = model(xyz, ["Point to the part you would press to turn it on."]).sigmoid()   # (1, N)
```

### Weights

| File | Stage |
|---|---|
| `espresso3d_affogato_pretrain.safetensors` | Affogato-750K pretraining, 185 min (3.2 epochs) |
| `espresso3d_laso_seen_best.safetensors` / `_final` | + LASO Seen finetuning (epoch 13 / 34) |
| `espresso3d_laso_unseen_best.safetensors` / `_final` | + LASO Unseen finetuning (epoch 9 / 44) |

Each file (fp16, 341M parameters, 683 MB) holds the decoder and the finetuned Recap-CLIP text tower.
PartField weights are **not** included; they are fetched from the official repository.

### Training

```bash
bash scripts/download_data.sh          # Affogato-750K (~39 GB extracted) + LASO into ./data
bash scripts/run_all.sh                # pretrain (PRE_MIN=185) -> LASO Seen/Unseen finetune -> test eval
```

Or step by step:

```bash
python train.py --dataset affogato --data_root data/affogato --out runs/pretrain \
  --time_budget_min 185 --bs 80 --workers 12 --val_every 0.25 --backup_every 0.25
python train.py --dataset laso --setting unseen --data_root data/LASO_dataset \
  --init runs/pretrain/ckpt/final.pt --out runs/laso_unseen --epochs 50 --bs 64
python eval_laso.py --ckpt runs/laso_unseen/ckpt/best.pt --data_root data/LASO_dataset \
  --setting unseen --out runs/eval
```

Throughput on one RTX PRO 6000: 0.36 s per Affogato step (16 objects × 5 queries × 16,384 points),
≈ 58 min per Affogato epoch; ≈ 1 min per LASO epoch with cached PartField features. The released run was
executed on a Google Colab G4 runtime through the [Colab CLI](https://github.com/googlecolab/google-colab-cli).

A 0.7% object-level validation split of Affogato is taken by hashing object ids
(`espresso/data.py: affogato_split`), so it is reproducible without any split file.

## Repository layout

```
espresso/     model.py (Espresso-3D, loaders) · data.py (Affogato, LASO) · metrics.py (loss, LASO metrics)
partfield/    PartField model code (NVIDIA), vendored with small dependency patches — see partfield/NOTICE.md
train.py      pretraining / finetuning with a wall-clock budget, resumable
eval_laso.py  LASO test evaluation (overall, per object, per affordance, held-out pairs)
inference.py  heatmap prediction for your own point cloud
scripts/      data download and the full training recipe
```

## License

- Code written for this repository: MIT (`LICENSE`).
- `partfield/`: NVIDIA License, **non-commercial research and educational use only** (`partfield/LICENSE`).
  The model cannot run without PartField, so the project as a whole is usable for non-commercial purposes only.
- Released weights: CC BY-NC 4.0. They contain a finetuned copy of the Recap-CLIP text tower
  (UCSC-VLAA, CC BY 4.0) and were trained on Affogato-750K and LASO; follow those datasets' terms.

## Citation

Please cite the original works:

```bibtex
@article{lee2025affogato,
  title   = {Affogato: Open-Vocabulary Affordance Grounding with Automated Data Generation at Scale},
  author  = {Lee, Junha and Park, Eunha and Park, Chunghyun and Kang, Dahyun and Cho, Minsu},
  journal = {arXiv preprint arXiv:2506.12009},
  year    = {2025}
}
@inproceedings{liu2025partfield,
  title     = {PartField: Learning 3D Feature Fields for Part Segmentation and Beyond},
  author    = {Liu, Minghua and Uy, Mikaela Angelina and Xiang, Donglai and Su, Hao and Fidler, Sanja and Sharp, Nicholas and Gao, Jun},
  booktitle = {ICCV},
  year      = {2025}
}
@inproceedings{li2024laso,
  title     = {LASO: Language-guided Affordance Segmentation on 3D Object},
  author    = {Li, Yicong and Zhao, Na and Xiao, Junbin and Feng, Chun and Wang, Xiang and Chua, Tat-Seng},
  booktitle = {CVPR},
  year      = {2024}
}
@article{li2024recap,
  title   = {What If We Recaption Billions of Web Images with LLaMA-3?},
  author  = {Li, Xianhang and Tu, Haoqin and Hui, Mude and Wang, Zeyu and Zhao, Bingchen and Xiao, Junfei and Ren, Sucheng and Mei, Jieru and Liu, Qing and Zheng, Huangjie and others},
  journal = {arXiv preprint arXiv:2406.08478},
  year    = {2024}
}
```

---

## 한국어 요약

Affogato 논문의 3D 어포던스 모델 **Espresso-3D를 논문만 보고 다시 구현한 비공식 코드**입니다.
원저자와 관계없으며, 작성 시점(2026년 9월)에 공식 코드와 가중치는 공개되지 않은 상태였습니다.

- **구조**: PartField(고정)로 점마다 448차원 특징을 만들고, Recap-CLIP 텍스트 임베딩을 쿼리로 하는
  크로스 어텐션 디코더가 점별 히트맵을 예측합니다.
- **학습**: RTX PRO 6000 한 장에서 Affogato 사전학습 185분(3.2에폭, 논문은 50에폭) 후
  LASO Seen/Unseen 파인튜닝을 했습니다.
- **결과 (LASO test, best)**: Seen aIoU 19.5 / AUC 84.1 / SIM 60.2 / MAE 0.099,
  Unseen 17.6 / 81.7 / 59.1 / 0.104. 논문(Seen 21.9, Unseen 20.8 aIoU)보다 2~3포인트 낮습니다.
- **LASO Unseen 분할**: 공개 데이터에 없어서 LASO 보충자료 Table 6으로 재구성했고,
  학습 샘플 수가 논문과 정확히 일치합니다(11,558개).
- **라이선스**: 직접 작성한 코드는 MIT, PartField 부분은 NVIDIA 비상업 라이선스라
  **전체적으로 비상업적 연구·교육 용도로만** 쓸 수 있습니다.
