# PartField (vendored)

The files in this directory are taken from [nv-tlabs/PartField](https://github.com/nv-tlabs/PartField)
(`partfield/model/`), © NVIDIA Corporation, and are distributed under the NVIDIA License in `LICENSE`
(non-commercial research and educational use only).

Only the model definition needed to extract PartField features is included. Modifications:

- `model/PVCNN/encoder_pc.py`: `torch_scatter.scatter_mean` is replaced by an equivalent pure-PyTorch
  `scatter_mean`, and the import of `dnnlib_util` (profiling helpers, requires `boto3`) is removed.
- `model/PVCNN/dnnlib_util.py`: removed (unused at inference).
- Empty `__init__.py` files were added so the directories import as packages.

Pretrained weights are not redistributed; `espresso.model.partfield_checkpoint()` downloads the official
checkpoint from [mikaelaangel/partfield-ckpt](https://huggingface.co/mikaelaangel/partfield-ckpt).
