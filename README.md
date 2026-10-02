# FACETS: Cross-Granularity Vision–Language Modeling for 3D Anomaly Detection

Official PyTorch implementation of **FACETS** (NeurIPS 2026).

**Yuchuan Li, Jae-Mo Kang, Il-Min Kim** | [Paper](https://openreview.net/forum?id=GU2KPq4O7s)

## Installation

```bash
conda create -n facets python=3.11 -y
conda activate facets
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128  # match your CUDA version
pip install -r requirements.txt
```

## Pretrained backbone

The frozen ULIP-2 Point-BERT checkpoint pre-trained on Objaverse only:

```bash
wget -P pretrained https://huggingface.co/datasets/SFXX/ulip/resolve/main/ULIP-2/pretrained_models/ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt
```

## Datasets

- [Real3D-AD](https://github.com/M-3LAB/Real3D-AD) (PCD version): place the category folders in `dataset/Real3D-AD/`.
- [Anomaly-ShapeNet](https://github.com/Chopper-233/Anomaly-ShapeNet): place the `pcd` folder in `dataset/Anomaly-ShapeNet/`. We use the 40 categories of v1; the 10 categories added in v2 are kept in a `new` folder, which is skipped.

Other locations can be set with `--data-root`.

## Training

```bash
torchrun --standalone --nproc_per_node=4 -m facets.train --config configs/real3d_ad.json
torchrun --standalone --nproc_per_node=4 -m facets.train --config configs/anomaly_shapenet.json
```

The configs hold the hyperparameters used in the paper; `batch_size` is the global batch size across GPUs.

## Evaluation

```bash
python -m facets.evaluate --run-dir runs/real3d
python -m facets.evaluate --run-dir runs/shapenet
```

Per-category and mean O-AUROC / P-AUROC are printed and saved to `eval_results.json` in the run directory.

## Citation

```bibtex
@inproceedings{li2026facets,
  title     = {{FACETS}: Cross-Granularity Vision--Language Modeling for {3D} Anomaly Detection},
  author    = {Li, Yuchuan and Kang, Jae-Mo and Kim, Il-Min},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgments

This code builds on [ULIP-2](https://github.com/salesforce/ULIP), [Point-BERT](https://github.com/lulutang0608/Point-BERT) and [OpenCLIP](https://github.com/mlfoundations/open_clip).

## License

[MIT](LICENSE)
