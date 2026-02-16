# Off-Road Segmentation Pipeline

This document describes the full workflow:
1. Continue training from NVIDIA SegFormer weights.
2. Fine-tune with `CAL` (Confusion-Aware Loss).
3. Generate segmentation outputs.
4. Merge segmentation + RGB video using `Merge_Offroad_Semantic_refactored.py`.

## Key Files

- `Merge_Offroad_Semantic_refactored.py`  
  Video merge script (semantic + photo) that creates `hybrid` and `mask`.
- `/Users/omerspring/Desktop/ReTrainCityScapesCAL.py`  
  Cityscapes training/continued-training script with `CAL/OHEM`.
- `/Users/omerspring/Desktop/metrics.py`  
  Metrics implementation: IoU, mIoU, F1, Pixel Accuracy, confusion matrix.
- `/Users/omerspring/Desktop/config_fn.json`  
  Training configuration file.

## Requirements

- Python 3.9+
- PyTorch
- Transformers (Hugging Face)
- Evaluate
- Datasets
- NumPy, Pandas, Matplotlib, Pillow, tqdm
- OpenCV (`opencv-python`)

Example install:

```bash
pip install torch transformers evaluate datasets numpy pandas matplotlib pillow tqdm opencv-python
```

## Step 1: Continue Training from NVIDIA + CAL

Base model in config:
- `BASE_MODEL_NAME`: `nvidia/segformer-b1-finetuned-cityscapes-1024-1024`

Config example (from `config_fn.json`):

```json
{
  "NUM_EPOCHS": 500,
  "BATCH_SIZE": 6,
  "BASE_MODEL_NAME": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
  "EXPERIMENT_NAME": "segformer-b1 second-finetuned-cityscapes-1024-1024-cal-new",
  "DATA_SET_PATH": "/gpfs0/bgu-hadar/users/itaidro/data/CityScape",
  "LOSS_TYPE": "cal_ohem",
  "CONFUSION_TYPE": "fn"
}
```

Run training:

```bash
python3 /Users/omerspring/Desktop/ReTrainCityScapesCAL.py /Users/omerspring/Desktop/config_fn.json
```

### What Happens During Training

- If a checkpoint exists: training resumes from `last_checkpoint.pt`.
- If no checkpoint exists: pretrained NVIDIA weights are loaded.
- `LOSS_TYPE`:
  - `cal_ohem` -> `CALoss`
  - `ohem` -> `OHEMCrossEntropy`
- `CONFUSION_TYPE`:
  - `fn` or `fn_fp` for confusion-matrix-based class weighting.
- Logs and metrics are saved every epoch.

### Training Outputs

In:

```text
./experiments/<EXPERIMENT_NAME>/
```

Files:
- `training_log.csv`
- `last_checkpoint.pt`
- `best_model.pt`

## Step 2: Generate Segmentation Video

After training, run inference to generate a color-coded semantic video.  
This depends on your inference script, but it should produce:
- `semantic.avi`

You also need the source RGB video:
- `photo.avi`

## Step 3: Merge Offroad (Spatial Overlay)

Basic run from project directory:

```bash
python3 Merge_Offroad_Semantic_refactored.py
```

Defaults:
- Input: `semantic.avi`, `photo.avi`
- Output: `hybrid.avi`, `mask.avi`

Run with parameters:

```bash
python3 Merge_Offroad_Semantic_refactored.py \
  --semantic semantic.avi \
  --photo photo.avi \
  --hybrid-out hybrid.avi \
  --mask-out mask.avi \
  --log-level INFO \
  --show-preview
```

### What the Merge Script Does

1. Builds a fast LUT from class BGR colors.
2. Creates a mask for selected "real-world classes":
   - `Unpaved Route`, `Paved Road`, `Person`, `Vehicle`, `Animal`
3. Expands route regions by including `Rocks` connected to `Unpaved Route`.
4. Creates `hybrid` by taking masked pixels from `photo`.
5. Also writes a binary `mask` video (0/255).

## Recommended Spatial Overlay Checks (ROI vs Background)

1. Full synchronization: `semantic` and `photo` must match FPS, length, and resolution.
2. ROI validation: ensure selected overlay classes match the intended ROI (route + key objects).
3. Background validation: ensure background classes (for example sky/vegetation) do not leak into overlay.
4. Boundary quality: inspect hard transitions and check for edge artifacts.
5. `mask.avi` quality: verify continuity and remove unexpected holes/noise.
6. A/B comparison: view `photo`, `semantic`, `hybrid`, and `mask` side by side.

## Common Issues

- `Cannot open semantic video` / `Cannot open photo video`: wrong input path or unsupported codec.
- Layer misalignment: videos are not synchronized.
- Empty/black mask: color mapping does not match segmentation output colors.
- Slow runtime: run without `--show-preview` and monitor FPS logs.

## Notes

- The workflow is based on continued training of NVIDIA weights, not training from scratch.
- `CAL` is used to better handle inter-class confusion with dynamic confusion-matrix weighting.
- For better spatial overlay quality, tighten ROI class selection and tune class inclusion rules.
