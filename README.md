# Volumetric Medical Image Segmentation via SAM 2

This project adapts Meta's **Segment Anything Model 2 (SAM 2)** to segment 3D organs from abdominal CT scans in the **BTCV** dataset. By treating 3D CT slices as sequential video frames, SAM 2 tracks organs across slices starting from a single 2D prompt slice.

---

## 1. System Overview

```
[Raw 3D CT Scan (.nii)]
       │
       ▼
[Preprocessing: Isotropic Resampling & HU Windowing]
       │
       ▼
[Find Best Start Slice (Largest Organ Area)]
       │
       ▼
[Initialize Memory Bank with Bounding Box Prompt]
       │
       ├───────────────────────────────┐
       ▼                               ▼
[Forward Tracking: slice z -> N]      [Backward Tracking: slice z -> 0]
       │                               │
       └───────────────┬───────────────┘
                       ▼
    [Early Halting: Area < min_area_pixels]
                       │
                       ▼
    [Final 3D Volumetric Segmentation Mask]
```

### Preprocessing
* **Isotropic Resampling**: Resamples CT scans to uniform $1.5\text{ mm}$ spacing so organ scale is consistent across patients.
* **HU Windowing**: Clips raw CT values to $[-150, +250\text{ HU}]$ to isolate soft-tissue organs.
* **Frame Conversion**: Converts slices into 3-channel pseudo-RGB images for the SAM 2 ViT encoder.

---

## 2. Technical Components

### Space-Depth (SD-Trans) Adapters
* Standard 2D ViTs process each slice independently without depth awareness.
* SD-Trans adapters transpose tokens to allow feature communication across the depth ($z$) axis.
* Uses a bottleneck layer with a learnable scaling factor ($\gamma$) initialized at zero.

### Parameter-Efficient Fine-Tuning (LoRA)
* Freezes the SAM 2 backbone and injects trainable low-rank matrices into attention layers.
* **PEFT**: Fine-tunes combined Query, Key, and Value ($QKV$) projections.
* **Custom Q/V**: Fine-tunes only Query ($Q$) and Value ($V$) projections.

### Hybrid Loss Function
* Combines **Volumetric Dice Loss** (measures 3D region overlap and handles organ-background size imbalance) and **Binary Cross-Entropy Loss** (refines pixel-level boundaries).

### Dual-Memory Bank & Early Halting
* **Short-Term Memory**: Remembers recent $k$ slices for smooth slice-to-slice transitions.
* **Long-Term Memory**: Anchors to the starting prompt slice to prevent shape drift.
* **Early Halting**: Stops tracking if organ slice area drops below `min_area_pixels` (5 px) to avoid false positive masks on non-organ slices.

---

## 3. Experimental Results

Evaluated on Organ 6 (Liver) across 6 validation cases:

| Metric | Zero-shot | LoRA | Target |
| :--- | :--- | :--- | :--- |
| **DSC** (Dice Similarity) | $0.843 \pm 0.066$ | **$0.918 \pm 0.040$** | Higher is better |
| **HD95** (Hausdorff Distance, mm) | $37.0 \pm 28.6$ | **$26.6 \pm 35.6$** | Lower is better |
| **VPE** (Volume Error, $\text{cm}^3$) | $616.53 \pm 296.19$ | **$269.88 \pm 207.33$** | Lower is better |

LoRA + SD Adapters boosted Dice overlap (+7.5%), reduced boundary error (-10.4 mm), and cut volume error by over 56%.

---

## 4. Execution Commands

### 1. Setup
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Preprocess
```bash
python scripts/01_preprocess.py --case img0001 --organ 6
```

### 3. Fine-Tune
```bash
python scripts/03_train_lora.py --organ 6 --sd-adapter
```

### 4. Inference
```bash
python scripts/02_run_inference.py --case img0001 --organ 6 --lora checkpoints/lora_organ6.pt
```

### 5. Evaluation
```bash
python scripts/04_evaluate.py --organ 6 --lora checkpoints/lora_organ6.pt
```
