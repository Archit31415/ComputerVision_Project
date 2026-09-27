#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Seed global random number generators prior to model construction for reproducibility
from src.utils.seed import set_seed
set_seed(42)

import yaml
from src.data.dataset import BTCVSliceDataset
from src.engine.predictor import build_predictor
from src.train.lora import add_lora_peft, inject_lora_qv, trainable_report
from src.train.adapters import inject_sd_adapters
from src.train.train import run_training


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune SAM2 on BTCV")
    parser.add_argument("--organ", type=int, required=True, help="BTCV organ label (6=liver, 1=spleen, 11=pancreas)")
    parser.add_argument("--qv-only", action="store_true", help="Use custom Q&V-only LoRA instead of peft")
    parser.add_argument("--sd-adapter", action="store_true", help="Inject Space-Depth (SD-Trans) Adapters into vision encoder")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])
    model = predictor

    # Optionally inject Space-Depth adapters to mix features across depth slices
    if args.sd_adapter:
        model.image_encoder = inject_sd_adapters(model.image_encoder)

    # Inject PEFT QKV LoRA or custom Q/V LoRA parameters
    if args.qv_only:
        inject_lora_qv(model.image_encoder, r=cfg["train"]["rank"], alpha=cfg["train"]["alpha"])
    else:
        model.image_encoder = add_lora_peft(
            model.image_encoder,
            r=cfg["train"]["rank"],
            alpha=cfg["train"]["alpha"],
        )

    trainable_report(model)

    val_set = set(cfg["split"]["val_cases"])
    img_dir = Path(cfg["paths"]["raw_images"])
    lbl_dir = Path(cfg["paths"]["raw_labels"])
    all_cases = [p.stem for p in sorted(img_dir.glob("*.nii")) if p.stem not in val_set]

    print(f"Training cases: {len(all_cases)}  |  organ={args.organ}")
    dataset = BTCVSliceDataset(
        all_cases, args.organ,
        str(img_dir), str(lbl_dir),
        image_size=cfg["model"]["image_size"],
    )
    print(f"Dataset slices: {len(dataset)}")

    if len(dataset) == 0:
        print("No slices found for this organ. Check that imagesTr/*.nii exist.")
        return

    save_path = Path(cfg["paths"]["checkpoints"]) / f"lora_organ{args.organ}.pt"
    losses = run_training(cfg, model, dataset, args.organ, str(save_path))

    print(f"\nTraining complete. Final epoch loss: {losses[-1]:.4f}")
    print(f"LoRA adapter saved -> {save_path}")


if __name__ == "__main__":
    main()
