import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
external_sam2 = project_root / "external" / "sam2"
if str(external_sam2) not in sys.path:
    sys.path.append(str(external_sam2))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_predictor(model_cfg, ckpt_path):
    return build_sam2_video_predictor(
        config_file=model_cfg,
        ckpt_path=ckpt_path,
        device="cpu",
    )


def build_image_predictor(model_cfg, ckpt_path):
    model = build_sam2(
        config_file=model_cfg,
        ckpt_path=ckpt_path,
        device="cpu",
    )
    return SAM2ImagePredictor(model)


# Offloading video and memory states to CPU prevents OOM issues on low VRAM hardware.
def init_state(predictor, frames_dir):
    return predictor.init_state(
        video_path=str(frames_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )