from PIL import Image
import numpy as np
from src.engine.memory_bank import ShortLongMemoryBank


# Bidirectional tracking propagates from prompt start_z forward and backward with early halting.
def propagate_bidirectional(
    predictor, state, start_z, bbox, target_hw=None, min_area_pixels=5, k_slices=3
):
    predictor.add_new_points_or_box(
        state,
        frame_idx=start_z,
        obj_id=1,
        box=bbox,
    )

    memory_bank = ShortLongMemoryBank(k_slices=k_slices)
    memory_bank.register_prompt(start_z, bbox)

    slice_masks = {}
    num_frames = state["num_frames"] if "num_frames" in state else len(state.get("images", []))
    if num_frames == 0:
        num_frames = getattr(state, "num_frames", 0)

    def process_and_store_mask(frame_idx, mask_tensor):
        mask_bool = (mask_tensor > 0).cpu().numpy()

        if target_hw is not None and mask_bool.shape != target_hw:
            target_h, target_w = target_hw
            img = Image.fromarray(mask_bool)
            resized_img = img.resize((target_w, target_h), resample=Image.NEAREST)
            mask_bool = np.array(resized_img, dtype=bool)

        slice_masks[frame_idx] = mask_bool
        memory_bank.add_slice_mask(frame_idx, mask_bool)
        return mask_bool

    # Forward tracking pass (start_z -> top slice)
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(
        state, start_frame_idx=start_z, reverse=False
    ):
        mask_bool = process_and_store_mask(frame_idx, masks[0, 0])
        # Early halting stops tracking when organ area drops below threshold to prevent boundary drift.
        if frame_idx != start_z and mask_bool.sum() < min_area_pixels:
            print(
                f"  Forward propagation halted early at slice z={frame_idx} "
                f"(area {mask_bool.sum()} < threshold {min_area_pixels} px)"
            )
            break

    memory_bank.reset_short_term()

    # Backward tracking pass (start_z -> slice 0)
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(
        state, start_frame_idx=start_z, reverse=True
    ):
        if frame_idx not in slice_masks:
            mask_bool = process_and_store_mask(frame_idx, masks[0, 0])
            if frame_idx != start_z and mask_bool.sum() < min_area_pixels:
                print(
                    f"  Backward propagation halted early at slice z={frame_idx} "
                    f"(area {mask_bool.sum()} < threshold {min_area_pixels} px)"
                )
                break

    if target_hw is not None:
        H, W = target_hw
    elif len(slice_masks) > 0:
        sample_mask = next(iter(slice_masks.values()))
        H, W = sample_mask.shape
    else:
        H, W = 512, 512

    max_z = max(slice_masks.keys()) if len(slice_masks) > 0 else 0
    total_z = max(num_frames, max_z + 1)

    empty_mask = np.zeros((H, W), dtype=bool)
    sorted_frames = [slice_masks.get(z, empty_mask) for z in range(total_z)]
    return np.stack(sorted_frames, axis=2)