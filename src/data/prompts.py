import numpy as np


def bbox_from_mask(mask2d, pad=4):
    rows, cols = np.where(mask2d)
    if len(rows) == 0:
        return None

    h, w = mask2d.shape
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()

    x0 = max(0, x_min - pad)
    y0 = max(0, y_min - pad)
    x1 = min(w - 1, x_max + pad)
    y1 = min(h - 1, y_max + pad)

    return np.array([x0, y0, x1, y1], dtype=np.float32)


# Starting slice heuristic selects the axial slice with largest organ area to minimize prompt ambiguity.
def best_start_slice(label_vol, organ_id):
    binary_vol = label_vol == organ_id
    if not np.any(binary_vol):
        raise ValueError(f"Organ {organ_id} is not present anywhere in the label volume.")

    slice_counts = np.sum(binary_vol, axis=(0, 1))
    return int(np.argmax(slice_counts))
