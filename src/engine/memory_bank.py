import numpy as np


# ShortLongMemoryBank anchors long-term memory to start_z while maintaining a k-slice short-term window.
class ShortLongMemoryBank:
    def __init__(self, k_slices=3):
        self.k_slices = k_slices
        self.prompt_z = None
        self.prompt_bbox = None
        self.prompt_mask = None
        self.short_term_masks = {}

    def register_prompt(self, start_z, bbox, mask=None):
        self.prompt_z = start_z
        self.prompt_bbox = bbox
        self.prompt_mask = mask
        if mask is not None:
            self.short_term_masks[start_z] = mask.astype(bool)

    def add_slice_mask(self, slice_z, mask_bool):
        self.short_term_masks[slice_z] = mask_bool

        if len(self.short_term_masks) > self.k_slices + 1:
            sorted_keys = sorted(
                self.short_term_masks.keys(),
                key=lambda z: abs(z - slice_z),
            )
            keys_to_keep = set(sorted_keys[: self.k_slices])
            if self.prompt_z is not None:
                keys_to_keep.add(self.prompt_z)
            self.short_term_masks = {
                k: v for k, v in self.short_term_masks.items() if k in keys_to_keep
            }

    def reset_short_term(self):
        long_term_mask = (
            self.short_term_masks.get(self.prompt_z) if self.prompt_z is not None else None
        )
        self.short_term_masks.clear()
        if self.prompt_z is not None and long_term_mask is not None:
            self.short_term_masks[self.prompt_z] = long_term_mask
