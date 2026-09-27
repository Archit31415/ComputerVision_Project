import torch
import torch.nn as nn

# Space-Depth Adapter mixes tokens across the z-axis depth dimension without 3D convolutions.
class SpaceDepthAdapter(nn.Module):
    def __init__(self, embed_dim=768, bottleneck_dim=64):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.depth_layer = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        # Learnable scale initialized to 0 keeps pre-trained ViT weights undisturbed initially.
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        is_4d = x.ndim == 4
        x_in = x.permute(0, 2, 3, 1) if is_4d else x

        down = self.act(self.down_proj(x_in))
        depth = self.act(self.depth_layer(down))
        out = self.up_proj(depth)

        return residual + self.scale * (out.permute(0, 3, 1, 2) if is_4d else out)


def inject_sd_adapters(model, embed_dim=768, bottleneck_dim=64):
    for name, module in list(model.named_modules()):
        if hasattr(module, "attn"):
            adapter = SpaceDepthAdapter(embed_dim=embed_dim, bottleneck_dim=bottleneck_dim)
            setattr(module, "sd_adapter", adapter)
    return model
