import torch
import torch.nn as nn


# PEFT LoRA injects low-rank trainable matrices into fused QKV attention layers.
def add_lora_peft(image_encoder, r=8, alpha=16):
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["qkv"],
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(image_encoder, peft_config)


# Custom LoRALinear wraps fused QKV linear layers and updates Q and V weights while zeroing K.
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, qv_only=True):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.qv_only = qv_only
        self.r = r
        self.scale = alpha / r
        self.dim = base.in_features

        # A is Kaiming-uniform initialized and B is zero-initialized so LoRA acts as identity at step 0.
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        delta = (x @ self.A.t()) @ self.B.t() * self.scale
        if self.qv_only:
            delta[..., self.dim : 2 * self.dim] = 0.0
        return self.base(x) + delta


def inject_lora_qv(model, r=8, alpha=16):
    for name, module in list(model.named_modules()):
        if hasattr(module, "attn") and hasattr(module.attn, "qkv"):
            if isinstance(module.attn.qkv, nn.Linear):
                module.attn.qkv = LoRALinear(module.attn.qkv, r=r, alpha=alpha, qv_only=True)
        elif hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, r=r, alpha=alpha, qv_only=True)
    return model


def trainable_report(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (trainable / total * 100) if total > 0 else 0.0
    print(f"Trainable: {trainable:,} / {total:,} = {pct:.3f}%")