from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.train.losses import total_loss

_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD = torch.tensor([0.229, 0.224, 0.225])


def _normalize(img_t):
    mean = _MEAN.to(img_t.device).view(1, 3, 1, 1)
    std = _STD.to(img_t.device).view(1, 3, 1, 1)
    return (img_t - mean) / std


# Bypasses set_image() which disables gradients, invoking the internal backbone and decoder directly to keep LoRA gradients attached.
def forward_prompted(model, img_t_norm, box_batch, image_size):
    batch_size = img_t_norm.shape[0]
    img_t_norm = img_t_norm.type(torch.cuda.FloatTensor)

    backbone_out = model.forward_image(img_t_norm)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)

    feats = [
        f.permute(1, 2, 0).view(batch_size, -1, *sz)
        for f, sz in zip(vision_feats[::-1], feat_sizes[::-1])
    ]
    feats = feats[::-1]
    image_embed = feats[-1]
    high_res_feats = feats[:-1]

    box_t = box_batch.to(img_t_norm.device).view(batch_size, 1, 4)
    sparse_emb, dense_emb = model.sam_prompt_encoder(
        points=None, boxes=box_t, masks=None
    )

    dec_kwargs = dict(
        image_embeddings=image_embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
    )
    if high_res_feats:
        dec_kwargs["high_res_features"] = high_res_feats

    low_res_masks = model.sam_mask_decoder(**dec_kwargs)[0]

    logits = F.interpolate(
        low_res_masks, (image_size, image_size), mode="bilinear", align_corners=False
    ).squeeze(1)

    return logits


# Filters and saves only trainable adapter parameters (.A, .B, sd_adapter) to minimize checkpoint size.
def lora_state_dict(model):
    filtered_sd = {}
    for k, v in model.image_encoder.state_dict().items():
        if any(substring in k for substring in [".A", ".B", "lora_A", "lora_B", "sd_adapter"]):
            filtered_sd[k] = v
    return filtered_sd


def run_training(cfg, model, dataset, organ_id, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["micro_batch"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(cfg["train"]["lr"]))
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    opt.zero_grad(set_to_none=True)

    epoch_losses = []
    accum_steps = cfg["train"]["accum_steps"]
    epochs = cfg["train"]["epochs"]
    image_size = cfg["model"]["image_size"]

    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0.0
        num_batches = 0

        for step, (img, gt, box) in enumerate(loader):
            img_t = _normalize(img.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor))
            gt = gt.type(torch.cuda.FloatTensor)
            box = box.type(torch.cuda.FloatTensor)

            with torch.amp.autocast("cuda", enabled=False):
                logits = forward_prompted(model, img_t, box, image_size)
                loss_scaled = total_loss(logits, gt) / accum_steps

            scaler.scale(loss_scaled).backward()

            total_epoch_loss += loss_scaled.item() * accum_steps
            num_batches += 1

            # Gradient accumulation steps update optimizer periodically
            if (step + 1) % accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        if (len(loader) % accum_steps) != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        mean_loss = total_epoch_loss / max(1, num_batches)
        epoch_losses.append(mean_loss)
        print(f"Organ {organ_id} | Epoch {epoch + 1}/{epochs} | Loss: {mean_loss:.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), save_path)

    return epoch_losses