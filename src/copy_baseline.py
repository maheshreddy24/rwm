import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel
from icecream import ic

from datasets.rvm_dataset_tf import RVMDataset

with open("/home/rvm/configs/train_ema.yaml", "r") as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_set = RVMDataset(config["dataset"]["train"], deterministic=False)
train_loader = DataLoader(
    train_set,
    batch_size=config["dataloader"]["batch_size"],
    num_workers=config["dataloader"]["num_workers"],
    pin_memory=config["dataloader"]["pin_memory"],
    collate_fn=RVMDataset.collate_fn,
    shuffle=True,
)

encoder_name = config["model"]["encoder_name"]  # facebook/dinov2-with-registers-small
encoder = AutoModel.from_pretrained(encoder_name).to(device).eval()
for p in encoder.parameters():
    p.requires_grad_(False)
num_register_tokens = getattr(encoder.config, "num_register_tokens", 0)

C = config["dataset"]["train"]["context_frames"]  # 5


def encode(frames: torch.Tensor) -> torch.Tensor:
    """(M, 3, H, W) -> (M, 1+P, D), registers dropped, CLS kept."""
    h = encoder.encoder(encoder.embeddings(frames)).last_hidden_state
    if num_register_tokens > 0:
        h = torch.cat([h[:, :1], h[:, 1 + num_register_tokens:]], dim=1)  # [CLS, reg...] -> [CLS]
    return encoder.layernorm(h)


total_loss, n_batches = 0.0, 0
with torch.no_grad():
    for step, batch in tqdm(enumerate(train_loader), total = len(train_loaderbas)):
        context = batch["context"].to(device, non_blocking=True)      # (B, roll_out, 3, H, W)
        B = context.shape[0]

        ctx_frames = context[:, :C]                                    # (B, C, 3, H, W)
        tgt_frames = context[:, C:]                                    # (B, S, 3, H, W)
        S = tgt_frames.shape[1]

        ctx_feat = encode(ctx_frames.reshape(B * C, *ctx_frames.shape[2:]))
        ctx_feat = ctx_feat.view(B, C, *ctx_feat.shape[1:])             # (B, C, n_tok, D)
        tgt_feat = encode(tgt_frames.reshape(B * S, *tgt_frames.shape[2:]))
        tgt_feat = tgt_feat.view(B, S, *tgt_feat.shape[1:])             # (B, S, n_tok, D)

        copy_pred = ctx_feat[:, -1:].expand(-1, S, -1, -1)              # last context frame, repeated for every target step

        loss = F.smooth_l1_loss(copy_pred[:, :, 1:, :], tgt_feat[:, :, 1:, :], beta=0.1)  # drop CLS
        ic(step, loss.item())

        total_loss += loss.item()
        n_batches += 1

print(f"copy baseline avg loss over {n_batches} batches: {total_loss / max(n_batches, 1):.4f}")
