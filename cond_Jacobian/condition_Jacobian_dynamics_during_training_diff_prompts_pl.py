# condition_Jacobian_dynamics_during_training_diff_prompts_pl.py
import os
import math
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler

import wandb


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    seed: int = 42
    out_dir: str = "./results/condition_Jacobian_dynamics_during_training_diff_prompts_pl/"

    # data
    image_size: int = 32
    batch_size: int = 128
    num_workers: int = 4

    # diffusion
    num_train_timesteps: int = 1000
    ddim_steps_eval: int = 200  # for faster sampling at milestones

    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # wandb
    wandb_project: str = "condition_Jacobian_dynamics_during_training_diff_prompts_pl"
    wandb_entity: Optional[str] = None  # 팀/계정(없으면 None)
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"          # "online" | "offline" | "disabled"
    log_every: int = 50     

    # model/text
    clip_name: str = "openai/clip-vit-base-patch32"
    max_length: int = 77  # CLIP typical
    uncond_drop_prob: float = 0.1  # classifier-free training

    # optimization
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int = 25_000
    grad_clip: float = 1.0

    # memorization injection
    add_mem_sample: bool = True
    mem_prompt: str = "Memorized sample made by changsu shin, the student of university student"
    mem_oversample_factor: int = 1              # 기존 데이터는 50으로 실험했음
    mem_loss_weight: float = 1.0

    # evaluation schedule
    num_milestones: int = 100

    # tracked prompts
    track_class: str = "cat"

    # resume
    resume_ckpt_path: Optional[str] = None


CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

SYNONYMS = {
    "airplane": "aircraft",
    "automobile": "car",
    "bird": "avian",
    "cat": "feline",
    "deer": "stag",
    "dog": "canine",
    "frog": "amphibian",
    "horse": "stallion",
    "ship": "vessel",
    "truck": "lorry",
}


def set_seed(seed: int, rank: int = 0):
    # Set seed differently per rank to avoid identical batches if not using proper sampler,
    # but DistributedSampler handles indices.
    # However, for noise generation, we might want different seeds or same seeds depending on detailed requirement.
    # Usually, we set same seed for reproducibility of model init, but different seeds for data loaders are handled by workers.
    # Here just set base seed + rank to ensure noise is different across GPUs if needed, or same.
    # For DDP, it's common to set same seed for model init, then different for noise.
    # But usually DistributedSampler handles data split.
    # Let's keep it simple: seed + rank
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)


# ----------------------------
# Dataset with an injected "memorized" sample
# ----------------------------
class CIFAR10WithMem(Dataset):
    def __init__(self, root: str, train: bool, transform, cfg: Config):
        self.base = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.cfg = cfg

        # Build prompt list for each base sample
        if train:
            # Load prompts from CSV for training set
            try:
                # Assuming CSV is in the same directory as the script or current working directory
                # Check for csv in typical locations
                csv_path = "cifar10_prompts.csv"
                if not os.path.exists(csv_path):
                    # Fallback to absolute path assumption if running from project root
                     csv_path = os.path.join(os.path.dirname(__file__), "cifar10_prompts.csv")
                
                # Check if file exists, if not printing warning (rank 0 handles prints mostly but harmless here)
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if len(df) != len(self.base):
                        self.prompts = [f"A photo of {CIFAR10_CLASSES[y]}" for _, y in self.base]
                    else:
                        self.prompts = df["simple_prompt"].tolist()
                else:
                    self.prompts = [f"A photo of {CIFAR10_CLASSES[y]}" for _, y in self.base]
            except Exception:
                self.prompts = [f"A photo of {CIFAR10_CLASSES[y]}" for _, y in self.base]
        else:
            self.prompts = [f"A photo of {CIFAR10_CLASSES[y]}" for _, y in self.base]

        # Inject one weird sample (train only)
        self.has_mem = (train and cfg.add_mem_sample)
        if self.has_mem:
            # example: pure noise image (uint8 0..255) -> PIL-like array
            # Fixed pattern for memorization (should be same across all runs/ranks)
            rng_state = np.random.get_state()
            np.random.seed(cfg.seed) # Fix seed for this generation to be checking consistency
            noise_img = (np.random.rand(cfg.image_size, cfg.image_size, 3) * 255).astype(np.uint8)
            np.random.set_state(rng_state)
            
            self.mem_image = noise_img
            self.mem_prompt = cfg.mem_prompt
            
            # For oversampling, append repeated indices referencing the mem sample
            self.mem_repeats = cfg.mem_oversample_factor
        else:
            self.mem_image = None
            self.mem_prompt = None
            self.mem_repeats = 0

        self.length = len(self.base) + self.mem_repeats

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if idx < len(self.base):
            img, y = self.base[idx]
            img = self.transform(img)
            prompt = self.prompts[idx]
            is_mem = False
        else:
            # memorized injected sample
            img = self.mem_image
            img = transforms.ToPILImage()(img)
            img = self.transform(img)
            prompt = self.mem_prompt
            is_mem = True

        return {
            "pixel_values": img,     # [-1, 1]
            "prompt": prompt,
            "is_mem": is_mem,
        }


# ----------------------------
# Text encoder helpers
# ----------------------------
@torch.no_grad()
def encode_prompt(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, prompts: List[str], device: torch.device, max_length: int):
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attn_mask = tokens.attention_mask.to(device)
    out = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
    # last_hidden_state: [B, seq_len, hidden]
    return out.last_hidden_state


def maybe_drop_condition(cond_embeds: torch.Tensor, uncond_embeds: torch.Tensor, drop_prob: float):
    if drop_prob <= 0:
        return cond_embeds
    B = cond_embeds.shape[0]
    mask = (torch.rand(B, device=cond_embeds.device) < drop_prob).view(B, 1, 1)
    return torch.where(mask, uncond_embeds, cond_embeds)


# ----------------------------
# Jacobian norm (Hutchinson) at a fixed (x_t, t)
# ||J||_F^2 = E_v ||J^T v||^2,  with v ~ N(0, I_d)
# ----------------------------
def hutchinson_jacobian_frob_norm(
    unet: UNet2DConditionModel,
    x_t: torch.Tensor,             # [B, C, H, W]
    t_idx: torch.Tensor,           # [B] (int timesteps)
    prompt_embeds: torch.Tensor,   # [B, seq, hidden], leaf requiring grad
    num_projections: int = 4,
) -> torch.Tensor:
    # Ensure we only take grad w.r.t prompt_embeds
    x_t = x_t.detach()
    prompt_embeds = prompt_embeds.detach().requires_grad_(True)

    with torch.enable_grad():
        noise_pred = unet(x_t, t_idx, encoder_hidden_states=prompt_embeds).sample  # [B,C,H,W]

        sq = torch.zeros(prompt_embeds.shape[0], device=x_t.device)

        for k in range(num_projections):
            v = torch.randn_like(noise_pred)
            # per-sample contraction: <eps_i, v_i> = sum_{chw} eps_i * v_i
            s = (noise_pred * v).flatten(1).sum(dim=1)  # [B]
            total = s.sum()  # single scalar

            grads = torch.autograd.grad(
                total,
                prompt_embeds,
                retain_graph=(k < num_projections - 1),
                create_graph=False,
            )[0]  # [B, seq, hidden]

            # ||J^T v||^2 per sample
            sq = sq + grads.flatten(1).pow(2).sum(dim=1).detach()

        frob_sq = sq / float(max(1, num_projections))
        return torch.sqrt(torch.clamp(frob_sq, min=0.0))

# ----------------------------
# Main train/eval loop
# ----------------------------
def main():
    # Distributed Setup
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    cfg = Config()
    
    # Only Rank 0 creates directories
    if rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)

    # Sync
    dist.barrier()

    set_seed(cfg.seed, rank)

    # WandB only on Rank 0
    use_wandb = (cfg.wandb_mode != "disabled") and (rank == 0)
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            config=asdict(cfg),
            dir=cfg.out_dir,
            mode=cfg.wandb_mode,
        )
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")

    # transforms: CIFAR10 -> [-1,1]
    tfm = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_ds = CIFAR10WithMem("/home/dataset/", train=True, transform=tfm, cfg=cfg)
    test_ds  = CIFAR10WithMem("/home/dataset/", train=False, transform=tfm, cfg=cfg)

    def collate_fn(items):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in items], dim=0),
            "prompt": [x["prompt"] for x in items],
            "is_mem": torch.tensor([x["is_mem"] for x in items], dtype=torch.bool),
        }

    # DistributedSampler
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    # Test sampler likely not needed if only Rank 0 evals, but for safety in finding data
    test_sampler = DistributedSampler(test_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=False, # Sampler handles shuffle
        sampler=train_sampler,
        num_workers=cfg.num_workers, 
        collate_fn=collate_fn, 
        drop_last=True
    )
    test_loader  = DataLoader(
        test_ds, 
        batch_size=cfg.batch_size, 
        shuffle=False,
        sampler=test_sampler,
        num_workers=cfg.num_workers, 
        collate_fn=collate_fn, 
        drop_last=True
    )

    # text encoder (frozen) - loaded on all ranks
    tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_name)
    text_encoder = CLIPTextModel.from_pretrained(cfg.clip_name).to(device)
    text_encoder.requires_grad_(False)

    cross_attn_dim = text_encoder.config.hidden_size

    # UNet conditional
    unet = UNet2DConditionModel(
        sample_size=cfg.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=cross_attn_dim,
    ).to(device)

    # Resume from checkpoint if specified
    start_step = 0
    if cfg.resume_ckpt_path and os.path.exists(cfg.resume_ckpt_path):
        if rank == 0:
            print(f"Resuming from checkpoint: {cfg.resume_ckpt_path}")
        
        # Load checkpoint
        ckpt = torch.load(cfg.resume_ckpt_path, map_location=device)
        
        # Check if it's a raw state dict or a dictionary containing metadata
        # The current save logic saves raw state_dict, but we handle robustly.
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            unet.load_state_dict(ckpt["model_state_dict"])
            if "step" in ckpt:
                start_step = ckpt["step"]
        else:
            # Assume raw state_dict
            unet.load_state_dict(ckpt)
            
        # Try to infer step from filename if not explicit
        if start_step == 0:
            import re
            # Extract number after 'step_' usually found in 'ckpt_step_1234.pt'
            match = re.search(r"step_(\d+)", os.path.basename(cfg.resume_ckpt_path))
            if match:
                start_step = int(match.group(1))
        
        if rank == 0:
            print(f"Model loaded. Resuming training from step {start_step}...")

    # Wrap DDP
    unet = DDP(unet, device_ids=[local_rank])

    ddpm = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps,
                        beta_schedule=cfg.beta_schedule,
                        beta_start=cfg.beta_start,
                        beta_end=cfg.beta_end)
    ddim = DDIMScheduler(num_train_timesteps=cfg.num_train_timesteps,
                        beta_schedule=cfg.beta_schedule,
                        beta_start=cfg.beta_start,
                        beta_end=cfg.beta_end)

    opt = torch.optim.AdamW(unet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # for f_mem: cache a tensor of unique train images (including injected sample once)
    # Only needed on Rank 0 for evaluation, but calculating on all is fine (cheap overhead) or just Rank 0.
    # To reduce memory on others, only Rank 0.
    train_images_tensor = None
    if rank == 0:
        print("Caching train images for f_mem NN computation (Rank 0)...")
        unique_train_imgs = []
        seen_mem = False
        # Access dataset directly, bypassing loader
        for i in range(len(train_ds)):
            item = train_ds[i]
            if item["prompt"] == cfg.mem_prompt:
                if seen_mem:
                    continue
                seen_mem = True
            unique_train_imgs.append(item["pixel_values"])
        train_images_tensor = torch.stack(unique_train_imgs, dim=0)  # [Ntrain,3,H,W]

    # prompts to track
    cls = cfg.track_class
    p_train = f"A photo of {cls}"
    p_syn   = f"{SYNONYMS.get(cls, cls)} picture"
    p_mem   = cfg.mem_prompt
    tracked_prompts = [p_train, p_syn, p_mem]

    # eval timesteps
    t_eval_idx = max(1, int(cfg.loss_t_cont * cfg.num_train_timesteps))

    milestones = np.linspace(0, cfg.max_steps, cfg.num_milestones, dtype=int).tolist()
    milestones = sorted(set([m for m in milestones if m > 0]))

    logs = []
    step = start_step
    unet.train()
    text_encoder.eval()

    epoch = 0
    train_sampler.set_epoch(epoch)
    train_iter = iter(train_loader)

    if rank == 0:
        print('학습 시작 (DDP)')

    while step < cfg.max_steps:
        batch = next(train_iter, None)
        if batch is None:
            epoch += 1
            train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x0 = batch["pixel_values"].to(device)
        prompts = batch["prompt"]
        is_mem = batch["is_mem"].to(device)

        with torch.no_grad():
            cond = encode_prompt(tokenizer, text_encoder, list(prompts), device, cfg.max_length)
            uncond = encode_prompt(tokenizer, text_encoder, [""] * len(prompts), device, cfg.max_length)
            cond = maybe_drop_condition(cond, uncond, drop_prob=cfg.uncond_drop_prob)

        noise = torch.randn_like(x0)
        t = torch.randint(0, cfg.num_train_timesteps, (x0.shape[0],), device=device).long()
        xt = ddpm.add_noise(x0, noise, t)

        # DDP forward
        eps = unet(xt, t, encoder_hidden_states=cond).sample

        # optionally weight memorized injected samples
        per_sample = (eps - noise).flatten(1).pow(2).mean(dim=1)  # [B]
        weights = torch.ones_like(per_sample)
        weights = torch.where(is_mem, torch.tensor(cfg.mem_loss_weight, device=device), weights)
        loss = (weights * per_sample).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.grad_clip)
        opt.step()

        if use_wandb and (step % cfg.log_every == 0):
            wandb.log(
                {
                    "train/step": step,
                    "train/loss_batch": float(loss.item()),
                },
                step=step,
            )

        step += 1

        if step in milestones:
            dist.barrier()
            if rank == 0:
                print(f"Saving checkpoint at step {step}...")
                save_path = os.path.join(cfg.out_dir, f"ckpt_step_{step}.pt")
                # DDP wrapper unwrap
                model_to_save = unet.module if hasattr(unet, "module") else unet
                torch.save(model_to_save.state_dict(), save_path)
            dist.barrier()

    if use_wandb:
        wandb.finish()
        
    # Destroy process group
    dist.destroy_process_group()
    if rank == 0:
        print("Done.")


def plot_metrics(df: pd.DataFrame, cfg: Config, tracked_prompts: List[str]):
    # 1) loss curve (like Fig.2 middle)
    plt.figure()
    plt.plot(df["step"], df["train_loss_t"], label="train loss (fixed t)")
    plt.plot(df["step"], df["test_loss_t"], label="test loss (fixed t)")
    plt.xlabel("training step")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "loss_curve.png"))
    plt.close()

    # 2) f_mem curves
    plt.figure()
    for p in tracked_prompts:
        plt.plot(df["step"], df[f"fmem%::{p}"], label=f"f_mem%: {short(p)}")
    plt.xlabel("training step")
    plt.ylabel("f_mem (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "fmem_curve.png"))
    plt.close()

    # 3) initial-step Jc curves
    plt.figure()
    for p in tracked_prompts:
        plt.plot(df["step"], df[f"Jc_mean::{p}"], label=f"Jc mean: {short(p)}")
    plt.xlabel("training step")
    plt.ylabel("||Jc||_F (initial step)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "jc_curve.png"))
    plt.close()


def short(p: str, n: int = 24):
    return p if len(p) <= n else (p[:n] + "...")


if __name__ == "__main__":
    main()
