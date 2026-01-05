# train_cifar10_cond_jacobian_diff_prompts.py
import os
import math
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
    out_dir: str = "./results/condition_Jacobian_dynamics_during_training_diff_prompts_test/"

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
    wandb_project: str = "condition_Jacobian_dynamics_during_training_diff_prompts_test"
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
    max_steps: int = 200_000
    grad_clip: float = 1.0

    # memorization injection
    add_mem_sample: bool = True
    mem_prompt: str = "Memorized sample made by changsu shin, the student of university student"
    mem_oversample_factor: int = 1  # IMPORTANT knob: make memorization emerge clearly
    mem_loss_weight: float = 1.0

    # evaluation schedule
    num_milestones: int = 1001
    eval_num_gen: int = 1000          # #generated samples per prompt for f_mem
    eval_nn_chunk: int = 256          # batch size for NN distance computation
    fmem_k: float = 1.0 / 3.0

    # Jacobian norm measurement (initial step)
    j_num_projections: int = 4        # Hutchinson probes
    j_num_seeds: int = 64             # #x_T draws (Ongoing_paper uses many seeds)
    j_batch: int = 4                  # batch for Jacobian evaluation

    # loss evaluation at fixed t (paper uses t=0.01)
    loss_t_cont: float = 0.01
    loss_eval_batches: int = 50       # approx eval, not full pass

    # tracked prompts
    track_class: str = "cat"
    target_classes: Optional[List[int]] = field(default_factory=lambda: [1, 3, 5])  # None means all classes



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


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


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
                
                print(f"Loading prompts from {csv_path}...")
                df = pd.read_csv(csv_path)
                
                # Verify size matches
                if len(df) != len(self.base):
                    print(f"Warning: CSV size {len(df)} does not match dataset size {len(self.base)}. Falling back to default prompts.")
                    all_prompts = [f"A photo of {CIFAR10_CLASSES[y]}" for _, y in self.base]
                else:
                    all_prompts = df["simple_prompt"].tolist()
            except Exception as e:
                print(f"Error loading prompt CSV: {e}. Falling back to default prompts.")
                all_prompts = [f"A photo of {CIFAR10_CLASSES[y]}" for _, y in self.base]
        else:
            # For test set, use default prompts (CSV only has 50k usually)
            all_prompts = [f"A photo of {CIFAR10_CLASSES[y]}" for _, y in self.base]

        # Filter by target_classes if specified
        if cfg.target_classes is not None:
            # Find indices to keep
            keep_indices = [i for i, t in enumerate(self.base.targets) if t in cfg.target_classes]
            print(f"Filtering dataset: keeping {len(keep_indices)} samples for classes {cfg.target_classes}")
            
            # Filter base dataset (in-place modification of numpy array and list)
            self.base.data = self.base.data[keep_indices]
            self.base.targets = [self.base.targets[i] for i in keep_indices]
            
            # Filter prompts
            self.prompts = [all_prompts[i] for i in keep_indices]
        else:
            self.prompts = all_prompts

        # Inject one weird sample (train only)
        self.has_mem = (train and cfg.add_mem_sample)
        if self.has_mem:
            # example: pure noise image (uint8 0..255) -> PIL-like array
            noise_img = (np.random.rand(cfg.image_size, cfg.image_size, 3) * 255).astype(np.uint8)
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
    """
    Classifier-free training: with prob drop_prob per sample, replace cond by uncond.
    cond_embeds/uncond_embeds: [B, seq, hidden]
    """
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
    """
    Returns: [B] Frobenius norm estimates for each sample in the batch.
    Note: This computes a single scalar per sample by summing over all embed dims.
    """
    # Ensure we only take grad w.r.t prompt_embeds
    x_t = x_t.detach()
    prompt_embeds = prompt_embeds.detach().requires_grad_(True)

    with torch.enable_grad():
        noise_pred = unet(x_t, t_idx, encoder_hidden_states=prompt_embeds).sample  # [B,C,H,W]

        # We'll estimate ||J||_F^2 per-sample.
        # For per-sample Hutchinson, we need separate v per sample and separate scalar contractions.
        # We'll loop projections; inside, do sum over each sample's contraction and backprop once via autograd.grad on sum.
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


@torch.no_grad()
def estimate_initial_step_jacobian(
    unet: UNet2DConditionModel,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    cfg: Config,
    device: torch.device,
    prompt: str,
) -> Dict[str, float]:
    """
    Compute initial-step ||Jc||_F stats for a single prompt:
    average over j_num_seeds draws of x_T, using Hutchinson probes.
    """
    unet.eval()
    text_encoder.eval()

    # Prepare prompt embedding (will be turned into leaf per batch inside hutchinson call)
    pe = encode_prompt(tokenizer, text_encoder, [prompt], device, cfg.max_length)  # [1, seq, hid]

    t_init = cfg.num_train_timesteps - 1  # initial diffusion step
    vals = []

    remaining = cfg.j_num_seeds
    while remaining > 0:
        b = min(cfg.j_batch, remaining)
        remaining -= b

        x_T = torch.randn(b, 3, cfg.image_size, cfg.image_size, device=device)
        t_idx = torch.full((b,), t_init, device=device, dtype=torch.long)

        pe_b = pe.repeat(b, 1, 1)

        # Need grad; call non-no_grad function
        v = hutchinson_jacobian_frob_norm(
            unet=unet,
            x_t=x_T,
            t_idx=t_idx,
            prompt_embeds=pe_b,
            num_projections=cfg.j_num_projections,
        )
        vals.append(v.detach().cpu())

    vals = torch.cat(vals, dim=0).numpy()
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        "p50": float(np.quantile(vals, 0.50)),
        "p90": float(np.quantile(vals, 0.90)),
    }


# ----------------------------
# f_mem computation (Eq. 6 style)
# ----------------------------
def nearest_two_distances_l2(
    gen_flat: torch.Tensor,      # [B, D]
    train_flat: torch.Tensor,    # [N, D]
    train_norms: torch.Tensor,   # [N]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute nearest and 2nd-nearest L2 distances from each gen sample to train set.
    Efficient using ||x-a||^2 = ||x||^2 + ||a||^2 - 2 x·a
    """
    # gen_norms: [B,1]
    gen_norms = (gen_flat ** 2).sum(dim=1, keepdim=True)  # [B,1]
    # dot: [B,N]
    dot = gen_flat @ train_flat.t()
    # dist2: [B,N]
    dist2 = gen_norms + train_norms.view(1, -1) - 2.0 * dot
    dist2 = torch.clamp(dist2, min=0.0)

    # get two smallest
    d2, _ = torch.topk(dist2, k=2, largest=False, dim=1)  # [B,2]
    d1 = torch.sqrt(d2[:, 0])
    d2 = torch.sqrt(d2[:, 1])
    return d1, d2


@torch.no_grad()
def compute_fmem_percent(
    generated: torch.Tensor,     # [Ngen, 3, H, W] in [-1,1]
    train_images: torch.Tensor,  # [Ntrain, 3, H, W] in [-1,1]
    cfg: Config,
    device: torch.device,
) -> float:
    """
    f_mem based on nearest/2nd-nearest ratio < k (Eq. 6).
    """
    # Flatten in pixel space
    gen = generated.to(device, dtype=torch.float32).flatten(1)   # [Ngen,D]
    trn = train_images.to(device, dtype=torch.float32).flatten(1) # [Ntrain,D]
    trn_norms = (trn ** 2).sum(dim=1)  # [Ntrain]

    k = cfg.fmem_k
    flags = []

    N = gen.shape[0]
    for i in range(0, N, cfg.eval_nn_chunk):
        gb = gen[i:i+cfg.eval_nn_chunk]
        d1, d2 = nearest_two_distances_l2(gb, trn, trn_norms)
        ratio = d1 / torch.clamp(d2, min=1e-12)
        flags.append((ratio < k).float().cpu())

    flags = torch.cat(flags, dim=0).numpy()
    return float(100.0 * flags.mean())


# ----------------------------
# Sampling (conditional + CFG)
# ----------------------------
@torch.no_grad()
def sample_images_cfg(
    unet: UNet2DConditionModel,
    ddim: DDIMScheduler,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    cfg: Config,
    device: torch.device,
    prompt: str,
    num_images: int,
    guidance_scale: float = 5.0,
) -> torch.Tensor:
    unet.eval()
    text_encoder.eval()

    cond = encode_prompt(tokenizer, text_encoder, [prompt] * num_images, device, cfg.max_length)
    uncond = encode_prompt(tokenizer, text_encoder, [""] * num_images, device, cfg.max_length)

    x = torch.randn(num_images, 3, cfg.image_size, cfg.image_size, device=device)
    ddim.set_timesteps(cfg.ddim_steps_eval, device=device)

    for t in ddim.timesteps:
        t_batch = torch.full((num_images,), int(t), device=device, dtype=torch.long)

        eps_u = unet(x, t_batch, encoder_hidden_states=uncond).sample
        eps_c = unet(x, t_batch, encoder_hidden_states=cond).sample
        eps = eps_u + guidance_scale * (eps_c - eps_u)

        x = ddim.step(eps, t, x).prev_sample

    # x in [-1,1] approximately
    return x.detach().cpu()


# ----------------------------
# Loss eval at fixed timestep
# ----------------------------
@torch.no_grad()
def eval_noise_pred_loss(
    unet: UNet2DConditionModel,
    ddpm: DDPMScheduler,
    loader: DataLoader,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    cfg: Config,
    device: torch.device,
    num_batches: int,
    t_eval_idx: int,
) -> float:
    unet.eval()
    text_encoder.eval()

    losses = []
    it = iter(loader)
    for _ in range(num_batches):
        batch = next(it, None)
        if batch is None:
            it = iter(loader)
            batch = next(it)

        x0 = batch["pixel_values"].to(device)
        prompts = batch["prompt"]

        cond = encode_prompt(tokenizer, text_encoder, list(prompts), device, cfg.max_length)
        uncond = encode_prompt(tokenizer, text_encoder, [""] * len(prompts), device, cfg.max_length)
        cond = maybe_drop_condition(cond, uncond, drop_prob=0.0)  # eval: no drop

        noise = torch.randn_like(x0)
        t = torch.full((x0.shape[0],), t_eval_idx, device=device, dtype=torch.long)
        xt = ddpm.add_noise(x0, noise, t)

        eps = unet(xt, t, encoder_hidden_states=cond).sample
        loss = F.mse_loss(eps, noise)
        losses.append(loss.item())

    return float(np.mean(losses))


# ----------------------------
# Main train/eval loop
# ----------------------------
def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    # DDP Init / Single GPU Fallback
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed + rank) # Seed with rank for DDP

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
        # 선택: step을 x축으로 고정
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

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=True)

    # text encoder (frozen)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_name)
    text_encoder = CLIPTextModel.from_pretrained(cfg.clip_name).to(device)
    text_encoder.requires_grad_(False)

    cross_attn_dim = text_encoder.config.hidden_size

    # UNet conditional
    unet = UNet2DConditionModel(
        sample_size=cfg.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=cross_attn_dim,
    ).to(device)

    ddpm = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps,
                        beta_schedule=cfg.beta_schedule,  # "linear"
                        beta_start=cfg.beta_start,
                        beta_end=cfg.beta_end,
                        )
    ddim = DDIMScheduler(num_train_timesteps=cfg.num_train_timesteps,
                        beta_schedule=cfg.beta_schedule,  # "linear"
                        beta_start=cfg.beta_start,
                        beta_end=cfg.beta_end,
                        )

    opt = torch.optim.AdamW(unet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # for f_mem: cache a tensor of unique train images (including injected sample once)
    # NOTE: If you oversampled mem sample, it appears many times in train_ds; for NN set, keep unique once.
    print("Caching train images for f_mem NN computation...")
    unique_train_imgs = []
    seen_mem = False
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
    step = 0
    unet.train()
    text_encoder.eval()

    train_iter = iter(train_loader)

    while step < cfg.max_steps:
        batch = next(train_iter, None)
        if batch is None:
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
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                print(f"Saving checkpoint at step {step}...")
                save_path = os.path.join(cfg.out_dir, f"ckpt_step_{step}.pt")
                # DDP wrapper unwrap
                model_to_save = unet.module if hasattr(unet, "module") else unet
                torch.save(model_to_save.state_dict(), save_path)
            if dist.is_initialized():
                dist.barrier()
            
    if use_wandb:
        wandb.finish()
        
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
