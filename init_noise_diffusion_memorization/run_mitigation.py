
import os
import torch
import pandas as pd
import argparse
from diffusers import DDIMScheduler
from local_sd_pipeline import LocalStableDiffusionPipeline
import numpy as np

# Setup Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Batch-wise Mitigation for Memorization")
    parser.add_argument("--gamma", type=float, default=0.01, help="Adjustment strength (step size)")
    parser.add_argument("--rho", type=float, default=0.05, help="Sharpness parameter")
    parser.add_argument("--M", type=int, default=5, help="Number of adjustments (M)")
    parser.add_argument("--tau_time", type=int, default=800, help="CFG application start timestep (e.g. 800 for early limit)")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output_dir", type=str, default="results/mitigation")
    parser.add_argument("--n_prompts", type=int, default=1, help="Number of prompts to test")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

class MitigationConfig:
    def __init__(self, gamma, rho, M, apply_cfg_step):
        # Parameters for Batch-wise mitigation
        self.batch_wise = True
        self.per_sample = False
        self.gamma = gamma
        self.rho = rho
        self.adj_iters = M
        self.apply_cfg_step = apply_cfg_step
        
        # Dummy params for per_sample (not used but required if accessed)
        self.lr = 0.0
        self.optim_iters = 0
        self.target_loss = 0.0

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Pipeline
    print(f"Loading model {args.model_id}...")
    pipeline = LocalStableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        requires_safety_checker=False
    ).to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    # Load Prompts
    try:
        df = pd.read_csv('prompts/memorized_laion_prompts.csv', sep=';')
        prompts = df['Caption'].tolist()[:args.n_prompts]
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return

    # Calculate apply_cfg_step from tau_time
    # We need to map tau (e.g. 800) to step index (e.g. 10)
    # The scheduler logic: timesteps go from ~1000 down to 0.
    # We want unconditional if t > tau.
    # Step indices i=0...T. 
    # Valid approximation: (1 - tau/1000) * total_steps
    # If tau=800, 1000, ratio=0.2. 0.2 * 50 = 10 steps.
    # So first 10 steps are > 800.
    apply_cfg_step = int((1.0 - args.tau_time / 1000.0) * args.steps)
    print(f"Configuration: Gamma={args.gamma}, Rho={args.rho}, M={args.M}, Tau(time)={args.tau_time} -> CFG Delayed Steps={apply_cfg_step}")

    config = MitigationConfig(args.gamma, args.rho, args.M, apply_cfg_step)
    
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # Run Mitigation Sampling
        # The pipeline implementation expects method='adj_init_noise' and args object
        output = pipeline(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=7.5,
            generator=generator,
            output_type="pil",
            method="adj_init_noise",
            args=config
        )
        
        image = output.images[0]
        sanitized_prompt = "".join([c if c.isalnum() else "_" for c in prompt[:20]])
        save_path = os.path.join(args.output_dir, f"mitigated_{i}_{sanitized_prompt}.png")
        image.save(save_path)
        print(f"Saved to {save_path}")

    # Also run BASELINE for comparison?
    # Optional, but good practice.
    # User only asked for the specific algorithm code.

if __name__ == "__main__":
    main()
