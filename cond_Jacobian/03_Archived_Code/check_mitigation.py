import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler

class MitigationPipeline(StableDiffusionPipeline):
    """
    논문 및 제공된 local_sd_pipeline.py의 로직을 기반으로 한 커스텀 파이프라인
    """
    
    def adj_init_noise_batch_wise(
        self,
        latents,
        prompt_embeds,
        num_inference_steps=50,
        adj_iters=2,
        rho=50,
        gamma=0.7,
    ):
        """Batch-wise mitigation: Sharpness-aware update rule (Algorithm 1)"""
        device = self._execution_device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 논문의 식 (10)에 따른 업데이트
        for _ in range(adj_iters):
            # 1. 현재 latent에 대한 그라디언트 근사 계산
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = self.unet(
                latent_model_input,
                timesteps[0],
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            eps_tilde = noise_pred_text - noise_pred_uncond
            
            # Norm 계산 및 섭동(delta) 생성
            eps_tilde_mag = torch.sqrt(torch.sum(eps_tilde**2, dim=(1, 2, 3), keepdim=True))
            delta_hat = rho * (eps_tilde / (eps_tilde_mag + 1e-8)) # 0으로 나누기 방지
            latents_w_delta_hat = latents + delta_hat

            # 2. 섭동된 위치에서의 그라디언트 계산
            latent_model_input = torch.cat([latents_w_delta_hat] * 2)
            noise_pred_w_delta_hat = self.unet(
                latent_model_input,
                timesteps[0],
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            noise_pred_uncond_w_delta_hat, noise_pred_text_w_delta_hat = noise_pred_w_delta_hat.chunk(2)
            eps_tilde_w_delta_hat = noise_pred_text_w_delta_hat - noise_pred_uncond_w_delta_hat
            
            # Sharpness gradient 근사
            nabla_l_sharp = eps_tilde_w_delta_hat - eps_tilde
            
            # Latent 업데이트
            latents = latents - gamma * nabla_l_sharp
            
        return latents

    def adj_init_noise_per_sample(
        self,
        latents,
        prompt_embeds,
        num_inference_steps=50,
        guidance_scale=7.5,

        lr=0.01, # Reduced LR to prevent divergence
        optim_iters=50, 
        target_loss=0.9,
    ):
        """Per-sample mitigation: Direct backpropagation (Appendix A)"""
        device = self._execution_device
        
        # Freeze models
        self.unet.requires_grad_(False)
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
        if hasattr(self, "vae") and self.vae is not None:
            self.vae.requires_grad_(False)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        t = timesteps[0] # 초기 타임스텝 T

        # Latent 복제 및 그래디언트 활성화
        latents = latents.detach().clone()
        latents.requires_grad = True
        optimizer = torch.optim.AdamW([latents], lr=lr)
        
        # Uncond/Cond 임베딩 분리 (Per-sample은 개별 최적화)
        # prompt_embeds 구조: [uncond, cond]
        prompt_embeds = prompt_embeds.detach()
        
        print(f"Starting Per-sample optimization (lr={lr}, iters={optim_iters})...")
        for i in range(optim_iters):
            # 입력 스케일링
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 노이즈 예측
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            # 조건부 가이던스 벡터 (Conditional Guidance Vector)
            # 논문에서는 epsilon_cond - epsilon_uncond의 norm을 최소화
            diff = noise_pred_text - noise_pred_uncond
            loss = torch.norm(diff, p=2).mean()

            if i % 10 == 0:
                print(f"Step {i}: Loss={loss.item():.4f}, Latent Max={latents.abs().max().item():.2f}")

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Step {i}: Loss is NaN/Inf! Stops optimization.")
                break

            if loss.item() <= target_loss:
                print(f"Target loss reached at step {i}: {loss.item()}")
                break
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_([latents], max_norm=1.0)
            
            optimizer.step()
        
        print(f"Final Latent Stats: Mean={latents.mean().item():.2f}, Std={latents.std().item():.2f}")
        return latents.detach()
        
        return latents.detach()

    @torch.no_grad()
    def collect_trajectory_norms(
        self,
        prompt,
        latents=None,
        num_inference_steps=50,

        guidance_scale=0.0, # Unconditional sampling (g=0) for trajectory, but we compute cond noise for norm
    ):
        """
        논문의 Figure 2를 재현하기 위해, Unconditional Sampling Trajectory를 따라가며
        매 스텝마다 조건부 노이즈 예측의 크기(||epsilon_cond - epsilon_uncond||)를 기록함.
        Note: 단순히 pipe(guidance_scale=0)을 쓰면 cond noise를 계산하지 않으므로, 
        여기서는 수동으로 두 가지를 계산하고 update는 uncond noise로 수행함.
        """
        device = self._execution_device
        
        # 1. 텍스트 임베딩 준비
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            prompt, device, 1, True, None 
        )
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if negative_prompt_embeds.ndim == 2:
            negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)
            
        # [negative, positive] 순서로 결합
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 2. 타임스텝 설정
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Latents 준비 (입력받은 latents가 없으면 랜덤 생성)
        if latents is None:
            latents = torch.randn(
                (1, self.unet.config.in_channels, 64, 64),
                device=device,
                dtype=prompt_embeds.dtype
            )
        
        norms = []

        # 4. Denoising Loop
        for i, t in enumerate(timesteps):
            # CFG 미적용 궤적을 따르더라도, Norm 측정을 위해 두 가지 입력 모두 준비
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            # --- Norm 측정 (Attraction Force) ---
            # 논문의 || bar{epsilon}_theta(x_t, t, y) ||_2
            diff = noise_pred_text - noise_pred_uncond
            norm_val = torch.norm(diff, p=2).item()
            norms.append(norm_val)
            
            # --- 궤적 업데이트 (Without CFG) ---
            # 논문 캡션: "sampling without CFG". 즉, Unconditional 예측(또는 guidance_scale=0/1)만 사용
            # 여기서는 순수한 latent 분포 궤적을 보기 위해 unconditional prediction을 사용합니다.
            latents = self.scheduler.step(
                noise_pred_uncond, t, latents, return_dict=False
            )[0]

        return norms, timesteps.cpu().numpy()

# --- 실행 및 시각화 코드 ---

def main():
    # 설정
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    
    # 1. 파이프라인 로드
    print("Loading pipeline...")
    pipe = MitigationPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # 2. 초기 Latent 고정 (동일한 시작점에서 비교하기 위함)
    generator = torch.Generator(device=device).manual_seed(seed)
    init_latents = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        generator=generator,
        device=device,
        dtype=pipe.unet.dtype
    )


    
    # Load prompt from CSV
    csv_path = os.path.join(os.path.dirname(__file__), "prompts/memorized_laion_prompts.csv")
    if not os.path.exists(csv_path):
        # Fallback if run from a different directory
        csv_path = "init_noise_diffusion_memorization/prompts/memorized_laion_prompts.csv"
        
    print(f"Loading prompts from {csv_path}...")
    df = pd.read_csv(csv_path, sep=';')
    print(f"Loading prompts from {csv_path}...")
    df = pd.read_csv(csv_path, sep=';')
    
    # 3. Analyze 5 Random Prompts
    num_samples = 5
    sampled_df = df.sample(n=num_samples, random_state=seed) # Fixed seed for reproducibility but random selection
    
    for idx, row in sampled_df.iterrows():
        prompt = row['Caption']
        prompt_idx = row['Index'] if 'Index' in row else idx
        print(f"\n[{idx+1}/{num_samples}] Processing Prompt: '{prompt}'")

        # Re-initialize Latents for each prompt to ensure fair "random start" comparison
        # Or keep same seed if we want "same noise, different prompt" comparison.
        # Here we keep same INIT NOISE for consistency across prompts as requested by 'randomly select 5'.
        
        # --- Baseline ---
        print("Generating Baseline trajectory...")
        norms_baseline, steps = pipe.collect_trajectory_norms(
            prompt, latents=init_latents.clone()
        )

        # --- Batch-wise Mitigation ---
        print("Applying Batch-wise mitigation...")
        prompt_embeds, neg_embeds = pipe._encode_prompt(prompt, device, 1, True, None)
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if neg_embeds.ndim == 2:
            neg_embeds = neg_embeds.unsqueeze(0)
        concat_embeds = torch.cat([neg_embeds, prompt_embeds])
        
        latents_batch = init_latents.clone()
        latents_batch = pipe.adj_init_noise_batch_wise(
            latents=latents_batch,
            prompt_embeds=concat_embeds,
            rho=50, gamma=0.7, adj_iters=2
        )
        
        norms_batch, _ = pipe.collect_trajectory_norms(
            prompt, latents=latents_batch
        )

        # --- Per-sample Mitigation ---
        print("Applying Per-sample mitigation...")
        latents_per_sample = init_latents.clone()
        latents_per_sample = pipe.adj_init_noise_per_sample(
            latents=latents_per_sample,
            prompt_embeds=concat_embeds,
            target_loss=0.9, lr=0.01, optim_iters=50 
        )

        norms_sample, _ = pipe.collect_trajectory_norms(
            prompt, latents=latents_per_sample
        )

        # --- Plotting Attraction Basin ---
        plt.figure(figsize=(10, 6))
        plt.plot(steps, norms_baseline, label='Baseline', color='red', linewidth=2)
        plt.plot(steps, norms_batch, label='Batch-wise', color='orange', linestyle='--')
        plt.plot(steps, norms_sample, label='Per-sample', color='blue', linestyle='--')
        
        plt.title(f"Attraction Basin Analysis\nPrompt: '{prompt[:50]}...'", fontsize=14)
        plt.xlabel("Time steps (1000 -> 0)")
        plt.ylabel(r"$||\bar{\epsilon}_\theta(x_t, t, y)||_2$")
        plt.gca().invert_xaxis()
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        results_folder = './results/check_mitigation'
        os.makedirs(results_folder, exist_ok=True)
        
        sanitized_prompt = "".join(x for x in prompt[:20] if x.isalnum())
        plot_filename = f"basin_plot_{prompt_idx}_{sanitized_prompt}.png"
        plt.savefig(os.path.join(results_folder, plot_filename), dpi=300)
        plt.close()
        print(f"Plot saved to {plot_filename}")

        # --- Image Generation & Visualization ---
        print("Generating images...")
        img_baseline = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, latents=init_latents).images[0]
        img_batch = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, latents=latents_batch).images[0]
        img_sample = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, latents=latents_per_sample).images[0]

        # Comparison Plot
        fig_img, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
        
        axs[0].imshow(img_baseline)
        axs[0].set_title("Baseline", fontsize=16, fontweight='bold')
        axs[0].axis("off")
        
        axs[1].imshow(img_batch)
        axs[1].set_title("Batch-wise", fontsize=16, fontweight='bold')
        axs[1].axis("off")
        
        axs[2].imshow(img_sample)
        axs[2].set_title("Per-sample", fontsize=16, fontweight='bold')
        axs[2].axis("off")
        
        plt.suptitle(f"Mitigation Comparison: '{prompt[:60]}...'", fontsize=20, y=1.05)
        plt.tight_layout()
        
        comp_filename = f"comparison_{prompt_idx}_{sanitized_prompt}.png"
        plt.savefig(os.path.join(results_folder, comp_filename), bbox_inches='tight')
        plt.close()
        print(f"Comparison image saved to {comp_filename}")


if __name__ == "__main__":
    main()