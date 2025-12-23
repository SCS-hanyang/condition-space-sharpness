
import torch
from local_sd_pipeline import LocalStableDiffusionPipeline

def check_structure():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/stable-diffusion-v1-4"
    
    pipe = LocalStableDiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None
    ).to(device)
    
    # 1. Run Text Encoder manually to get pooled output
    prompt = "A test prompt"
    input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Enable grad
    pipe.text_encoder.train() # Make sure we can trace
    
    # Hook output
    outputs = pipe.text_encoder(input_ids, output_hidden_states=True)
    last_hidden_state = outputs.last_hidden_state # [1, 77, 768]
    pooler_output = outputs.pooler_output # [1, 768]
    
    last_hidden_state.requires_grad_(True)
    pooler_output.requires_grad_(True)
    
    # 2. Run UNet
    latents = torch.randn((1, 4, 64, 64), device=device)
    t = torch.tensor([1], device=device)
    
    # Standard SD 1.4 passed last_hidden_state as encoder_hidden_states
    # Does it use pooled?
    # Usually NO.
    try:
        # Some custom pipelines allow passing pooled/class_labels
        res = pipe.unet(latents, t, encoder_hidden_states=last_hidden_state).sample
        loss = res.sum()
        loss.backward()
        
        print(f"Grad of sequence (encoder_hidden_states): {last_hidden_state.grad is not None}")
        if last_hidden_state.grad is not None:
             print(f"  Sum: {last_hidden_state.grad.abs().sum().item()}")
             
        print(f"Grad of pooled (pooler_output): {pooler_output.grad is not None}")
        if pooler_output.grad is not None:
             print(f"  Sum: {pooler_output.grad.abs().sum().item()}")
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_structure()
