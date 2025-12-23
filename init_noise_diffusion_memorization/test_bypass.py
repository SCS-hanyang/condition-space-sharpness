
import torch
from local_sd_pipeline import LocalStableDiffusionPipeline

def test_bypass():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = LocalStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        safety_checker=None
    ).to(device)
    
    prompt = "Test prompt"
    text_input = pipe.tokenizer(prompt, return_tensors="pt").to(device)
    
    # 1. Get embeddings manually
    input_layer = pipe.text_encoder.get_input_embeddings()
    inputs_embeds = input_layer(text_input.input_ids)
    
    print(f"Embeddings shape: {inputs_embeds.shape}")
    
    # 2. Try calling text_model directly
    print("Calling pipe.text_encoder.text_model(inputs_embeds=...)")
    try:
        outputs = pipe.text_encoder.text_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )
        print("Success!")
        print(f"Result keys: {outputs.keys() if hasattr(outputs, 'keys') else 'tuple'}")
        
        if hasattr(outputs, 'pooler_output'):
             print(f"Pooler output shape: {outputs.pooler_output.shape}")
        if hasattr(outputs, 'last_hidden_state'):
             print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
             
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_bypass()
