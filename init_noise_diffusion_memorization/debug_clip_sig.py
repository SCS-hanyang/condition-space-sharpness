
import torch
import inspect
from local_sd_pipeline import LocalStableDiffusionPipeline
from transformers import CLIPTextModel

def debug_text_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/stable-diffusion-v1-4"
    
    print(f"Loading model: {model_id}")
    pipe = LocalStableDiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None
    ).to(device)
    
    text_encoder = pipe.text_encoder
    print(f"Text Encoder Type: {type(text_encoder)}")
    
    sig = inspect.signature(text_encoder.forward)
    print(f"Forward Signature: {sig}")
    
    if "inputs_embeds" not in sig.parameters:
        print("WARNING: 'inputs_embeds' is MISSING from forward signature!")
        
        # Check if it has a 'text_model' attribute which might be the actual transformer
        if hasattr(text_encoder, 'text_model'):
            print(f"Has text_model? Yes. Type: {type(text_encoder.text_model)}")
            sub_sig = inspect.signature(text_encoder.text_model.forward)
            print(f"Sub-model Forward Signature: {sub_sig}")
            
    else:
        print("'inputs_embeds' is present in forward signature.")

if __name__ == "__main__":
    debug_text_encoder()
