import torch
from collections import OrderedDict

def convert_muge_model_to_chinese_clip(muge_model_path, output_path):
    """
    Convert a MUGE training checkpoint to Chinese-CLIP compatible format.
    
    Args:
        muge_model_path (str): Path to the MUGE checkpoint (usually epoch_*.pt)
        output_path (str): Path to save the converted model
    """
    # Load the MUGE checkpoint
    muge_checkpoint = torch.load(muge_model_path, map_location="cpu")
    
    # Create new state dict with Chinese-CLIP compatible keys
    new_state_dict = OrderedDict()
    
    # Mapping dictionary for key name conversion
    # This mapping is based on Chinese-CLIP's expected model structure
    key_mapping = {
        # Visual encoder mappings
        "visual.": "visual_encoder.",
        "visual_encoder.transformer": "visual_encoder.transformer.resblocks",
        "visual_encoder.class_embedding": "visual_encoder.visual.class_embedding",
        "visual_encoder.positional_embedding": "visual_encoder.visual.positional_embedding",
        "visual_encoder.conv1": "visual_encoder.visual.conv1",
        "visual_encoder.ln_pre": "visual_encoder.visual.ln_pre",
        "visual_encoder.ln_post": "visual_encoder.visual.ln_post",
        "visual_encoder.proj": "visual_encoder.visual.proj",
        
        # Text encoder mappings
        "text.": "text_encoder.",
        "text_encoder.transformer": "text_encoder.transformer.resblocks",
        "text_encoder.token_embedding": "text_encoder.token_embedding.weight",
        "text_encoder.positional_embedding": "text_encoder.positional_embedding",
        "text_encoder.ln_final": "text_encoder.ln_final",
        "text_encoder.text_projection": "text_encoder.text_projection",
        
        # Logit scale mapping
        "logit_scale": "logit_scale",
    }
    
    # Perform key conversion
    for old_key, param in muge_checkpoint["state_dict"].items():
        # Find the corresponding new key
        new_key = old_key
        for old_pattern, new_pattern in key_mapping.items():
            if old_pattern in old_key:
                new_key = old_key.replace(old_pattern, new_pattern)
                break
        
        # Add to the new state dict
        new_state_dict[new_key] = param
    
    # Create the final model dictionary with the structure Chinese-CLIP expects
    converted_model = {
        "state_dict": new_state_dict,
        "epoch": muge_checkpoint.get("epoch", 0),
        "version": "chinese-clip-converted"
    }
    
    # Save the converted model
    torch.save(converted_model, output_path)
    print(f"Converted model saved to {output_path}")

# Example usage:
# convert_muge_model_to_chinese_clip("./muge_epoch3.pt", "./chinese_clip_compatible.pt") 