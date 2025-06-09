import torch
import safetensors.torch
import argparse
import os
from collections import OrderedDict

def convert_checkpoint_to_lora_safetensors(checkpoint_dir, output_file):
    """
    Loads a DeepSpeed-style checkpoint, extracts LoRA weights,
    and saves them as a ComfyUI-compatible .safetensors file.
    """
    print(f"Loading checkpoint from: {checkpoint_dir}")

    # --- 1. Aggregate all checkpoint parts into a single state dictionary ---
    full_state_dict = OrderedDict()
    
    # Find all model state parts. They can be named differently depending on the
    # DeepSpeed configuration (e.g., mp_rank_XX_model_states.pt or layer_XX-model_states.pt)
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith("model_states.pt"):
            checkpoint_files.append(os.path.join(checkpoint_dir, filename))

    if not checkpoint_files:
        print(f"Error: No checkpoint files (*model_states.pt) found in '{checkpoint_dir}'")
        return

    print(f"Found {len(checkpoint_files)} checkpoint file(s).")

    for pt_file in checkpoint_files:
        print(f"  - Loading {os.path.basename(pt_file)}...")
        part_state_dict = torch.load(pt_file, map_location="cpu")
        full_state_dict.update(part_state_dict)

    print("Successfully loaded all checkpoint parts.")

    # --- 2. Extract only the LoRA weights ---
    lora_state_dict = OrderedDict()
    for key, value in full_state_dict.items():
        if "lora" in key:
            lora_state_dict[key] = value

    if not lora_state_dict:
        print("\nWarning: No LoRA weights (keys containing 'lora') were found in the checkpoint.")
        print("Please ensure you were training a LoRA and that the checkpoint is correct.")
        return

    print(f"\nExtracted {len(lora_state_dict)} LoRA tensors.")

    # --- 3. Remap keys for ComfyUI compatibility ---
    # The goal is to get clean keys like 'diffusion_model.blocks.0.attn.q.lora_A.weight'
    comfy_lora_state_dict = OrderedDict()
    for key, value in lora_state_dict.items():
        # Remove common wrapper prefixes like 'module.'
        clean_key = key.replace("module.", "")
        
        # Add the prefix ComfyUI expects for U-Net/transformer LoRAs
        final_key = f"diffusion_model.{clean_key}"
        comfy_lora_state_dict[final_key] = value
    
    print("Remapped keys to ComfyUI format.")

    # --- 4. Save the final .safetensors file ---
    try:
        safetensors.torch.save_file(comfy_lora_state_dict, output_file)
        print(f"\nSuccessfully saved ComfyUI LoRA to: {output_file}")
    except Exception as e:
        print(f"\nError saving .safetensors file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Diffusion Pipe DeepSpeed checkpoint into a ComfyUI-compatible LoRA .safetensors file."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory (e.g., '.../global_step4063/')."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output .safetensors file (e.g., 'my_lora.safetensors')."
    )
    args = parser.parse_args()

    convert_checkpoint_to_lora_safetensors(args.checkpoint_dir, args.output_file) 