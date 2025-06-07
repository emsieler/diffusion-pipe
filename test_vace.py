import os
import torch
from pathlib import Path
from models.wan_vace import WanVacePipeline
from utils.common import AUTOCAST_DTYPE
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/vace_test.yaml')
    return parser.parse_args()

def prepare_pipeline_inputs(pipeline, vae, text_encoder, video_tensor, mask_tensor, caption, is_video=True):
    """
    Uses the pipeline's component functions to construct the complete
    input dictionary required by the `prepare_inputs` method.
    This mirrors the logic used by the internal data loader.
    """
    pixel_values = (video_tensor, mask_tensor)
    
    # 1. Call the VAE function to get the dictionary of latents
    latents_dict = pipeline.get_call_vae_fn(vae)(pixel_values)

    # 2. Call the text encoder function
    text_encoder_outputs = pipeline.get_call_text_encoder_fn(text_encoder)(caption, is_video)
    
    # 3. Merge the results and add the mask
    inputs = {**latents_dict, **text_encoder_outputs, 'mask': mask_tensor}
    
    return inputs

def test_vace_pipeline():
    config = {
        'model': {
            'type': 'wan_vace', 
            'ckpt_path': '/workspace/models/Wan2.1-VACE-1.3B', 
            'dtype': 'bfloat16',
            'transformer_dtype': 'bfloat16',
            'timestep_sample_method': 'logit_normal',
            'llm_path': None,  # use default from ckpt
        }
    }

    # Initialize pipeline from the config dictionary
    pipeline = WanVacePipeline(config)
    print("Loading diffusion model...")
    
    # Load test video
    test_video_path = '/workspace/dataset/10frame_test.mp4'
    if not os.path.exists(test_video_path):
        print(f"Please provide a valid test video path. Current path {test_video_path} does not exist.")
        return

    print("Loading and preprocessing video...")
    preprocess_fn = pipeline.get_preprocess_media_file_fn()
    video_data_list = preprocess_fn(test_video_path, mask_filepath=None)
    
    video_tensor, spatial_mask = video_data_list[0]
    
    # Add batch dimension and move to correct device/dtype
    vae = pipeline.get_vae()
    p = next(vae.parameters())
    device, dtype = p.device, p.dtype
    video_tensor = video_tensor.unsqueeze(0).to(device, dtype)
    
    # Process through VAE to get latents
    vae_fn = pipeline.get_call_vae_fn(vae)
    latents_dict = vae_fn(video_tensor)
    latents = latents_dict['latents']
    
    print("Generating text embeddings...")
    text_encoder = pipeline.get_text_encoders()[0]
    
    # Move text encoder to CUDA first
    text_encoder = text_encoder.cuda()
    
    # Get text embeddings using the pipeline's text encoder
    test_prompt = ["A test video"]
    ids, text_mask = pipeline.text_encoder.tokenizer(
        test_prompt,
        return_mask=True,
        add_special_tokens=True
    )
    ids = ids.to(torch.device('cuda'))
    text_mask = text_mask.to(torch.device('cuda'))
    
    text_embeddings = text_encoder(ids, text_mask)

    # Use the new helper function to prepare inputs
    inputs = prepare_pipeline_inputs(pipeline, vae, text_encoder, video_tensor, spatial_mask, test_prompt, is_video=True)
    
    print("Running forward pass...")
    model_inputs, (target, target_mask) = pipeline.prepare_inputs(inputs)

    with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
        x_t, t, vace_context, text_embeddings, seq_lens, vace_context_scale, clip_fea, y = model_inputs
        
        print("Converting to layers...")
        layers = pipeline.to_layers()
        
        print("Running through layers...")
        for i, layer in enumerate(layers):
            print(f"Processing layer {i+1}/{len(layers)}")
            model_inputs = layer(*model_inputs)

        final_output = model_inputs
        print(f"Final output shape: {final_output.shape}")
        print(f"Final output dtype: {final_output.dtype}")
        
        # Print memory usage
        print("\nMemory usage:")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

if __name__ == "__main__":
    test_vace_pipeline() 