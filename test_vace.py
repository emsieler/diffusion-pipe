import os
import torch
from pathlib import Path
from models.wan_vace import WanVacePipeline
from utils.common import AUTOCAST_DTYPE

def test_vace_pipeline():
    # Test config based on wan_vace_14b_min_vram.toml
    config = {
        'model': {
            'type': 'wan_vace', 
            'ckpt_path': '/home/em/code/volumetric-fix/Wan2.1-VACE-14B', 
            'dtype': 'bfloat16',
            'transformer_dtype': 'bfloat16',
            'timestep_sample_method': 'logit_normal',
            'llm_path': None,  # use default from ckpt
        }
    }

    print("Initializing Vace pipeline...")
    pipeline = WanVacePipeline(config)
    
    # Load test video
    test_video_path = '/home/em/code/volumetric-fix/dataset/10frame_test.mp4'
    if not os.path.exists(test_video_path):
        print(f"Please provide a valid test video path. Current path {test_video_path} does not exist.")
        return

    print("Loading and preprocessing video...")
    preprocess_fn = pipeline.get_preprocess_media_file_fn()
    video_data_list = preprocess_fn(test_video_path, mask_filepath=None)
    
    video_tensor, mask = video_data_list[0]
    
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
    ids, mask = pipeline.text_encoder.tokenizer(
        test_prompt,
        return_mask=True,
        add_special_tokens=True
    )
    ids = ids.to(torch.device('cuda'))
    mask = mask.to(torch.device('cuda'))
    
    text_embeddings = text_encoder(ids, mask)

    # Prepare inputs
    inputs = {
        'latents': latents,
        'text_embeddings': text_embeddings,
        'seq_lens': None,  
        'mask': mask
    }

    print("Running forward pass...")
    with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
        model_inputs, targets = pipeline.prepare_inputs(inputs)
        x_t, t, vace_context, text_embeddings, seq_lens, vace_context_scale, clip_fea, y = model_inputs
        
        print("Converting to layers...")
        layers = pipeline.to_layers()
        
        print("Running through layers...")
        x = x_t
        for i, layer in enumerate(layers):
            print(f"Processing layer {i+1}/{len(layers)}")
            x = layer(model_inputs)
            
        print("\nTest completed successfully!")
        print(f"Input shape: {x_t.shape}")
        print(f"Output shape: {x.shape}")
        print(f"Number of layers: {len(layers)}")
        
        # Print memory usage
        print("\nMemory usage:")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

if __name__ == "__main__":
    test_vace_pipeline() 