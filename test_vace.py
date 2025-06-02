import os
import torch
from pathlib import Path
from models.wan_vace import WanVacePipeline
from utils.common import AUTOCAST_DTYPE

def test_vace_pipeline():
    # Test configuration based on wan_vace_14b_min_vram.toml
    config = {
        'model': {
            'type': 'wan_vace',  # Changed to wan_vace
            'ckpt_path': '/data2/imagegen_models/Wan2.1-T2V-14B',  # From toml
            'dtype': 'bfloat16',
            'transformer_dtype': 'float8',  # From toml
            'timestep_sample_method': 'logit_normal',
            'llm_path': None,  # Will use default from checkpoint
        }
    }

    print("Initializing Vace pipeline...")
    pipeline = WanVacePipeline(config)
    
    # Load test video
    test_video_path = 'path/to/test/video.mp4'  # Update this path
    if not os.path.exists(test_video_path):
        print(f"Please provide a valid test video path. Current path {test_video_path} does not exist.")
        return

    print("Loading and preprocessing video...")
    preprocess_fn = pipeline.get_preprocess_media_file_fn()
    video_data = preprocess_fn(test_video_path)
    
    print("Generating text embeddings...")
    text_encoder = pipeline.get_text_encoders()[0]
    text_embeddings = text_encoder(
        ["A test video"],  # Test prompt
        torch.device('cuda')
    )

    # Prepare inputs
    inputs = {
        'latents': video_data['latents'],
        'text_embeddings': text_embeddings,
        'seq_lens': video_data['seq_lens'],
        'mask': video_data.get('mask', None)
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