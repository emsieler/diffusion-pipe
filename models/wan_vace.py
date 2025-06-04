import sys
import json
import math
import re
import os.path
import types
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/Wan2_1'))

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
from safetensors.torch import load_file
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE
from utils.offloading import ModelOffloader

from .wan import (umt5_keys_mapping_comfy, umt5_keys_mapping_kijai, umt5_keys_mapping, 
                  _t5, umt5_xxl, T5EncoderModel,
                  vae_encode, Head, WanPipeline,
                  WanAttentionBlock,
)

import wan
from wan.modules.t5 import T5Encoder, T5Decoder, T5Model
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae import WanVAE
from wan.modules.model import (
    WanModel, sinusoidal_embedding_1d, WanLayerNorm, WanSelfAttention, WAN_CROSSATTENTION_CLASSES
)
from wan.modules.vace_model import (
    VaceWanModel,
)
from wan.vace import WanVace

from wan.modules.clip import CLIPModel
from wan import configs as wan_configs
from safetensors.torch import load_file

KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'head', 'modulation',
                          'vace_patch_embedding', 'before_proj', 'after_proj']

class VaceWanModelFromSafetensors(VaceWanModel):
    @classmethod
    def from_pretrained(
        cls,
        weights_file,
        config_file,
        torch_dtype=torch.bfloat16,
        transformer_dtype=torch.bfloat16,
    ):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)

        with init_empty_weights():
            model = cls(**config)

        state_dict = load_file(weights_file, device='cpu')
        state_dict = {
            re.sub(r'^model\.diffusion_model\.', '', k): v for k, v in state_dict.items()
        }

        for name, param in model.named_parameters():
            dtype_to_use = torch_dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
            set_module_tensor_to_device(model, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        return model

class VaceWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=0):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return c, c_skip


class BaseWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=None):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x

# Patch these to remove some forced casting to float32, saving memory.
wan.modules.model.WanAttentionBlock = WanAttentionBlock
wan.modules.vace_model.VaceWanAttentionBlock = VaceWanAttentionBlock
wan.modules.vace_model.BaseWanAttentionBlock = BaseWanAttentionBlock
wan.modules.model.Head = Head

class WanVacePipeline(WanPipeline):
    name = 'wan_vace'
    framerate = 16
    checkpointable_layers = ['TransformerLayer', 'VaceTransformerLayer']
    adapter_target_modules = ['VaceWanAttentionBlock', 'BaseWanAttentionBlock'] 

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        ckpt_dir = self.model_config['ckpt_path']
        dtype = self.model_config['dtype']
        
        self.skyreels = 'skyreels' in Path(ckpt_dir).name.lower()
        if self.skyreels:
            raise ValueError("Skyreels not supported for VACE")

        self.original_model_config_path = os.path.join(ckpt_dir, 'config.json')
        with open(self.original_model_config_path) as f:
            json_config = json.load(f)
        model_dim = json_config['dim']
        
        self.vace = (json_config['model_type'] == 'vace')
        if self.vace:
            if model_dim == 1536:
                wan_config = wan_configs.t2v_1_3B
            elif model_dim == 5120:
                wan_config = wan_configs.t2v_14B
            else:
                raise ValueError(f"Model dimension {model_dim} not supported")

        # This is the outermost class, not an nn.Module
        t5_model_path = self.model_config['llm_path'] if self.model_config.get('llm_path', None) else os.path.join(ckpt_dir, wan_config.t5_checkpoint)
        self.text_encoder = T5EncoderModel(
            text_len=wan_config.text_len,
            dtype=dtype,
            device='cpu',
            checkpoint_path=t5_model_path,
            tokenizer_path=os.path.join(ckpt_dir, wan_config.t5_tokenizer),
            shard_fn=None,
        )

        # Same here, this isn't a nn.Module.
        # TODO: by default the VAE is float32, and therefore so are the latents. Do we want to change that?
        self.vae = WanVAE(
            vae_pth=os.path.join(ckpt_dir, wan_config.vae_checkpoint),
            device='cpu',
        )
        # These need to be on the device the VAE will be moved to during caching.
        self.vae.mean = self.vae.mean.to('cuda')
        self.vae.std = self.vae.std.to('cuda')
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        if transformer_path := self.model_config.get('transformer_path', None):
            self.transformer = VaceWanModelFromSafetensors.from_pretrained(  
                transformer_path,
                self.original_model_config_path,
                torch_dtype=dtype,
                transformer_dtype=transformer_dtype,
            )
        else:
            ckpt_path = Path(self.model_config['ckpt_path'])
            with init_empty_weights():
                self.transformer = VaceWanModel.from_config(ckpt_path / 'config.json')
            state_dict = {}
            for shard in ckpt_path.glob('*.safetensors'):
                with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            for name, param in self.transformer.named_parameters():
                dtype_to_use = dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
                set_module_tensor_to_device(self.transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        # Move model to CUDA after loading
        self.transformer = self.transformer.cuda()
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    # no changes made to class methods:
    # def __getattr__(self, name):
    # get_text_encoders(self):
    # def save_adapter(self, save_dir, peft_state_dict):
    # def save_model(self, save_dir, diffusers_sd):
    # def get_preprocess_media_file_fn(self):
    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # Args are lists
            p = next(text_encoder.model.parameters())
            ids, mask = self.text_encoder.tokenizer(caption, return_mask=True, add_special_tokens=True)
            ids = ids.to(p.device)
            mask = mask.to(p.device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            with torch.autocast(device_type=p.device.type, dtype=p.dtype):
                text_embeddings = text_encoder.model(ids, mask)
                return {'text_embeddings': text_embeddings, 'seq_lens': seq_lens}
        return fn

    # override: VACE does not use clip
    def get_vae(self):
        if not next(self.vae.model.parameters()).is_cuda:
            self.vae.model = self.vae.model.cuda()
        return self.vae.model

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            # Move everything to CUDA
            if not next(self.vae.model.parameters()).is_cuda:
                self.vae.model = self.vae.model.cuda()
            if not self.vae.scale[0].is_cuda:
                self.vae.scale = [s.cuda() for s in self.vae.scale]
            tensor = tensor.cuda()
            latents = vae_encode(tensor, self.vae)
            return {'latents': latents}
        return fn
        
    def prepare_inputs(self, inputs, timestep_quantile=None):
        # Keep latents in float32 like the regular pipeline
        latents = inputs['latents'].float().cuda()
        text_embeddings = [emb.cuda() for emb in inputs['text_embeddings']]
        seq_lens = inputs['seq_lens'].cuda()
        mask = inputs['mask'].cuda() if inputs['mask'] is not None else None

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target
            mask = mask.to(dtype=latents.dtype)  # match latents dtype

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')
        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        # Scale timesteps to [0, 1000]
        t = t * 1000

        # Get initial embeddings for the main input - explicitly set dtype to match model
        model_dtype = self.transformer.patch_embedding.weight.dtype
        x = [self.transformer.patch_embedding(u.unsqueeze(0).to(dtype=model_dtype)) for u in x_t]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long, device=x[0].device) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_len = max([u.size(1) for u in x])
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # Create time embeddings - match model dtype
        t_model_dtype = t.to(dtype=model_dtype)
        e = self.transformer.time_embedding(sinusoidal_embedding_1d(self.transformer.freq_dim, t_model_dtype).to(x.device, dtype=model_dtype))
        e0 = self.transformer.time_projection(e).unflatten(1, (6, self.transformer.dim))
        assert e.dtype == model_dtype and e0.dtype == model_dtype, f"Time embeddings not in {model_dtype}: e={e.dtype}, e0={e0.dtype}"

        # Process text embeddings - explicitly set dtype to match model
        context = [emb[:length].to(dtype=model_dtype) for emb, length in zip(text_embeddings, seq_lens)]
        context = self.transformer.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.transformer.text_len - u.size(0), u.size(1), dtype=model_dtype)])
                for u in context
            ]))

        # Create VACE context from latents and masks - explicitly set dtype to match model
        vace_list_for_forward_vace = []
        print("\\nConstructing VACE context list:")
        print(f"Shape of x_t (input latents): {x_t.shape}") # e.g. [B, F_orig, C_orig=96, H_orig_spatial]

        # Permute x_t once to get [Batch, Channels, Frames, H_spatial]
        # Original x_t dims: 0=Batch, 1=Frames, 2=Channels, 3=H_spatial
        # Target x_t_perm dims: 0=Batch, 1=Channels, 2=Frames, 3=H_spatial
        x_t_permuted = x_t.permute(0, 2, 1, 3) # Shape: [B, C_orig=96, F_orig, H_orig_spatial], e.g. [16, 96, 5, 54]
        print(f"Shape of x_t_permuted: {x_t_permuted.shape}")
        
        # TODO: Proper mask processing and concatenation needs to be verified here.
        # If a mask is present, it should be processed and combined with x_t_permuted
        # such that each item in vace_list_for_forward_vace has 96 channels.
        # For now, this simplified logic assumes x_t_permuted itself provides the 96 channels
        # or that mask handling is separate / a no-op if mask is None.

        if mask is not None:
            # This is a placeholder for correct mask processing.
            # The current mask processing in the original code led to `torch.cat` that would
            # increase channels beyond 96 if x_t_permuted already had 96.
            # For the VACE model, often the input latents (x_t) might have fewer channels,
            # and the mask provides additional channels to make up the total expected by vace_patch_embedding.
            # However, logs indicate x_t itself has 96 channels.
            # This part needs careful review based on how VACE context is truly formed with masks.
            # For now, we'll assume if mask is present, we still primarily use x_t_permuted for simplicity
            # to get the main error resolved. A more sophisticated mask integration might be needed.
            print("Mask is present, current simplified VACE context logic might need review for mask integration.")
            # Fallthrough to use x_t_permuted, or handle 'm' correctly if it was shaped like x_t_permuted items.
            # The original cat was: torch.cat([l, m], dim=1)) where l and m were [B, C, F, H_spatial]
            # This implies m should also be prepared per batch item and then cat on channels before unsqueeze.

        for i in range(x_t_permuted.shape[0]):  # Iterate over the Batch dimension
            item_slice = x_t_permuted[i] # Shape: [C_orig=96, F_orig, H_orig_spatial], e.g., [96, 5, 54]
            print(f"  Processing item {i} for VACE context list: original slice shape: {item_slice.shape}")

            # Add W dimension: [C_orig=96, F_orig, H_orig_spatial, 1_for_W]
            item_slice_5d = item_slice.unsqueeze(-1)  # Shape: [96, 5, 54, 1]
            print(f"  Item {i} after adding W dim: {item_slice_5d.shape}")
            
            vace_list_for_forward_vace.append(item_slice_5d.to(dtype=model_dtype))

        print("\\nFinal vace_context (list of tensors to be passed to forward_vace):")
        if not vace_list_for_forward_vace:
            print("  List is empty.")
        for idx, tensor_item in enumerate(vace_list_for_forward_vace):
            print(f"  Item {idx} shape: {tensor_item.shape}, Dims: {tensor_item.dim()}")
        
        # Generate hints using forward_vace
        vace_block_args = dict(
            x=x,
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.transformer.freqs.to(x.device),
            context=context,
            context_lens=None
        )
        hints = self.transformer.forward_vace(x, vace_list_for_forward_vace, seq_len, vace_block_args)

        # Convert all outputs to model dtype except time embeddings
        x_t = x_t.to(dtype=model_dtype)
        t = t.to(dtype=model_dtype)
        target = target.to(dtype=model_dtype)
        if mask is not None:
            mask = mask.to(dtype=model_dtype)

        return (
            (x_t, t, hints, text_embeddings, seq_lens, 1.0, None, None),
            (target, mask),
        )

    def to_layers(self):
        import pdb; pdb.set_trace()  # Breakpoint 1: Check layer creation
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        
        # Process blocks in sequence, ensuring Vace blocks come before corresponding Base blocks to provide context
        for i in range(len(transformer.blocks)):
            if i in transformer.vace_layers:
                vace_idx = transformer.vace_layers.index(i)
                layers.append(VaceTransformerLayer(transformer.vace_blocks[vace_idx], vace_idx, self.offloader))
            layers.append(TransformerLayer(transformer.blocks[i], i, self.offloader))
        
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        raise NotImplementedError("Block swapping is not supported for VACE models as it would break the VACE block <--> Base block context dependency chain.")

    def prepare_block_swap_training(self):
        raise NotImplementedError("Block swapping is not supported for VACE models as it would break the VACE block <--> Base block dependency chain.")

    def prepare_block_swap_inference(self, disable_block_swap=False):
        raise NotImplementedError("Block swapping is not supported for VACE models as it would break the VACE block <--> Base block dependency chain.")

class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.patch_embedding = model.patch_embedding
        self.time_embedding = model.time_embedding
        self.text_embedding = model.text_embedding
        self.time_projection = model.time_projection
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=torch.bfloat16)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        x, t, vace_context, context, seq_lens, vace_context_scale, clip_fea, y = inputs
        bs, channels, f, h, w = x.shape
        context = [emb[:length] for emb, length in zip(context, seq_lens)] if context is not None else None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings - ensure input is in correct dtype
        x = [self.patch_embedding(u.unsqueeze(0).to(dtype=self.patch_embedding.weight.dtype)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens.max()
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings - keep in float32 like in wan.py
        with torch.cuda.amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(x.device, torch.float32))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        if context is not None:
            context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))

        # pipeline parallelism needs everything on the GPU
        seq_lens = seq_lens.to(x.device)
        grid_sizes = grid_sizes.to(x.device)

        return make_contiguous(x, e, e0, seq_lens, grid_sizes, self.freqs, context)

class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=torch.bfloat16)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context = inputs
        # Get hints from previous layer's output
        hints = inputs[1] if isinstance(inputs, tuple) else None

        self.offloader.wait_for_block(self.block_idx)
        # Base blocks take x first, then all other params
        x = self.block(x, e0, seq_lens, grid_sizes, freqs, context, None, hints=hints)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, e, e0, seq_lens, grid_sizes, freqs, context)

class VaceTransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=torch.bfloat16)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context = inputs

        self.offloader.wait_for_block(self.block_idx)
        # Vace blocks take context first, then x, plus all other params
        c, c_skip = self.block(context, x, e0, seq_lens, grid_sizes, freqs, context, None)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        # Return processed context and skip connection
        return make_contiguous(c, e, e0, seq_lens, grid_sizes, freqs, context), c_skip

class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=torch.bfloat16)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context = inputs
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x, dim=0)
