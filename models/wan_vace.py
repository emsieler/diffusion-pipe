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

# --- ALL PATCHES MUST BE APPLIED BEFORE ANY OTHER MODULES ARE IMPORTED ---

# --- Patch 1: VacePatchedWanSelfAttention ---
# Reason: The public VACE checkpoints have a different architecture than the code
# in the Wan2_1 submodule. This patch makes this file compatible
# with the public checkpoint by:
#   1. Adding bias terms to the Q, K, V, and O linear layers (`bias=True`), which exist
#      in the checkpoint but are disabled in the submodule code.
#   2. Implementing a custom `LayerNormNoBias` for QK normalization to match the
#      checkpoint's layers, which do not have an affine bias term.
# The `forward` method is also reimplemented to explicitly use the submodule's
# `rope_apply` function for applying 2D positional embeddings, ensuring correctness.
import wan
from wan.modules.model import WanSelfAttention, rope_apply
from wan.modules.attention import flash_attention

class LayerNormNoBias(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias = None

class VacePatchedWanSelfAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super(WanSelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.dim_head = dim // num_heads
        assert self.dim_head * num_heads == dim
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.o = nn.Linear(dim, dim, bias=True)
        if qk_norm:
            self.norm_q = LayerNormNoBias(dim, eps, elementwise_affine=True)
            self.norm_k = LayerNormNoBias(dim, eps, elementwise_affine=True)
        else:
            self.norm_q = self.norm_k = nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        original_dtype = x.dtype
        b, s, c = x.shape
        n, d = self.num_heads, self.dim_head
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        q, k, v = qkv_fn(x)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            q_applied = rope_apply(q, grid_sizes, freqs)
            k_applied = rope_apply(k, grid_sizes, freqs)
            x = flash_attention(q=q_applied, k=k_applied, v=v, k_lens=seq_lens, window_size=self.window_size)
        x = x.transpose(1, 2).reshape(b, s, c)
        x = self.o(x)
        return x.to(original_dtype)

wan.modules.model.WanSelfAttention = VacePatchedWanSelfAttention

# --- Patch 2: VaceWanAttentionBlock ---
# Reason: This patch fixes a latent bug in the original submodule code. The
# `VaceWanAttentionBlock.forward` method incorrectly passes all of its keyword
# arguments (`**kwargs`) to its parent class, `WanAttentionBlock`.
# Our pipeline explicitly passes `x` (original latents) and `e0` (projected
# time embedding) as part of these kwargs. However, the parent class's
# `forward` method does not accept them, which causes a `TypeError`.
# This patch fixes the bug by intercepting the call, removing the unexpected
# arguments, and then calling the parent method with a clean set of kwargs.
from wan.modules.vace_model import VaceWanAttentionBlock

def patched_vace_wan_attention_block_forward(self, c, **kwargs):
    if self.block_id == 0:
        c = self.before_proj(c) + kwargs['x']
    
    parent_kwargs_for_super = {
        'e': kwargs.get('e'),
        'seq_lens': kwargs.get('seq_lens'),
        'grid_sizes': kwargs.get('grid_sizes'),
        'freqs': kwargs.get('freqs'),
        'context': kwargs.get('context'),
        'context_lens': kwargs.get('context_lens'),
    }
    
    c = super(VaceWanAttentionBlock, self).forward(c, **parent_kwargs_for_super)
    c_skip = self.after_proj(c)
    return c, c_skip

VaceWanAttentionBlock.forward = patched_vace_wan_attention_block_forward

# --- Patch 3: BaseWanAttentionBlock ---
# Reason: This patch fixes a latent bug identical to the one in `VaceWanAttentionBlock`.
# The original `BaseWanAttentionBlock.forward` method passes all of its keyword
# arguments (`**kwargs`) to its parent, `WanAttentionBlock`.
# Our pipeline explicitly passes `e0`, `hints`, and `context_scale` as part of these kwargs,
# but the parent class does not accept them, causing a `TypeError`. This patch fixes the
# bug by intercepting the call, removing the unexpected arguments, and ensuring the
# required `e` (original time embedding) is passed correctly.
from wan.modules.vace_model import BaseWanAttentionBlock

def patched_base_wan_attention_block_forward(self, x, e, hints, context_scale=1.0, **kwargs):
    parent_kwargs_for_super = {
        'seq_lens': kwargs.get('seq_lens'),
        'grid_sizes': kwargs.get('grid_sizes'),
        'freqs': kwargs.get('freqs'),
        'context': kwargs.get('context'),
        'context_lens': kwargs.get('context_lens'),
    }
    
    x = super(BaseWanAttentionBlock, self).forward(x, e=e, **parent_kwargs_for_super)
    
    if self.block_id is not None:
        # hints is a tensor stacked on dim 0
        x = x + hints[self.block_id] * context_scale
    return x

BaseWanAttentionBlock.forward = patched_base_wan_attention_block_forward

# --- End of Patches ---

from models.base import BasePipeline, PreprocessMediaFile
from utils.common import AUTOCAST_DTYPE
from utils.offloading import ModelOffloader

from .wan import (umt5_keys_mapping_comfy, umt5_keys_mapping_kijai, umt5_keys_mapping, 
                  _t5, umt5_xxl, T5EncoderModel,
                  vae_encode, Head, WanPipeline,
)

from wan.modules.t5 import T5Encoder, T5Decoder, T5Model
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae import WanVAE
from wan.modules.model import (
    WanModel, sinusoidal_embedding_1d, WanLayerNorm, WAN_CROSSATTENTION_CLASSES
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

class WanVacePipeline(WanPipeline):
    name = 'wan_vace'
    framerate = 16
    checkpointable_layers = ['VaceHintGenerationLayer', 'TransformerLayer']
    adapter_target_modules = ['VacePatchedWanSelfAttention', 'BaseWanAttentionBlock'] 

    def __init__(self, config):
        super().__init__(config)
        self.transformer = self.load_diffusion_model()
        
        # VACE's architecture is not compatible with the block-swapping memory optimization
        # in ModelOffloader. Each VaceHintGenerationLayer
        # must run before its corresponding TransformerLayer. Block swapping would break
        # the dependency chain, so we init a "dummy" offloader with swapping
        # disabled  to satisfy the training pipeline's requirement
        # for an offloader object without causing errors.
        self.offloader = ModelOffloader(
            block_type=None,
            blocks=[],
            num_blocks=0,
            blocks_to_swap=0,
            supports_backward=False,
            device=torch.device('cpu'),
            reentrant_activation_checkpointing=False,
        )

    def load_diffusion_model(self):
        model_config = self.model_config
        ckpt_path = Path(model_config['ckpt_path'])
        
        dtype_val = model_config['dtype']
        if isinstance(dtype_val, str):
            dtype = getattr(torch, dtype_val)
        else:
            dtype = dtype_val

        transformer_dtype_val = model_config.get('transformer_dtype', dtype_val)
        if isinstance(transformer_dtype_val, str):
            transformer_dtype = getattr(torch, transformer_dtype_val)
        else:
            transformer_dtype = transformer_dtype_val

        with open(ckpt_path / 'config.json', "r", encoding="utf-8") as f:
            config = json.load(f)

        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)

        with init_empty_weights():
            model = VaceWanModel(**config)

        # Multi-part safetensors loading
        state_dict = {}
        shards = sorted(list(ckpt_path.glob('*.safetensors')))
        if not shards:
             raise FileNotFoundError(f"No safetensors files found in {ckpt_path}")
        for shard in shards:
            print(f"Loading shard: {shard}")
            with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        
        state_dict = {
            re.sub(r'^model\.diffusion_model\.', '', k): v for k, v in state_dict.items()
        }

        for name, param in model.named_parameters():
            dtype_to_use = dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
            set_module_tensor_to_device(model, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        model = model.cuda().eval().requires_grad_(False)
        
        for name, p in model.named_parameters():
            p.original_name = name

        return model

    def get_loss_fn(self):
        def loss_fn(pred, labels):
            target, target_mask = labels
            # target_mask may be None or an empty tensor (from pipeline splitting)
            if target_mask is None or target_mask.numel() == 0:
                loss = F.mse_loss(pred.float(), target.float())
            else:
                loss = (F.mse_loss(pred.float(), target.float(), reduction="none") * target_mask).sum() / target_mask.sum()

            return loss
        return loss_fn

    def get_text_encoders(self):
        if not next(self.text_encoder.model.parameters()).is_cuda:
            self.text_encoder.model = self.text_encoder.model.cuda()
        return [self.text_encoder.model]

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            p = next(text_encoder.parameters())
            device = p.device
            ids, mask = self.text_encoder.tokenizer(caption, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            with torch.amp.autocast('cuda', dtype=p.dtype):
                text_embeddings = text_encoder(ids, mask)
                return {'text_embeddings': text_embeddings, 'seq_lens': seq_lens}
        return fn

    def get_vae(self):
        if not next(self.vae.model.parameters()).is_cuda:
            self.vae.model = self.vae.model.cuda()
        return self.vae.model

    def get_call_vae_fn(self, vae):
        def fn(pixel_values):
            if isinstance(pixel_values, tuple):
                tensor, mask = pixel_values
            else:
                tensor, mask = pixel_values, None

            if not next(self.vae.model.parameters()).is_cuda:
                self.vae.model = self.vae.model.cuda()
            if not self.vae.scale[0].is_cuda:
                self.vae.scale = [s.cuda() for s in self.vae.scale]
            
            tensor = tensor.cuda()

            if mask is not None:
                mask = mask.cuda()
                mask = torch.where(mask > 0.5, 1.0, 0.0)
                
                inactive_tensor = tensor * (1 - mask)
                reactive_tensor = tensor * mask

                inactive_latents = vae_encode(self.vae, inactive_tensor)
                reactive_latents = vae_encode(self.vae, reactive_tensor)
                
                latents = inactive_latents + reactive_latents
                return {'latents': latents, 'inactive_latents': inactive_latents, 'reactive_latents': reactive_latents, 'mask': mask}
            else:
                latents = vae_encode(self.vae, tensor)
                inactive_latents = latents
                reactive_latents = torch.zeros_like(latents)
                return {'latents': latents, 'inactive_latents': inactive_latents, 'reactive_latents': reactive_latents, 'mask': mask}
        return fn

    def process_vace_mask(self, mask, vae_stride=(1, 8, 8), device=None, dtype=None):
        """
        Process mask according to original VACE implementation.
        Args:
            mask: Input mask tensor of shape (bs, 1, num_frames, h, w)
            vae_stride: Tuple of (temporal_stride, height_stride, width_stride)
            device: Device to place output tensor on
            dtype: Data type of output tensor
        Returns:
            Processed mask tensor of shape (bs, 64, num_frames, h, w)
        """
        bs, _, num_frames, height, width = mask.shape
        
        # Reshape to match original processing
        mask = mask.view(bs, num_frames, height, vae_stride[1], width, vae_stride[2])
        mask = mask.permute(0, 3, 5, 1, 2, 4)
        mask = mask.reshape(bs, vae_stride[1] * vae_stride[2], num_frames, height, width)
        
        if device is not None:
            mask = mask.to(device)
        if dtype is not None:
            mask = mask.to(dtype)
        
        return mask

    def prepare_inputs(self, inputs, timestep_quantile=None):
        device = self.transformer.patch_embedding.weight.device
        model_dtype = self.transformer.patch_embedding.weight.dtype

        inactive_latents = inputs['inactive_latents'].to(device=device, dtype=model_dtype)
        reactive_latents = inputs['reactive_latents'].to(device=device, dtype=model_dtype)
        text_embeddings = inputs['text_embeddings']
        if isinstance(text_embeddings, list):
            text_embeddings = [emb.to(device) for emb in text_embeddings]
        else:
            text_embeddings = text_embeddings.to(device)
        seq_lens = inputs['seq_lens'].to(device)
        mask = inputs['mask']

        bs, channels, num_frames, h, w = inactive_latents.shape

        # Handle mask tensor of any dimensions
        # convert to shape (bs, 1, num_frames, h_latent, w_latent)
        if mask is not None:
            mask = mask.to(device)
            # Possible shapes:
            #   (bs, H, W)                    -> image mask (no channel, no frames)
            #   (bs, 1, H, W)                -> image mask with channel dim
            #   (bs, F, H, W)                -> video mask without channel dim
            #   (bs, 1, F, H, W)             -> fully-specified video mask
            if mask.ndim == 3:
                # (bs, H, W)  -> add channel & frame dims
                mask = mask.unsqueeze(1).unsqueeze(2)  # (bs,1,1,H,W)
            elif mask.ndim == 4:
                # Could be (bs,1,H,W) or (bs,F,H,W)
                if mask.shape[1] == 1:
                    # (bs,1,H,W) -> add frame dim
                    mask = mask.unsqueeze(2)  # (bs,1,1,H,W)
                else:
                    # (bs,F,H,W) -> add channel dim
                    mask = mask.unsqueeze(1)  # (bs,1,F,H,W)
            # At this point, mask is 5-D (bs,1,F?,H,W)
            if mask.ndim != 5:
                raise ValueError(f"Unsupported mask shape {mask.shape}, expected 2-5 dims.")
            # If temporal dimension is 1 but we need num_frames>1, expand without copy
            if mask.shape[2] == 1 and num_frames > 1:
                mask = mask.expand(-1, -1, num_frames, -1, -1)

            # Downsample/upsample to latent resolution (num_frames,h,w)
            interpolated_mask = F.interpolate(
                mask,
                size=(num_frames, h, w),
                mode='nearest-exact',
            )

            single_channel_mask = interpolated_mask

            try:
                # Try the original VACE processing
                mask_processed = self.process_vace_mask(single_channel_mask, vae_stride=(1, 8, 8), device=device, dtype=model_dtype)
            except Exception as e:
                print(f"Warning: Original VACE mask processing failed, falling back to repeat method: {e}")
                # Fallback to the working repeat method
                mask_processed = single_channel_mask.repeat(1, 64, 1, 1, 1).to(dtype=model_dtype)

            # loss
            target_mask = single_channel_mask.to(dtype=model_dtype)
        else:
            mask_processed = torch.zeros(bs, 64, num_frames, h, w, device=device, dtype=model_dtype)
            target_mask = None

        
        vace_context = torch.cat([inactive_latents, reactive_latents, mask_processed], dim=1)

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')
        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            x_1 = inactive_latents.float() + reactive_latents.float()
            x_0 = torch.randn_like(x_1)
            t_expanded = t.view(-1, 1, 1, 1, 1).to(x_1.dtype)
            x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
            target = x_0 - x_1
        
        x_t = x_t.to(dtype=model_dtype)

        t = t * 1000
        
        for item in [x_t, t, vace_context, text_embeddings, seq_lens, torch.tensor([1.0], device=device, dtype=model_dtype), torch.empty(0, device=device, dtype=model_dtype), x_1]:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        model_inputs = (
            x_t, 
            t, 
            vace_context, 
            text_embeddings, 
            seq_lens, 
            torch.tensor([1.0], device=device, dtype=model_dtype), # context_scale
            torch.empty(0, device=device, dtype=model_dtype), # clip_fea
            x_1, # y
        )
        
        labels = (target, target_mask)
        return model_inputs, labels

    def to_layers(self):
        transformer = self.transformer
        transformer.offloader = self.offloader
        layers = [
            InitialLayer(transformer),
            VaceHintGenerationLayer(transformer, self.offloader)
        ]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        raise NotImplementedError("Block swapping is not supported for VACE models.")

    def prepare_block_swap_training(self):
        raise NotImplementedError("Block swapping is not supported for VACE models.")

    def prepare_block_swap_inference(self, disable_block_swap=False):
        raise NotImplementedError("Block swapping is not supported for VACE models.")


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        x, t, vace_context, context, seq_lens, context_scale, clip_fea, y = inputs
        
        device = self.model.patch_embedding.weight.device
        
        if self.model.freqs.device != device:
            self.model.freqs = self.model.freqs.to(device)

        # Patch embedding
        bs, _, num_frames, h, w = x.shape
        x = self.model.patch_embedding(x)
        
        # Positional embeddings
        grid_sizes = torch.tensor([[num_frames, h // self.model.patch_size[1], w // self.model.patch_size[2]]], device=device).repeat(bs, 1)
        x = x.flatten(2).transpose(1, 2)
        
        # Time embeddings
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.model.time_embedding(sinusoidal_embedding_1d(self.model.freq_dim, t).float())
            e0 = self.model.time_projection(e).unflatten(1, (6, self.model.dim))

        # Project text embeddings from T5 dim (4096) to model dim (1536)
        context = self.model.text_embedding(context)

        return x, e, e0, seq_lens, grid_sizes, self.model.freqs, context, vace_context, context_scale


class VaceHintGenerationLayer(nn.Module):
    def __init__(self, model, offloader):
        super().__init__()
        self.model = model
        self.offloader = offloader

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context, vace_context, context_scale = inputs

        # This logic is from VaceWanModel.forward_vace and VaceWanModel.forward
        
        # VACE context embedding
        c = self.model.vace_patch_embedding(vace_context)
        c = c.flatten(2).transpose(1, 2)

        # Prepare arguments for VACE blocks (VaceWanAttentionBlock)
        kwargs = {
            'x': x,
            'e': e0,
            'seq_lens': seq_lens,
            'grid_sizes': grid_sizes,
            'freqs': freqs,
            'context': context,
            'context_lens': None, # None in the original implementation
        }

        # Run VACE blocks to generate hints
        hints = []
        for block in self.model.vace_blocks:
            c, c_skip = block(c, **kwargs)
            hints.append(c_skip)
        # Stack hints into a single tensor of shape (num_hints, bs, seq_len, dim)
        hints_tensor = torch.stack(hints, dim=0)

        return x, e, e0, seq_lens, grid_sizes, freqs, context, hints_tensor, context_scale


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context, hints, context_scale = inputs
        # The WanAttentionBlock expects the *projected* time embedding e0
        # and context_lens, which is default None
        x = self.block(
            x, e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs,
            context=context, hints=hints, context_scale=context_scale,
            context_lens=None,
        )
        return x, e, e0, seq_lens, grid_sizes, freqs, context, hints, context_scale


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context, hints, context_scale = inputs
        # The head layer uses the *unprojected* time embedding `e`.
        x = self.model.head(x, e)
        # Reshape the output from patches back to the original video shape
        x = self.model.unpatchify(x, grid_sizes)
        # Stack the list of tensors into a single batched tensor
        x = torch.stack(x, dim=0)
        return x
