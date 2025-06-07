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
    parent_kwargs = kwargs.copy()
    parent_kwargs.pop('x', None)
    parent_kwargs.pop('e0', None)
    c = super(VaceWanAttentionBlock, self).forward(c, **parent_kwargs)
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
    parent_kwargs = kwargs.copy()
    parent_kwargs.pop('e0', None)
    parent_kwargs.pop('hints', None)
    parent_kwargs.pop('context_scale', None)
    
    x = super(BaseWanAttentionBlock, self).forward(x, e=e, **parent_kwargs)
    
    if self.block_id is not None:
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

    def load_diffusion_model(self):
        model_config = self.model_config
        ckpt_path = Path(model_config['ckpt_path'])
        dtype = getattr(torch, model_config['dtype'])
        transformer_dtype = getattr(torch, model_config.get('transformer_dtype', model_config['dtype']))

        with open(ckpt_path / 'config.json', "r", encoding="utf-8") as f:
            config = json.load(f)

        model_dim = config.get('dim', 0)
        if model_dim == 1536:  # 1.3B model
            correct_params = {'ffn_dim': 8960, 'num_heads': 24, 'num_layers': 16, 'vace_in_dim': 96}
        elif model_dim == 5120:  # 14B model
            correct_params = {'ffn_dim': 20480, 'num_heads': 40, 'num_layers': 64, 'vace_in_dim': 96}
        else:
            raise ValueError(f"Unsupported or missing model dimension in config.json: {model_dim}")
        
        config.update(correct_params)
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
            dtype_to_use = dtype if 'norm' in name or 'bias' in name else transformer_dtype
            set_module_tensor_to_device(model, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        model = model.cuda().eval().requires_grad_(False)
        
        for name, p in model.named_parameters():
            p.original_name = name

        return model

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

                inactive_latents = vae_encode(inactive_tensor, self.vae)
                reactive_latents = vae_encode(reactive_tensor, self.vae)
                
                latents = inactive_latents + reactive_latents
                return {'latents': latents, 'inactive_latents': inactive_latents, 'reactive_latents': reactive_latents}
            else:
                latents = vae_encode(tensor, self.vae)
                inactive_latents = latents
                reactive_latents = torch.zeros_like(latents)
                return {'latents': latents, 'inactive_latents': inactive_latents, 'reactive_latents': reactive_latents}
        return fn
        
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
        mask = inputs['mask'].to(device) if inputs['mask'] is not None else None

        bs, channels, num_frames, h, w = inactive_latents.shape

        if mask is not None and mask.ndim > 2:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')
            mask_processed = mask.repeat(1, 64, num_frames, 1, 1).to(dtype=model_dtype)
        else:
            mask_processed = torch.zeros(bs, 64, num_frames, h, w, device=device, dtype=model_dtype)
        
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

        x_1 = inactive_latents + reactive_latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1
        t = t * 1000

        return (
            (x_t, t, vace_context, text_embeddings, seq_lens, 1.0, None, None),
            (target, mask.unsqueeze(2).to(dtype=model_dtype) if mask is not None else None),
        )

    def to_layers(self):
        transformer = self.transformer
        layers = [
            InitialLayer(transformer),
            VaceHintGenerationLayer(transformer, self.offloader)
        ]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
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

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, x, t, vace_context, context, seq_lens, context_scale, clip_fea, y):
        bs = x.shape[0]
        context = [emb[:length] for emb, length in zip(context, seq_lens)] if context is not None else None

        device = self.patch_embedding.weight.device
        model_dtype = self.patch_embedding.weight.dtype
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0).to(dtype=model_dtype)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_len = max([u.size(1) for u in x])
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_float = t.to(device=device, dtype=torch.float32)
            sinusoidal_output = sinusoidal_embedding_1d(self.freq_dim, t_float)
            e = self.time_embedding(sinusoidal_output.to(torch.float32))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        e0 = e0.to(dtype=model_dtype)

        # context
        if context is not None:
            context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        
        vace_context = vace_context.to(device=device, dtype=model_dtype)
        
        return (x, e, e0, seq_lens, grid_sizes, self.freqs, context, vace_context, context_scale)

# The VACE architecture introduces a separate "hint generation" pipeline that runs
# in parallel to the main denoising U-Net. The VaceHintGenerationLayer is my implementation of that pipeline.
# Instead of  concatenating the VACE context (inactive latents, reactive
# latents, and mask) to the main input, the VACE model processes it separately
# to create "hint" tensors. 
class VaceHintGenerationLayer(nn.Module):
    def __init__(self, model, offloader):
        super().__init__()
        self.vace_patch_embedding = model.vace_patch_embedding
        self.vace_blocks = model.vace_blocks
        self.offloader = offloader
        self.model = [model]
    
    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, x, e, e0, seq_lens, grid_sizes, freqs, context, vace_context, context_scale):
        vace_list = [v for v in vace_context]
        seq_len = x.size(1)

        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_list]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in c
        ])

        new_kwargs = dict(
            x=x, e=e, e0=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs,
            context=context, context_lens=None
        )

        hints = []
        for vace_block_idx, block in enumerate(self.vace_blocks):
            self.offloader.wait_for_block(vace_block_idx)
            c, c_skip = block(c, **new_kwargs)
            hints.append(c_skip)
            self.offloader.submit_move_blocks_forward(vace_block_idx)
        
        return (x, e, e0, seq_lens, grid_sizes, freqs, context, hints, context_scale)

class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, x, e, e0, seq_lens, grid_sizes, freqs, context, hints, context_scale):
        self.offloader.wait_for_block(self.block_idx)
        # The BaseWanAttentionBlock uses the hints list and its own block_id to get the correct hint
        x = self.block(x, e=e, e0=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs, context=context, context_lens=None, hints=hints, context_scale=context_scale)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return (x, e, e0, seq_lens, grid_sizes, freqs, context, hints, context_scale)

class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, x, e, e0, seq_lens, grid_sizes, freqs, context, hints, context_scale):
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x, dim=0)
