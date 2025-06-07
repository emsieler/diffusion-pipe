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

# Patch to remove forced casting to float32, saving memory.
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
        super().__init__(config)
        self.transformer = self.load_diffusion_model()

    def load_diffusion_model(self):
        model_config = self.model_config
        ckpt_path = Path(model_config['ckpt_path'])
        dtype = getattr(torch, model_config['dtype'])
        transformer_dtype = getattr(torch, model_config.get('transformer_dtype', model_config['dtype']))

        # Load model
        if transformer_path := model_config.get('transformer_path', None):
            model = VaceWanModelFromSafetensors.from_pretrained(
                transformer_path,
                os.path.join(ckpt_path, 'config.json'),
                torch_dtype=dtype,
                transformer_dtype=transformer_dtype,
            )
        else:
            # Multi-part safetensors loading
            with init_empty_weights():
                model = VaceWanModel.from_config(ckpt_path / 'config.json')
            
            state_dict = {}
            for shard in ckpt_path.glob('*.safetensors'):
                print(f"Loading shard: {shard}")
                with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            
            for name, param in model.named_parameters():
                dtype_to_use = dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
                set_module_tensor_to_device(model, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        model = model.cuda().eval().requires_grad_(False)
        
        # Store original parameter names
        for name, p in model.named_parameters():
            p.original_name = name

        return model

    def get_text_encoders(self):
        if not next(self.text_encoder.model.parameters()).is_cuda:
            self.text_encoder.model = self.text_encoder.model.cuda()
        return [self.text_encoder.model]

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            device = next(text_encoder.parameters()).device
            ids, mask = self.text_encoder.tokenizer(caption, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            with torch.autocast(device_type=device.type, dtype=text_encoder.dtype):
                text_embeddings = text_encoder(ids, mask)
                return {'text_embeddings': text_embeddings, 'seq_lens': seq_lens}
        return fn

    # override: VACE does not use clip
    def get_vae(self):
        if not next(self.vae.model.parameters()).is_cuda:
            self.vae.model = self.vae.model.cuda()
        return self.vae.model

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            if not next(self.vae.model.parameters()).is_cuda:
                self.vae.model = self.vae.model.cuda()
            if not self.vae.scale[0].is_cuda:
                self.vae.scale = [s.cuda() for s in self.vae.scale]
            tensor = tensor.cuda()
            latents = vae_encode(tensor, self.vae)
            return {'latents': latents}
        return fn
        
    def prepare_inputs(self, inputs, timestep_quantile=None):
        device = self.transformer.patch_embedding.weight.device
        model_dtype = self.transformer.patch_embedding.weight.dtype

        latents = inputs['latents'].float().to(device)
        text_embeddings = inputs['text_embeddings']
        if isinstance(text_embeddings, list):
            text_embeddings = [emb.to(device) for emb in text_embeddings]
        else:
            text_embeddings = text_embeddings.to(device)
        seq_lens = inputs['seq_lens'].to(device)
        mask = inputs['mask'].to(device) if inputs['mask'] is not None else None

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
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

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

        # Initial embeddings for main input - set dtype to match model
        x = [self.transformer.patch_embedding(u.unsqueeze(0).to(dtype=model_dtype)) for u in x_t]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_len = max([u.size(1) for u in x])
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # Time embeddings - match model dtype
        t_model_dtype = t.to(dtype=model_dtype)
        e = self.transformer.time_embedding(sinusoidal_embedding_1d(self.transformer.freq_dim, t_model_dtype).to(device=device, dtype=model_dtype))
        e0 = self.transformer.time_projection(e).unflatten(1, (6, self.transformer.dim))
        assert e.dtype == model_dtype and e0.dtype == model_dtype, f"Time embeddings not in {model_dtype}: e={e.dtype}, e0={e0.dtype}"

        # Text embeddings - set dtype to match model
        context = [emb[:length].to(dtype=model_dtype) for emb, length in zip(text_embeddings, seq_lens)]
        context = self.transformer.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.transformer.text_len - u.size(0), u.size(1), dtype=model_dtype)])
                for u in context
            ]).to(device))

        # VACE context from latents and masks
        vace_list_for_forward_vace = []
        print("\nConstructing VACE context list:")
        print(f"Shape of x_t (input latents): {x_t.shape}")

        x_t = x_t.to(device=device, dtype=model_dtype)
        x_t_permuted = x_t.permute(0, 2, 1, 3)
        print(f"Shape of x_t_permuted: {x_t_permuted.shape}")

        for i in range(x_t_permuted.shape[0]):
            item_slice = x_t_permuted[i]
            print(f"  Processing item {i} for VACE context list: original slice shape: {item_slice.shape}")
            item_slice_5d = item_slice.unsqueeze(-1)
            print(f"  Item {i} after adding W dim: {item_slice_5d.shape}")
            vace_list_for_forward_vace.append(item_slice_5d.to(device=device, dtype=model_dtype))

        print("\nFinal vace_context (list of tensors to be passed to forward_vace):")
        if not vace_list_for_forward_vace:
            print("  List is empty.")
        for idx, tensor_item in enumerate(vace_list_for_forward_vace):
            print(f"  Item {idx} shape: {tensor_item.shape}, Dims: {tensor_item.dim()}")
        
        # Generate hints using forward_vace
        vace_block_args = dict(
            x=x.to(device=device),
            e=e0.to(device=device),
            seq_lens=seq_lens.to(device=device),
            grid_sizes=grid_sizes.to(device=device),
            freqs=self.transformer.freqs.to(device=device),
            context=context.to(device=device) if context is not None else None,
            context_lens=None
        )
        hints = self.transformer.forward_vace(x, vace_list_for_forward_vace, seq_len, vace_block_args)

        x_t = x_t.to(device=device, dtype=model_dtype)
        t = t.to(device=device, dtype=model_dtype)
        target = target.to(device=device, dtype=model_dtype)
        if mask is not None:
            mask = mask.to(device=device, dtype=model_dtype)

        return (
            (x_t, t, hints, text_embeddings, seq_lens, 1.0, None, None),
            (target, mask),
        )

    def to_layers(self):
        import pdb; pdb.set_trace()  # Breakpoint 1: Check layer creation
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        
        # Process blocks in sequence, (Vace blocks come before corresponding Base blocks to provide context)
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

        # embeddings -  make sure correct dtype
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
