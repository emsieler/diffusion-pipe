import sys
import json
import math
import re
import os.path
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
)

import wan
from wan.modules.t5 import T5Encoder, T5Decoder, T5Model
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae import WanVAE
from wan.modules.model import (
    WanModel, VaceWanModel, sinusoidal_embedding_1d, WanLayerNorm, WanSelfAttention, WAN_CROSSATTENTION_CLASSES
)
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
        return

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

        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    # no changes made to class methods:
    # def __getattr__(self, name):
    # get_text_encoders(self):
    # def save_adapter(self, save_dir, peft_state_dict):
    # def save_model(self, save_dir, diffusers_sd):
    # def get_preprocess_media_file_fn(self):
    # def get_call_text_encoder_fn(self, text_encoder):

    # override: VACE does not use clip
    def get_vae(self):
        vae = self.vae.model
        return vae 

    def get_call_vae_fn(self, vae_and_clip):
        vae = self.get_vae()
        p = next(vae.parameters())
        tensor = tensor.to(p.device, p.dtype)
        latents = vae_encode(tensor, self.vae)
        return {'latents': latents}
        
    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        text_embeddings = inputs['text_embeddings']
        seq_lens = inputs['seq_lens']
        mask = inputs['mask']

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

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

        # Create VACE context
        latents_list = [x for x in x_t]
        masks_list = [m for m in mask] if mask is not None else None
        z0 = self.transformer.vace_encode_frames(latents_list, None, masks_list)
        m0 = self.transformer.vace_encode_masks(masks_list if masks_list else [None] * len(latents_list))
        vace_context = self.transformer.vace_latent(z0, m0)

        return (
            (x_t, t, vace_context, text_embeddings, seq_lens, 1.0, None, None),
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

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
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

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens.max()
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(x.device, torch.float32))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

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

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        import pdb; pdb.set_trace()  # Breakpoint 3: Check Base layer inputs/outputs
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

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        import pdb; pdb.set_trace()  # Breakpoint 2: Check Vace layer inputs/outputs
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

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context = inputs
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x, dim=0)

if __name__ == "__main__":

