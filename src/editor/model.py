import torch
import torch.nn as nn
import os
import numpy as np
import xformers.ops
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.models.layers import DropPath


"""
Data = [{
    "image_name" : .. ,
    "input_image_slots: ... , 
    "output_image_slots": ... ,
    "edit_prompt": ... ,
    "prompt_embedding": ...
}]
"""


class CaptionEmbedder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=300,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )

    def forward(self, caption):
        return self.y_proj(caption)




class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, **block_kwargs
    ):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        B, N, C = x.shape
        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(
            q, k, v, p=self.attn_drop.p, attn_bias=attn_bias
        )
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SlotEditor_Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, y, mask=None):
        B, N, C = x.shape

        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.mlp(self.norm2(x))
        return x


class SlotEditor(nn.Module):
    """
    SLOTS : Bx9x64
    TEXT_EMB: 1x300x4096

    """

    def __init__(
        self,
        hidden_size=512,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        model_max_length=300,
        caption_channels=4096,
        slot_dim=64
    ):
        super().__init__()
        self.num_heads = num_heads
        self.depth = depth

        # add positional embedding : don't do it to maintain multiset order of slots.
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.x_embedder = Mlp(
            in_features=slot_dim,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=approx_gelu,
            drop=0,
        )

        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        self.blocks = nn.ModuleList(
            [
                SlotEditor_Block(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.final_layer = nn.Linear(hidden_size, slot_dim, bias=True)

        self.initialize_weights()

    def forward(self, x, y, mask=None):
        x = x.to(self.dtype)
        y = y.to(self.dtype)

        x = self.x_embedder(x)
        y = self.y_embedder(y)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = (
                y.squeeze(1)
                .masked_select(mask.unsqueeze(-1) != 0)
                .view(1, -1, x.shape[-1])
            )
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        for block in self.blocks:
            x = block(x, y, y_lens)
        x = self.final_layer(x)
        return x

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
