from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from .diffusion_utils import checkpoint

from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.lora import LoRALinearLayer

# try:
#     import xformers
#     import xformers.ops
#     XFORMERS_IS_AVAILBLE = True
# except:
#     XFORMERS_IS_AVAILBLE = False

XFORMERS_IS_AVAILBLE = False


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.inner_dim = inner_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        processor = CrossAttnProcessor()
        self.set_processor(processor)

    def forward(self, x, context=None, mask=None):
        
        return self.processor(
            attn=self,
            hidden_states=x,
            encoder_hidden_states=context,
            attention_mask=mask,
        )

    def forward_next(self, x, context=None, mask=None):
        assert mask is None, 'not supported yet'
        x0 = rearrange(x, 'b n c -> n b c')
        if context is not None:
            c0 = rearrange(context, 'b n c -> n b c')
        else:
            c0 = x0
        r, _ = F.multi_head_attention_forward(
            x0, c0, c0,
            embed_dim_to_check = self.inner_dim,
            num_heads = self.heads,
            in_proj_weight = None, in_proj_bias = None,
            bias_k = None, bias_v = None,
            add_zero_attn = False, dropout_p = 0,
            out_proj_weight = self.to_out[0].weight,
            out_proj_bias = self.to_out[0].bias,
            use_separate_proj_weight = True,
            q_proj_weight = self.to_q.weight,
            k_proj_weight = self.to_k.weight,
            v_proj_weight = self.to_v.weight,)
        r = rearrange(r, 'n b c -> b n c')
        r = self.to_out[1](r) # dropout
        return r

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor, out_dim=3):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

        return tensor
    
    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs
    
    def prepare_attention_mask(self, attention_mask, target_length, batch_size, out_dim=3):
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask
    
    def set_processor(self, processor: "CrossAttnProcessor"):
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (hasattr(self, "processor") and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)):
            self._modules.pop("processor")

        self.processor = processor 

class LoRAAttnProcessor(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, network_alpha=None, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        q_rank = kwargs.pop("q_rank", None)
        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_rank = q_rank if q_rank is not None else rank
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size

        v_rank = kwargs.pop("v_rank", None)
        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_rank = v_rank if v_rank is not None else rank
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size

        out_rank = kwargs.pop("out_rank", None)
        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_rank = out_rank if out_rank is not None else rank
        out_hidden_size = out_hidden_size if out_hidden_size is not None else hidden_size

        self.to_q_lora = LoRALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(out_hidden_size, out_hidden_size, out_rank, network_alpha)

    def __call__(
        self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None
    ):
        
        h = attn.heads

        q = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        encoder_hidden_states = default(encoder_hidden_states, hidden_states)
        k = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * attn.scale

        if exists(attention_mask):
            attention_mask = rearrange(attention_mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            attention_mask = repeat(attention_mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~attention_mask, max_neg_value)

        # attention, what we cannot get enough of
        attention = sim.softmax(dim=-1)
        
        out = einsum('b i j, b j d -> b i d', attention, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = attn.to_out[0](out) + scale * self.to_out_lora(hidden_states)
        out = attn.to_out[1](out)
        return out
    
class CrossAttnProcessor:
    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        h = attn.heads

        q = attn.to_q(hidden_states)
        encoder_hidden_states = default(encoder_hidden_states, hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * attn.scale

        if exists(attention_mask):
            attention_mask = rearrange(attention_mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            attention_mask = repeat(attention_mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~attention_mask, max_neg_value)

        # attention, what we cannot get enough of
        attention = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attention, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return attn.to_out(out)
    
class AttentionMask:
    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError()
    
class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None
        
        processor = AttnProcessor()
        self.set_processor(processor)

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        out = out.to(x.dtype) # toothable debug the float16 and float32 error
        return self.to_out(out)
    
    def set_processor(self, processor: "AttnProcessor"):
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (hasattr(self, "processor") and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)):
            self._modules.pop("processor")

        self.processor = processor
    
    def memory_efficient_attention_substitute(
                                            self,
                                            query: torch.Tensor,
                                            key: torch.Tensor,
                                            value: torch.Tensor,
                                            attn_bias: Optional[Union[torch.Tensor, AttentionMask]] = None,
                                            p: float = 0.0,
                                            *,
                                            op=None,):
        pass
        


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


##########################
# transformer no context #
##########################

class BasicTransformerBlockNoContext(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, 
                                    dropout=dropout, context_dim=None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, 
                                    dropout=dropout, context_dim=None)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.checkpoint)

    def _forward(self, x):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x)) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformerNoContext(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0.,):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlockNoContext(inner_dim, n_heads, d_head, dropout=dropout)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in


#######################################
# Spatial Transformer with Two Branch #
#######################################

class DualSpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        # First crossattn
        self.norm_0 = Normalize(in_channels)
        self.proj_in_0 = nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks_0 = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   disable_self_attn=disable_self_attn)
                for d in range(depth)]
        )
        self.proj_out_0 = zero_module(nn.Conv2d(
            inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

        # Second crossattn
        self.norm_1 = Normalize(in_channels)
        self.proj_in_1 = nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks_1 = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   disable_self_attn=disable_self_attn)
                for d in range(depth)]
        )
        self.proj_out_1 = zero_module(nn.Conv2d(
            inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context=None, which=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        if which==0:
            norm, proj_in, blocks, proj_out = \
                self.norm_0, self.proj_in_0, self.transformer_blocks_0, self.proj_out_0
        elif which==1:
            norm, proj_in, blocks, proj_out = \
                self.norm_1, self.proj_in_1, self.transformer_blocks_1, self.proj_out_1
        else:
            # assert False, 'DualSpatialTransformer forward with a invalid which branch!'
            # import numpy.random as npr
            # rwhich = 0 if npr.rand() < which else 1
            # context = context[rwhich]
            # if rwhich==0:
            #     norm, proj_in, blocks, proj_out = \
            #         self.norm_0, self.proj_in_0, self.transformer_blocks_0, self.proj_out_0
            # elif rwhich==1:
            #     norm, proj_in, blocks, proj_out = \
            #         self.norm_1, self.proj_in_1, self.transformer_blocks_1, self.proj_out_1

            # import numpy.random as npr
            # rwhich = 0 if npr.rand() < 0.33 else 1
            # if rwhich==0:
            #     context = context[rwhich]
            #     norm, proj_in, blocks, proj_out = \
            #         self.norm_0, self.proj_in_0, self.transformer_blocks_0, self.proj_out_0
            # else:

            norm, proj_in, blocks, proj_out = \
                self.norm_0, self.proj_in_0, self.transformer_blocks_0, self.proj_out_0
            x0 = norm(x)
            x0 = proj_in(x0)
            x0 = rearrange(x0, 'b c h w -> b (h w) c').contiguous()
            for block in blocks:
                x0 = block(x0, context=context[0])
            x0 = rearrange(x0, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            x0 = proj_out(x0)

            norm, proj_in, blocks, proj_out = \
                self.norm_1, self.proj_in_1, self.transformer_blocks_1, self.proj_out_1
            x1 = norm(x)
            x1 = proj_in(x1)
            x1 = rearrange(x1, 'b c h w -> b (h w) c').contiguous()
            for block in blocks:
                x1 = block(x1, context=context[1])
            x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            x1 = proj_out(x1)
            return x0*which + x1*(1-which) + x_in

        x = norm(x)
        x = proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = proj_out(x)
        return x + x_in
