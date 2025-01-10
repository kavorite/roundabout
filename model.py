# adapted from https://github.com/lucidrains/nGPT-pytorch/blob/main/nGPT_pytorch/nGPT.py
from functools import partial

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from einops.layers.torch import Rearrange
from fla.ops.delta_rule import chunk_delta_rule
from torch import nn
from torch.nn import Module

# constants
from torch.nn.utils.parametrize import register_parametrization

# functions


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def cast_tuple(t, length=1):
    out = t if isinstance(t, tuple) else ((t,) * length)
    assert len(out) == length
    return out


def l2norm(t, dim=-1, norm_eps=1e-5, eps=None, groups=1):
    if groups > 1:
        t = t.chunk(groups, dim=dim)
        t = torch.stack(t)

    if norm_eps == 0.0:
        out = F.normalize(t, dim=dim, p=2)
    else:
        eps = default(
            eps, 1e-6 if t.dtype in (torch.bfloat16, torch.float16) else 1e-10
        )
        norm = t.norm(dim=dim, keepdim=True)
        target_norm = norm.detach().clamp(min=1.0 - norm_eps, max=1.0 + norm_eps)
        divisor = norm / target_norm
        out = t / divisor.clamp(min=eps)

    if groups > 1:
        out = torch.cat([*out], dim=dim)

    return out


# scale


class Scale(Module):
    """
    latter part of section 2.5 in the paper
    """

    def __init__(self, dim, init=1.0, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * scale)
        self.forward_scale = init / scale

    def forward(self):
        return self.scale * self.forward_scale


# residual slerp update with learned scale


class Residual(Module):
    def __init__(
        self,
        fn: Module,
        dim: int,
        init: float,
        scale: float | None = None,
        groups=1,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.fn = fn
        self.branch_scale = Scale(dim, init, default(scale, dim**-0.5))
        self.l2norm = L2Norm(dim=-1, norm_eps=norm_eps, groups=groups)

    def forward(self, x, **kwargs):
        residual = x

        out = self.fn(x, **kwargs)

        tuple_output = isinstance(out, tuple)

        if tuple_output:
            out, *rest = out

        out = self.l2norm(out)
        out = self.l2norm(residual.lerp(out, self.branch_scale()))

        if tuple_output:
            out = (out, *rest)

        return out


# for use with parametrize


class L2Norm(Module):
    def __init__(self, dim=-1, norm_eps=1e-5, groups=1):
        super().__init__()
        self.dim = dim
        self.norm_eps = norm_eps
        self.groups = groups

    def forward(self, t):
        return l2norm(t, dim=self.dim, norm_eps=self.norm_eps, groups=self.groups)


class NormLinear(Module):
    def __init__(
        self, dim, dim_out, norm_dim_in=True, parametrize=True, norm_eps=1e-5, groups=1
    ):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias=False)

        self.scale = groups**-1
        self.parametrize = parametrize
        self.l2norm = L2Norm(
            dim=-1 if norm_dim_in else 0, norm_eps=norm_eps, groups=groups
        )

        if parametrize:
            register_parametrization(self.linear, "weight", self.l2norm)

        self.norm_weights_()

    @torch.no_grad()
    def norm_weights_(self):
        if self.parametrize:
            normed = self.weight
            original = self.linear.parametrizations.weight.original

            original.copy_(normed)
        else:
            self.weight.copy_(self.l2norm(self.weight))

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x) * self.scale


# attention


class Attention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        norm_qk=True,
        causal=True,
        manual_norm_weights=False,
        s_qk_init=1.0,
        s_qk_scale=None,
        flash_kwargs: dict = dict(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ),
        norm_eps=1e-5,
        num_hyperspheres=1,
        chunk_size=64,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.heads = heads
        self.causal = causal
        self.norm_qk = norm_qk

        NormLinear_ = partial(
            NormLinear,
            parametrize=not manual_norm_weights,
            norm_eps=norm_eps,
            groups=num_hyperspheres,
        )
        self.l2norm = partial(l2norm, norm_eps=norm_eps, groups=num_hyperspheres)

        dim_inner = dim_head * heads

        self.to_q = NormLinear_(dim, dim_inner)
        self.to_k = NormLinear_(dim, dim_inner)
        self.to_v = NormLinear_(dim, dim_inner)

        self.to_beta = NormLinear_(dim_head, 1)
        self.beta_scale = Scale(1, init=0.1)

        self.qk_scale = Scale(dim_inner, s_qk_init, default(s_qk_scale, dim**-1))

        self.split_heads = Rearrange("b n (h d) -> b h n d", h=heads)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")

        self.to_out = NormLinear_(dim_inner, dim, norm_dim_in=False)

    def forward(
        self,
        x,
        offsets,
        value_residual=None,
        return_values=False,
        return_states=False,
        prev_states=None,
    ):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(self.split_heads, (q, k, v))

        if exists(value_residual):
            v = 0.5 * (v + value_residual)

        if self.norm_qk:
            q, k = map(self.l2norm, (q, k))

        betas = torch.sigmoid(self.to_beta(k)[..., 0] * self.beta_scale())

        q = q * rearrange(self.qk_scale(), "(h d) -> h 1 d", h=self.heads)

        bsize = q.shape[0]
        q, k, v = map(
            lambda x: rearrange(x, "b h n d -> 1 (b n) h d", b=bsize), (q, k, v)
        )

        offsets = (offsets * bsize).to(device=q.device, dtype=torch.long)

        out, states = chunk_delta_rule(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            beta=betas.contiguous(),
            offsets=offsets.contiguous(),
            head_first=False,
            initial_state=prev_states,
            output_final_state=return_states,
        )

        out = rearrange(out, "1 (b n) h d -> b h n d", b=bsize)
        out = self.merge_heads(out)
        out = self.to_out(out)

        if return_values:
            v = rearrange(v, "1 (b n) h d -> b h n d", b=bsize)
        else:
            v = None
        if return_states:
            states = rearrange(states, "1 (b n) h d -> b h n d", b=bsize)
        else:
            states = None

        return out, v, states


# feedforward


class FeedForward(Module):
    def __init__(
        self,
        dim,
        *,
        dim_out=None,
        expand_factor=4,
        manual_norm_weights=False,
        s_hidden_init=1.0,
        s_hidden_scale=1.0,
        s_gate_init=1.0,
        s_gate_scale=1.0,
        norm_eps=1e-5,
        num_hyperspheres=1,
    ):
        super().__init__()
        NormLinear_ = partial(
            NormLinear,
            parametrize=not manual_norm_weights,
            norm_eps=norm_eps,
            groups=num_hyperspheres,
        )

        self.dim = dim
        dim_inner = int(dim * expand_factor * 2 / 3)

        self.to_hidden = NormLinear_(dim, dim_inner)
        self.to_gate = NormLinear_(dim, dim_inner)

        self.hidden_scale = Scale(dim_inner, s_hidden_init, s_hidden_scale)
        self.gate_scale = Scale(dim_inner, s_gate_init, s_gate_scale)

        self.to_out = NormLinear_(dim_inner, default(dim_out, dim), norm_dim_in=False)

    def forward(self, x):
        hidden, gate = self.to_hidden(x), self.to_gate(x)

        hidden = hidden * self.hidden_scale()
        gate = gate * self.gate_scale() * (self.dim**0.5)

        hidden = F.silu(gate) * hidden
        return self.to_out(hidden)


# classes


class nGPT(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        token_embed=None,
        dim_head=64,
        heads=8,
        attn_norm_qk=True,  # they say the query/key normalization is optional
        ff_expand_factor=4.0,
        ce_ignore_index=-1,
        manual_norm_weights=False,
        tied_embedding=False,
        num_hyperspheres=1,
        causal=True,
        add_value_residual=True,
        # below are all the scale related hyperparameters, for controlling effective relative learning rates throughout the network
        alpha_init: float
        | None = None,  # this would set the alpha init for all residuals, but would be overridden by alpha_attn_init and alpha_ff_init if they are specified
        s_logit_init: float = 1.0,
        s_logit_scale: float | None = None,
        alpha_attn_init: float | tuple[float, ...] | None = None,
        alpha_attn_scale: float | tuple[float, ...] | None = None,
        alpha_ff_init: float | tuple[float, ...] | None = None,
        alpha_ff_scale: float | tuple[float, ...] | None = None,
        s_qk_init: float | tuple[float, ...] = 1.0,
        s_qk_scale: float | tuple[float, ...] | None = None,
        s_ff_hidden_init: float | tuple[float, ...] = 1.0,
        s_ff_hidden_scale: float | tuple[float, ...] = 1.0,
        s_ff_gate_init: float | tuple[float, ...] = 1.0,
        s_ff_gate_scale: float | tuple[float, ...] = 1.0,
        attn_flash_kwargs: dict = dict(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ),
        norm_eps=1e-5,  # greater than 0 allows the norm to be around (1. - norm_eps) to (1. + norm_eps)
        chunk_size=64,
    ):
        super().__init__()
        NormLinear_ = partial(
            NormLinear,
            parametrize=not manual_norm_weights,
            norm_eps=norm_eps,
            groups=num_hyperspheres,
        )
        self.l2norm = partial(l2norm, norm_eps=norm_eps, groups=num_hyperspheres)

        self.dim = dim
        self.causal = causal
        alpha_init = default(alpha_init, 1.0 / depth)

        self.add_value_residual = (
            add_value_residual  # https://arxiv.org/abs/2410.17897v1
        )

        self.token_embed = default(token_embed, NormLinear_(dim, num_tokens))

        self.layers = nn.ModuleList([])

        scale_hparams = (
            alpha_attn_init,
            alpha_attn_scale,
            alpha_ff_init,
            alpha_ff_scale,
            s_qk_init,
            s_qk_scale,
            s_ff_hidden_init,
            s_ff_hidden_scale,
            s_ff_gate_init,
            s_ff_gate_scale,
        )

        scale_hparams = tuple(cast_tuple(hparam, depth) for hparam in scale_hparams)

        for (
            alpha_attn_init_,
            alpha_attn_scale_,
            alpha_ff_init_,
            alpha_ff_scale_,
            s_qk_init_,
            s_qk_scale_,
            s_ff_hidden_init_,
            s_ff_hidden_scale_,
            s_ff_gate_init_,
            s_ff_gate_scale_,
        ) in zip(*scale_hparams):
            attn = Attention(
                dim,
                dim_head=dim_head,
                heads=heads,
                causal=causal,
                norm_qk=attn_norm_qk,
                manual_norm_weights=manual_norm_weights,
                s_qk_init=s_qk_init_,
                s_qk_scale=s_qk_scale_,
                flash_kwargs=attn_flash_kwargs,
                norm_eps=norm_eps,
                num_hyperspheres=num_hyperspheres,
                chunk_size=chunk_size,
            )

            ff = FeedForward(
                dim,
                expand_factor=ff_expand_factor,
                manual_norm_weights=manual_norm_weights,
                s_hidden_init=s_ff_hidden_init_,
                s_hidden_scale=s_ff_hidden_scale_,
                s_gate_init=s_ff_gate_init_,
                s_gate_scale=s_ff_gate_scale_,
                norm_eps=norm_eps,
                num_hyperspheres=num_hyperspheres,
            )

            attn_with_residual = Residual(
                attn,
                dim,
                default(alpha_attn_init_, alpha_init),
                default(alpha_attn_scale_, dim**-0.5),
            )

            ff_with_residual = Residual(
                ff,
                dim,
                default(alpha_ff_init_, alpha_init),
                default(alpha_ff_scale_, dim**-0.5),
            )

            self.layers.append(nn.ModuleList([attn_with_residual, ff_with_residual]))

        self.to_logits = NormLinear_(dim, num_tokens) if not tied_embedding else None

        self.logit_scale = Scale(
            num_tokens, s_logit_init, default(s_logit_scale, dim**-0.5)
        )

        self.ignore_index = ce_ignore_index

    def lookup(self, ids):
        token_embed = self.token_embed.weight
        return token_embed[ids]

    def encode(self, latents, offsets=None):
        """Convert input ids to encoded representations"""
        for attn, ff in self.layers:
            tokens = attn(latents, offsets=offsets)
            tokens = ff(tokens)

        return tokens

    def decode(self, tokens):
        """Convert final tokens to logits"""
        if exists(self.to_logits):
            logits = self.to_logits(tokens)
        else:
            # Use tied embeddings
            logits = einsum(tokens, self.token_embed.weight, "b n d, v d -> b n v")
        return logits * self.logit_scale()

    def forward(self, ids, offsets=None):
        tokens = self.lookup(ids)
        tokens = self.encode(tokens, offsets=offsets)
        return self.decode(tokens)


def fourier_act(x: torch.Tensor) -> torch.Tensor:
    # Create a new tensor with alternating sin/cos
    x_out = torch.empty_like(x)
    x_out[..., 0::2] = torch.cos(x[..., 0::2])
    x_out[..., 1::2] = torch.sin(x[..., 1::2])
    return x_out


class RFF(Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(d_in, d_out) * (2 / d_in) ** 0.5)
        self.pre_scale = Scale(d_out, init=1.0, scale=d_out**-0.5)
        self.l2norm = partial(l2norm, norm_eps=0.0)

    def forward(self, x):
        # Project with pre-activation scaling and apply Fourier features
        x = (x @ self.kernel) * self.pre_scale()
        x = fourier_act(x)
        return self.l2norm(x)


class DecoderBranch(nGPT):
    def __init__(self, *args, is_trend=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_trend = is_trend

    def generate_offsets(self, sequence_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate offsets tensor from sequence IDs for DeltaNet.
        Offsets mark the start/end positions of each sequence in the batch.

        Args:
            sequence_ids: Tensor of shape [batch_size, seq_length] containing sequence IDs
                        where each unique ID represents a separate sequence

        Returns:
            offsets: Tensor of shape [num_sequences + 1] where:
                    - offsets[i] is the start position of sequence i
                    - offsets[i+1] is the end position of sequence i
        """
        # Flatten batch dimension
        flat_ids = sequence_ids.reshape(-1)

        # Find where sequence IDs change to mark sequence boundaries
        boundaries = torch.cat(
            [
                torch.tensor([0], device=sequence_ids.device),
                torch.where(flat_ids[1:] != flat_ids[:-1])[0] + 1,
                torch.tensor([len(flat_ids)], device=sequence_ids.device),
            ]
        )

        return boundaries

    def forward(self, tokens, seq_id, return_states=False, prev_states=None):
        values = None
        match prev_states, return_states:
            case (None, True):
                states = [None] * len(self.layers)
            case (None, False):
                states = None
            case (_, _):
                states = prev_states

        for i, (attn, ff) in enumerate(self.layers):
            offsets = self.generate_offsets(seq_id)
            tokens, values, state = attn(
                tokens,
                offsets=offsets,
                value_residual=values,
                return_values=True,
                return_states=return_states,
                prev_states=states[i] if prev_states else None,
            )

            if return_states:
                states[i] = state
            tokens = ff(tokens)

        return tokens, states

    @torch.no_grad()
    def norm_weights_(self):
        for module in self.modules():
            if not isinstance(module, NormLinear):
                continue

            module.norm_weights_()

    def register_step_post_hook(self, optimizer):
        assert hasattr(optimizer, "register_step_post_hook")

        def hook(*_):
            self.norm_weights_()

        return optimizer.register_step_post_hook(hook)


class AudioDecoder(Module):
    def __init__(
        self,
        num_classes=10,  # Changed from num_tokens to num_classes
        input_dim=None,
        dim=1024,
        depth=4,
        heads=8,
        chunk_size=64,
        attn_norm_qk=True,
        ff_expand_factor=4.0,
        manual_norm_weights=False,
        num_hyperspheres=1,
        causal=True,
        add_value_residual=True,
        alpha_init=None,
        s_logit_init=1.0,
        s_logit_scale=None,
        alpha_attn_init=None,
        alpha_attn_scale=None,
        alpha_ff_init=None,
        alpha_ff_scale=None,
        s_qk_init=1.0,
        s_qk_scale=None,
        s_ff_hidden_init=1.0,
        s_ff_hidden_scale=1.0,
        s_ff_gate_init=1.0,
        s_ff_gate_scale=1.0,
        attn_flash_kwargs=dict(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ),
        norm_eps=1e-5,
    ):
        assert dim % heads == 0, "dim must be divisible by heads"
        dim_head = dim // heads
        super().__init__()

        # Audio processing components
        self.input_dim = input_dim
        self.audio_rff = RFF(input_dim or dim, dim)  # Process raw audio into embeddings

        # Main processing branch
        branch_kwargs = dict(
            num_tokens=dim,  # Using dim as token size since we're not doing token embedding
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            attn_norm_qk=attn_norm_qk,
            ff_expand_factor=ff_expand_factor,
            manual_norm_weights=manual_norm_weights,
            num_hyperspheres=num_hyperspheres,
            causal=causal,
            add_value_residual=add_value_residual,
            alpha_init=alpha_init,
            s_logit_init=s_logit_init,
            s_logit_scale=s_logit_scale,
            alpha_attn_init=alpha_attn_init,
            alpha_attn_scale=alpha_attn_scale,
            alpha_ff_init=alpha_ff_init,
            alpha_ff_scale=alpha_ff_scale,
            s_qk_init=s_qk_init,
            s_qk_scale=s_qk_scale,
            s_ff_hidden_init=s_ff_hidden_init,
            s_ff_hidden_scale=s_ff_hidden_scale,
            s_ff_gate_init=s_ff_gate_init,
            s_ff_gate_scale=s_ff_gate_scale,
            attn_flash_kwargs=attn_flash_kwargs,
            norm_eps=norm_eps,
            chunk_size=chunk_size,
        )

        self.processor = DecoderBranch(is_trend=False, **branch_kwargs)

        # Classification head
        self.classifier = FeedForward(dim, dim_out=num_classes)
        self.l2norm = L2Norm(dim=-1)

    def forward(self, waveform, seq_id=None, return_latent=False):
        """
        Args:
            waveform: Audio waveform of shape [batch, sequence_length]
            seq_id: Optional sequence IDs for batching
        """
        if seq_id is None:
            seq_id = torch.zeros_like(waveform[..., 0], dtype=torch.long)

        embeddings = self.audio_rff(waveform)
        embeddings = self.l2norm(embeddings)

        # Process through transformer
        latent, _ = self.processor(embeddings, seq_id=seq_id)
        if return_latent:
            return latent

        return self.classifier(latent)
