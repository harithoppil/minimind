# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                     TempModel - New Architecture
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#
# This file is for building a new architecture iteratively.
# Once complete, it will be integrated into the repo with a flag.
#
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict
from dataclasses import dataclass, field
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.activations import ACT2FN

# Reuse common components from minimind
from model.model_minimind import RMSNorm, repeat_kv


# ============================================================================
#                              ROUTING INFO
# ============================================================================

@dataclass
class RoutingInfo:
    """
    Tracks which tokens are routed to which experts at each layer.
    This enables tokens to merge/split across layers dynamically.
    
    Example: Token 1 and 2 in different experts at layer 1,
             but same expert (and hence same attention) at layer 2.
    """
    layer_id: int
    # expert_assignments[expert_id] = list of token indices routed to that expert
    expert_assignments: Dict[int, torch.Tensor] = field(default_factory=dict)
    # token_to_expert[token_idx] = list of expert_ids this token is routed to
    token_to_expert: Dict[int, List[int]] = field(default_factory=dict)
    # Routing weights per token per expert
    routing_weights: Optional[torch.Tensor] = None
    
    def get_tokens_for_expert(self, expert_id: int) -> torch.Tensor:
        """Get token indices routed to a specific expert."""
        return self.expert_assignments.get(expert_id, torch.tensor([]))
    
    def get_experts_for_token(self, token_idx: int) -> List[int]:
        """Get expert IDs that a specific token is routed to."""
        return self.token_to_expert.get(token_idx, [])


# ============================================================================
#                              PoPE (Polar Position Encoding)
# ============================================================================

def precompute_pope_freqs(dim: int, end: int = 32768, theta: float = 10000.0):
    """
    Precompute cos/sin frequencies for PoPE.
    Same frequency computation as RoPE, but used differently.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    # Expand to full dim by repeating each frequency
    freqs_cos = freqs_cos.repeat_interleave(2, dim=-1)  # [end, dim]
    freqs_sin = freqs_sin.repeat_interleave(2, dim=-1)  # [end, dim]
    return freqs_cos, freqs_sin


def apply_pope(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    """
    Polar Position Encoding (PoPE) - NOT standard RoPE!
    
    Key Difference from RoPE:
    - RoPE: Rotates the entire vector (mixes content + position)
    - PoPE: Treats vector as complex number z = r*e^(iθ)
            r (magnitude) = content ("King")
            θ (phase) = position (Token 5)
    
    Implementation:
    1. Extract magnitude: r = softplus(x)  ← Forces network to learn intensity
    2. Apply phase: z = r * (cos(θ) + i*sin(θ))
    3. Return: [real, imag] = [r*cos(θ), r*sin(θ)]
    
    This is geometrically cleaner - content and position are separated!
    
    Input: x with shape [B, T, H, D]
    Output: [B, T, H, 2*D] (doubled dimension!)
    """
    B, T, H, D = x.shape
    
    # Handle sequence length exceeding precomputed frequencies
    if T > freqs_cos.shape[0]:
        freqs_cos, freqs_sin = precompute_pope_freqs(D, T, theta=10000.0)
        freqs_cos = freqs_cos.to(x.device)
        freqs_sin = freqs_sin.to(x.device)
    
    # Extract magnitude (content/intensity) - KEY DIFFERENCE FROM ROPE!
    mu = F.softplus(x)  # r = magnitude = semantic content
    
    # Fetch phase (position) and reshape for broadcasting
    cos = freqs_cos[:T].view(1, T, 1, D).to(x.device)  # cos(θ)
    sin = freqs_sin[:T].view(1, T, 1, D).to(x.device)  # sin(θ)
    
    # Polar to Cartesian: z = r*e^(iθ) = r*cos(θ) + i*r*sin(θ)
    x_real = mu * cos  # Real part
    x_imag = mu * sin  # Imaginary part
    
    return torch.cat([x_real, x_imag], dim=-1)  # [B, T, H, 2*D]


# ============================================================================
#                              CONFIG
# ============================================================================

class TempModelConfig(PretrainedConfig):
    model_type = "tempmodel"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 50256,
            eos_token_id: int = 50256,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 50257,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            flash_attn: bool = True,
            # MoE config
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        # MoE config
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob


# ============================================================================
#                              MODEL COMPONENTS
# ============================================================================


class Attention(nn.Module):
    """
    Hybrid Attention with:
    - "Smart Vectors": MLP projections for Q/K (1x expansion) and V (4x expansion)
    - "Smart Ruler": Learned bilinear interaction matrix W with SiLU during comparison
    - PoPE (Polar Position Encoding): Separates content (magnitude) from position (phase)
    
    After PoPE, Q/K have dimension 2*head_dim (real + imaginary parts).
    Computes: silu(Q_pope @ W) @ K_pope^T
    """
    def __init__(self, config: TempModelConfig, layer_id: int = 0, expert_id: int = -1):
        """
        Args:
            config: Model configuration
            layer_id: Which layer this attention belongs to (0-indexed)
            expert_id: Which expert this is (-1 for non-MoE, 0+ for MoE experts)
        """
        super().__init__()
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads else config.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.pope_dim = self.head_dim * 2  # PoPE doubles the dimension
        self.n_rep = self.num_attention_heads // self.num_key_value_heads
        
        q_out_dim = config.num_attention_heads * self.head_dim
        k_out_dim = self.num_key_value_heads * self.head_dim
        v_out_dim = self.num_key_value_heads * self.head_dim
        
        # Q projection: MLP with 1x expansion (hidden -> hidden -> q_dim)
        self.q_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, q_out_dim, bias=False)
        )
        
        # K projection: MLP with 1x expansion (hidden -> hidden -> k_dim)
        self.k_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, k_out_dim, bias=False)
        )
        
        # V projection: MLP with 4x expansion (hidden -> 4*hidden -> v_dim)
        # V does NOT go through PoPE, so output stays at head_dim
        v_intermediate = config.hidden_size * 4
        self.v_proj = nn.Sequential(
            nn.Linear(config.hidden_size, v_intermediate, bias=False),
            nn.SiLU(),
            nn.Linear(v_intermediate, v_out_dim, bias=False)
        )
        
        # Bilinear interaction matrix W: per-head learned transformation
        # Shape: (n_heads, pope_dim, pope_dim) - works on PoPE-expanded Q
        self.W_bilinear = nn.Parameter(torch.empty(config.num_attention_heads, self.pope_dim, self.pope_dim))
        self._init_bilinear_weights()
        
        # Output projection: from pope_dim back to hidden_size
        self.o_proj = nn.Linear(config.num_attention_heads * self.pope_dim, config.hidden_size, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def _init_bilinear_weights(self):
        """Initialize W as identity + small noise for stable training start."""
        # Initialize each head's matrix as identity
        with torch.no_grad():
            for h in range(self.num_attention_heads):
                nn.init.eye_(self.W_bilinear[h])
            # Add small noise for symmetry breaking
            self.W_bilinear.add_(torch.randn_like(self.W_bilinear) * 0.02)

    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            attention_mask: Optional[torch.Tensor] = None,
            attend_indices: Optional[torch.Tensor] = None,
            routing_history: Optional[List['RoutingInfo']] = None
    ):
        """
        Forward pass with optional selective attention.
        
        Args:
            x: Input tensor [bsz, seq_len, hidden_size]
            position_embeddings: (freqs_cos, freqs_sin) for PoPE
            past_key_value: Cached (K, V) from previous forward passes
            use_cache: Whether to return updated KV cache
            attention_mask: Standard attention mask for padding
            attend_indices: Optional tensor of token indices to attend to.
                           Shape: [num_attend_tokens] - indices into the full KV sequence
                           E.g., [0, 1, 2, 3, 4, 7, 8] means only attend to these positions
                           If None, attend to all tokens (default behavior)
            routing_history: Optional list of RoutingInfo from previous layers.
                           Allows this attention to be aware of how tokens were routed
                           in earlier layers (e.g., tokens 1,2 in different experts at layer 1
                           but same expert at current layer).
        
        Returns:
            output: [bsz, seq_len, hidden_size]
            past_kv: Updated KV cache (full, not filtered)
        """
        bsz, seq_len, _ = x.shape
        
        # Project to Q, K, V using MLPs
        xq = self.q_proj(x).view(bsz, seq_len, self.num_attention_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply PoPE (Polar Position Encoding) - doubles dimension to pope_dim
        # position_embeddings = (freqs_cos, freqs_sin) from precompute_pope_freqs
        freqs_cos, freqs_sin = position_embeddings
        xq = apply_pope(xq, freqs_cos, freqs_sin)  # [B, T, H, 2*head_dim]
        xk = apply_pope(xk, freqs_cos, freqs_sin)  # [B, T, H, 2*head_dim]
        # V does NOT get position encoding - pure content
        
        # KV cache handling (with pope_dim)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        # Store full KV for cache (before filtering)
        past_kv = (xk, xv) if use_cache else None
        
        # ========== Selective Attention: Filter K/V by attend_indices ==========
        # If attend_indices specified, only attend to those token positions
        if attend_indices is not None:
            # attend_indices: [num_attend] - indices into sequence dimension
            # Filter K and V to only include specified positions
            # xk: [bsz, full_seq_len, n_kv_heads, pope_dim]
            # xv: [bsz, full_seq_len, n_kv_heads, head_dim]
            xk = xk[:, attend_indices, :, :]  # [bsz, num_attend, n_kv_heads, pope_dim]
            xv = xv[:, attend_indices, :, :]  # [bsz, num_attend, n_kv_heads, head_dim]
        
        # Transpose for attention: (bsz, n_heads, seq_len, pope_dim/head_dim)
        xq = xq.transpose(1, 2)  # [bsz, n_heads, seq_len, pope_dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)  # [bsz, n_heads, attend_len, pope_dim]
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)  # [bsz, n_heads, attend_len, head_dim]
        
        # ========== Bilinear Attention: silu(Q @ W) @ K^T ==========
        # Apply bilinear transformation: Q @ W per head
        # xq: (bsz, n_heads, seq_len, pope_dim)
        # W_bilinear: (n_heads, pope_dim, pope_dim)
        q_transformed = torch.einsum("bhld,hde->bhle", xq, self.W_bilinear)
        
        # Apply SiLU activation during comparison (the "smart ruler" non-linearity)
        q_activated = F.silu(q_transformed)
        
        # Compute attention scores: silu(Q @ W) @ K^T
        # Note: K may be filtered, so attend_len <= full_seq_len
        scale_factor = 1.0 / math.sqrt(self.pope_dim)
        scores = (q_activated @ xk.transpose(-2, -1)) * scale_factor
        
        # Causal mask - adjusted for potentially filtered K
        L = seq_len  # query length
        S = xk.size(-2)  # key length (may be filtered)
        
        if attend_indices is None:
            # Standard causal mask
            causal_mask = torch.triu(
                torch.full((L, S), float("-inf"), device=scores.device),
                diagonal=S - L + 1
            )
            scores = scores + causal_mask
        else:
            # With attend_indices: need position-aware causal masking
            # Each query at position q can only attend to indices where attend_indices[j] <= q's absolute position
            # For now, if using attend_indices, we assume the caller handles causality
            # or we compute based on absolute positions
            query_positions = torch.arange(S - L, S, device=scores.device).view(1, 1, L, 1)
            key_positions = attend_indices.view(1, 1, 1, S).to(scores.device)
            causal_mask = torch.where(
                key_positions <= query_positions,
                torch.zeros(1, device=scores.device),
                torch.full((1,), float("-inf"), device=scores.device)
            )
            scores = scores + causal_mask
        
        # Attention mask (for padding etc.)
        if attention_mask is not None:
            if attend_indices is not None:
                # Filter attention mask to match filtered K/V
                attention_mask = attention_mask[:, attend_indices]
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -1e9
            scores = scores + extended_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # attn_weights: [bsz, n_heads, seq_len, attend_len]
        # xv: [bsz, n_heads, attend_len, head_dim]
        output = attn_weights @ xv  # [bsz, n_heads, seq_len, head_dim]
        
        # Expand output to pope_dim to match o_proj expectation
        output = output.repeat(1, 1, 1, 2)  # [bsz, n_heads, seq_len, pope_dim]
        
        # Reshape and project output
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # [bsz, seq_len, n_heads * pope_dim]
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class MLP(nn.Module):
    """
    Feed-forward network with SwiGLU activation.
    SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    def __init__(self, config: TempModelConfig):
        super().__init__()
        # Calculate intermediate size if not specified
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)  # Round to multiple of 64
        else:
            intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  # SiLU by default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate) * up, then down project
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    Gating mechanism for Mixture of Experts.
    Routes tokens to top-k experts with load balancing auxiliary loss.
    Uses MLP with 4x expansion for richer routing decisions.
    """
    def __init__(self, config: TempModelConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        
        # Gate projection: MLP with 4x expansion (hidden -> 4*hidden -> n_experts)
        gate_intermediate = config.hidden_size * 4
        self.gate_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, gate_intermediate, bias=False),
            nn.SiLU(),
            nn.Linear(gate_intermediate, self.n_routed_experts, bias=False)
        )

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        
        # Compute router logits via MLP
        logits = self.gate_mlp(hidden_states)
        
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'Unsupported scoring function: {self.scoring_func}')
        
        # Select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        # Normalize top-k probabilities
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        
        # Compute auxiliary load balancing loss
        if self.training and self.alpha > 0.0:
            topk_idx_for_aux = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_aux = scores.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux,
                               torch.ones(bsz, seq_len * self.top_k, device=hidden_states.device))
                ce = ce / (seq_len * self.top_k / self.n_routed_experts)
                aux_loss = (ce * scores_for_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        
        return topk_idx, topk_weight, aux_loss


class MoEAttention(nn.Module):
    """
    Mixture of Experts with ATTENTION layers as experts (not MLPs!).
    
    Each routed expert is an Attention layer that only sees the tokens routed to it.
    This creates sparse attention patterns where different experts specialize
    on different subsets of tokens.
    
    Key insight: attend_indices allows each expert to only attend to tokens
    that were routed to it, creating efficient sparse attention.
    """
    def __init__(self, config: TempModelConfig, layer_id: int = 0):
        """
        Args:
            config: Model configuration
            layer_id: Which layer this MoE block belongs to
        """
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        # Pool of routed experts - ATTENTION layers, not MLPs!
        # Each expert knows its layer_id and expert_id
        self.experts = nn.ModuleList([
            Attention(config, layer_id=layer_id, expert_id=i) 
            for i in range(config.n_routed_experts)
        ])
        
        # Gating mechanism (MLP with 4x expansion)
        self.gate = MoEGate(config)
        
        # Optional shared attention expert that sees all tokens
        # Shared experts get expert_id starting after routed experts
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                Attention(config, layer_id=layer_id, expert_id=config.n_routed_experts + i)
                for i in range(config.n_shared_experts)
            ])
        
        self.aux_loss = None

    def forward(
            self, 
            x: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            routing_history: Optional[List[RoutingInfo]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], RoutingInfo]:
        """
        Forward pass with attention experts.
        
        Args:
            x: Input tensor [bsz, seq_len, hidden_size]
            position_embeddings: (freqs_cos, freqs_sin) for PoPE
            past_key_values: List of (K, V) caches, one per expert
            use_cache: Whether to return updated KV caches
            routing_history: List of RoutingInfo from previous layers (for awareness)
            
        Returns:
            output: [bsz, seq_len, hidden_size]
            expert_kv_caches: List of (K, V) caches per expert (if use_cache)
            routing_info: RoutingInfo for this layer (which tokens went where)
        """
        identity = x
        bsz, seq_len, hidden_size = x.shape
        
        # Get routing decisions: which tokens go to which experts
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # topk_idx: [bsz * seq_len, num_experts_per_tok] - expert indices per token
        # topk_weight: [bsz * seq_len, num_experts_per_tok] - weights per expert
        
        # Build RoutingInfo for this layer
        routing_info = RoutingInfo(
            layer_id=self.layer_id,
            routing_weights=topk_weight.view(bsz, seq_len, -1)
        )
        
        # Build expert_assignments and token_to_expert mappings
        topk_idx_reshaped = topk_idx.view(bsz, seq_len, -1)
        for token_idx in range(seq_len):
            # Get experts this token is routed to (across all batches, take first batch for simplicity)
            expert_ids = topk_idx_reshaped[0, token_idx].tolist()
            routing_info.token_to_expert[token_idx] = expert_ids
            
            for expert_id in expert_ids:
                if expert_id not in routing_info.expert_assignments:
                    routing_info.expert_assignments[expert_id] = []
                if token_idx not in routing_info.expert_assignments[expert_id]:
                    routing_info.expert_assignments[expert_id].append(token_idx)
        
        # Convert expert_assignments lists to tensors
        for expert_id in routing_info.expert_assignments:
            routing_info.expert_assignments[expert_id] = torch.tensor(
                routing_info.expert_assignments[expert_id], 
                device=x.device
            )
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Initialize KV caches for each expert
        expert_kv_caches = [None] * self.config.n_routed_experts if use_cache else None
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Get token indices routed to this expert
            attend_indices = routing_info.expert_assignments.get(expert_idx, None)
            
            if attend_indices is None or len(attend_indices) == 0:
                continue
            
            # CRITICAL FIX: Only pass tokens routed to this expert as QUERIES
            # This prevents NaN from queries that have no valid (non-future) keys
            # attend_indices are the token positions this expert owns
            
            # Extract only the tokens routed to this expert
            # attend_indices: [num_routed_tokens] - positions in the sequence
            x_expert = x[:, attend_indices, :]  # [bsz, num_routed, hidden_size]
            
            # Get past KV for this expert
            past_kv = past_key_values[expert_idx] if past_key_values else None
            
            # Run attention:
            # - Queries: only tokens routed to this expert
            # - Keys/Values: same tokens (attend_indices passed to filter K/V if needed)
            # - Both Q and K/V come from the same routed subset
            expert_out, expert_kv = expert(
                x_expert,  # Only routed tokens, not full x
                position_embeddings,
                past_key_value=past_kv,
                use_cache=use_cache,
                attend_indices=None,  # No filtering needed - Q and K/V are same set
                routing_history=routing_history
            )
            
            if use_cache:
                expert_kv_caches[expert_idx] = expert_kv
            
            # Get the routing weights for tokens assigned to this expert
            topk_weight_reshaped = topk_weight.view(bsz, seq_len, -1)
            
            # Compute weights for routed tokens
            token_weights = torch.zeros(bsz, len(attend_indices), 1, device=x.device, dtype=x.dtype)
            
            for k in range(self.config.num_experts_per_tok):
                # Get weights for positions in attend_indices
                mask = topk_idx_reshaped[:, attend_indices, k] == expert_idx
                weights = topk_weight_reshaped[:, attend_indices, k]
                token_weights[mask] = weights[mask].unsqueeze(-1)
            
            # Scatter expert output back to full output tensor at correct positions
            weighted_output = expert_out * token_weights  # [bsz, num_routed, hidden_size]
            output[:, attend_indices, :] = output[:, attend_indices, :] + weighted_output
        
        # Add shared experts (process all tokens)
        if self.config.n_shared_experts > 0:
            for shared_expert in self.shared_experts:
                shared_out, _ = shared_expert(
                    identity,
                    position_embeddings,
                    past_key_value=None,
                    use_cache=False,
                    attend_indices=None,  # Shared expert sees all tokens
                    routing_history=routing_history
                )
                output = output + shared_out
        
        self.aux_loss = aux_loss
        return output, expert_kv_caches, routing_info


# ============================================================================
#                              RESIDUAL GATE
# ============================================================================


class ResidualGate(nn.Module):
    """
    Learned residual mixing gate that allows each layer to attend to
    all previous layer outputs with learned weights.
    
    Instead of simple residual: x + layer(x)
    This allows: norm(sum(alpha[i,j] * past_outputs[j] for j in 0..i))
    
    Initialized with identity-like pattern (each layer attends to previous layer),
    but can learn to mix outputs from any previous layers.
    """
    def __init__(self, config: TempModelConfig):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        
        # Alpha[i, j] = weight for layer i attending to output of layer j
        # Shape: [num_layers, num_layers]
        # Alpha[i, :i+1] are the relevant weights for layer i
        self.alpha = nn.Parameter(torch.zeros(self.num_layers, self.num_layers))
        
        # Normalization for the weighted sum
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize: each layer primarily attends to its immediate predecessor
        with torch.no_grad():
            for i in range(1, self.num_layers):
                self.alpha[i, i - 1] = 1.0
            # Layer 0 attends to input (handled separately)
            self.alpha[0, 0] = 1.0
    
    def forward(
            self, 
            current_layer_idx: int, 
            past_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute weighted combination of past outputs for current layer input.
        
        Args:
            current_layer_idx: Index of current layer (0-indexed)
            past_outputs: List of tensors from layers 0..current_layer_idx
                         past_outputs[0] is the embedded input
                         past_outputs[i] is output of layer i-1
                         
        Returns:
            Weighted, normalized combination of past outputs
        """
        if current_layer_idx == 0:
            # First layer just uses the input (embedded tokens)
            return past_outputs[0]
        
        # Get weights for this layer attending to all previous outputs
        # past_outputs has current_layer_idx + 1 entries (input + outputs of layers 0..idx-1)
        num_past = len(past_outputs)
        weights = self.alpha[current_layer_idx, :num_past]
        
        # Stack past outputs: [num_past, bsz, seq_len, hidden_size]
        stack = torch.stack(past_outputs, dim=0)
        
        # Apply weights: [num_past, 1, 1, 1] for broadcasting
        w = weights.view(-1, 1, 1, 1)
        
        # Weighted sum with normalization
        return self.norm((w * stack).sum(dim=0))


# ============================================================================
#                              TRANSFORMER BLOCK
# ============================================================================


class TempModelBlock(nn.Module):
    """
    Single transformer block for TempModel.
    Uses MoEAttention (with Attention experts) instead of standard attention.
    
    NO standalone MLP - MLPs are only inside:
    1. Attention (Q/K/V projections with expansion)
    2. Router gate (4x expansion MLP for routing decisions)
    
    Structure:
        x → LayerNorm → MoEAttention → residual
    """
    def __init__(self, layer_id: int, config: TempModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Pre-attention layer norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MoE Attention - each expert is an Attention layer with sparse routing
        self.moe_attn = MoEAttention(config, layer_id=layer_id)
        
        self.aux_loss = None

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            attention_mask: Optional[torch.Tensor] = None,
            routing_history: Optional[List[RoutingInfo]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], RoutingInfo]:
        """
        Forward pass through one transformer block.
        
        Returns:
            hidden_states: Output tensor
            present_key_values: KV caches for each expert
            routing_info: Routing decisions for this layer
        """
        # Attention with residual (no MLP after)
        residual = hidden_states
        hidden_states, present_key_values, routing_info = self.moe_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
            routing_history=routing_history
        )
        hidden_states = residual + hidden_states
        
        # Store aux loss from MoE
        self.aux_loss = self.moe_attn.aux_loss
        
        return hidden_states, present_key_values, routing_info


# ============================================================================
#                              MAIN MODEL
# ============================================================================


class TempModel(nn.Module):
    """
    TempModel - Novel architecture with:
    - PoPE (Polar Position Encoding)
    - MoE Attention (experts are Attention layers, not MLPs)
    - Bilinear attention with silu: silu(Q @ W) @ K^T
    - Per-expert KV caches
    - Routing history tracking across layers
    - ResidualGate: learned mixing of all previous layer outputs
    """
    def __init__(self, config: TempModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TempModelBlock(layer_id=i, config=config) 
            for i in range(config.num_hidden_layers)
        ])
        
        # ResidualGate: learned mixing of all previous layer outputs
        self.residual_gate = ResidualGate(config)
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Precompute PoPE frequencies
        head_dim = config.hidden_size // config.num_attention_heads
        freqs_cos, freqs_sin = precompute_pope_freqs(
            dim=head_dim,
            end=config.max_position_embeddings,
            theta=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
            use_cache: bool = False,
            **kwargs
    ) -> Tuple[torch.Tensor, List, torch.Tensor, List[RoutingInfo]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: [bsz, seq_len] token IDs
            attention_mask: [bsz, seq_len] attention mask
            past_key_values: Nested list [layer_id][expert_id] = (K, V)
            use_cache: Whether to return KV caches
            
        Returns:
            hidden_states: [bsz, seq_len, hidden_size]
            all_key_values: Nested KV caches [layer_id][expert_id]
            aux_loss: Combined auxiliary loss from all layers
            routing_history: List of RoutingInfo, one per layer
        """
        batch_size, seq_length = input_ids.shape
        
        # Handle past_key_values
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # Compute start position from cache
        start_pos = 0
        if past_key_values[0] is not None and past_key_values[0][0] is not None:
            # past_key_values[0][0] = first layer, first expert's (K, V)
            start_pos = past_key_values[0][0][0].shape[1]
        
        # Embed tokens
        embedded = self.dropout(self.embed_tokens(input_ids))
        
        # Get position embeddings for current sequence
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )
        
        # Track all layer outputs for ResidualGate
        # past_outputs[0] = embedded input
        # past_outputs[i+1] = output of layer i
        past_outputs = [embedded]
        
        # Forward through layers
        all_key_values = []
        routing_history = []
        
        for layer_idx, layer in enumerate(self.layers):
            # Use ResidualGate to compute input from all previous outputs
            layer_input = self.residual_gate(layer_idx, past_outputs)
            
            # Forward through layer
            hidden_states, present_kv, routing_info = layer(
                layer_input,
                position_embeddings,
                past_key_values=past_key_values[layer_idx],
                use_cache=use_cache,
                attention_mask=attention_mask,
                routing_history=routing_history
            )
            
            # Store this layer's output for future layers
            past_outputs.append(hidden_states)
            
            all_key_values.append(present_kv)
            routing_history.append(routing_info)
        
        # Final output: use last layer output (or could use residual_gate for final)
        hidden_states = self.norm(past_outputs[-1])
        
        # Compute total auxiliary loss
        aux_loss = sum(
            [l.aux_loss for l in self.layers if l.aux_loss is not None],
            hidden_states.new_zeros(1).squeeze()
        )
        
        return hidden_states, all_key_values, aux_loss, routing_history


# ============================================================================
#                              CAUSAL LM WRAPPER
# ============================================================================


class TempModelForCausalLM(PreTrainedModel, GenerationMixin):
    """
    TempModel with language model head for causal language modeling.
    Compatible with HuggingFace training and generation.
    """
    config_class = TempModelConfig
    
    def __init__(self, config: TempModelConfig = None):
        self.config = config or TempModelConfig()
        super().__init__(self.config)
        
        # Core model
        self.model = TempModel(self.config)
        
        # Language model head (tied to embeddings)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # Weight tying: embed_tokens and lm_head share weights
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
            use_cache: bool = False,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with optional loss computation.
        
        Args:
            input_ids: [bsz, seq_len] token IDs
            attention_mask: [bsz, seq_len] attention mask
            labels: [bsz, seq_len] target token IDs for loss computation
            past_key_values: Nested KV caches
            use_cache: Whether to return KV caches
            logits_to_keep: How many logits to keep (for memory efficiency)
            
        Returns:
            CausalLMOutputWithPast with loss, logits, past_key_values, hidden_states
            Also includes aux_loss attribute for MoE auxiliary loss
        """
        # Forward through model
        hidden_states, past_key_values, aux_loss, routing_history = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        
        # Compute logits (optionally keep only last N for memory efficiency)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        # Create output
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )
        
        # Add custom attributes
        output.aux_loss = aux_loss
        output.routing_history = routing_history
        
        return output
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation - handle KV cache."""
        if past_key_values is not None:
            # Only use last token if we have cache
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            **kwargs
        }
