"""
GPU Training - Latent Predictor SentenceFormer (CORRECTED)
Architecture: VAE unfolded over time (Mixer→Decoder||Encoder parallel)

KEY ARCHITECTURAL POINTS:
1. Decoder is AUTOREGRESSIVE (attends to its own tokens like GPT)
2. Decoder CANNOT see previous sentence tokens (only vectors)
3. Encoder uses z_pred + actual tokens to create refined vector
4. During TRAINING: Encoder uses ground truth (no waiting)
5. During INFERENCE: Encoder waits for decoder to finish
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
import math
import time
import glob

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
CONFIG = {
    "vocab_size": 50258,
    "dim": 512,
    "n_layers": 28,
    "n_heads": 8,
    "head_dim": 64,
    "batch_size": 8,
    "learning_rate": 4e-4,
    "max_iters": 2000,
    "eval_interval": 100,
    "eval_iters": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dropout": 0.01,
    "max_sentences": 16,
    "max_tokens_per_sent": 48,
    "local_encoder_layers": 5,
    "global_mixer_layers": 20,
    "local_decoder_layers": 5,
    "data_dir": "processed_shards",
    "checkpoint_path": None,
    "save_path": "sentanceFormer_latent_predictor_gpu.pth",
    "seed": 42,
}

print(f"🎯 Latent Predictor - GPU Training")
print(f"Running on {CONFIG['device']}")

MAX_SENTENCES = CONFIG["max_sentences"]
MAX_TOKENS = CONFIG["max_tokens_per_sent"]
MAX_SEQ_LEN_TOTAL = MAX_SENTENCES + MAX_TOKENS

# ============================================================================
# 2. TOKENIZER & DATA
# ============================================================================
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<compress>"]})
COMPRESS_TOKEN_ID = tokenizer.convert_tokens_to_ids("<compress>")
PAD_TOKEN_ID = tokenizer.pad_token_id

def load_sharded_data(data_dir):
    all_stories = []
    shard_files = sorted(glob.glob(os.path.join(data_dir, "*_shard_*.pt")))
    if not shard_files:
        raise FileNotFoundError(f"No shards found in {data_dir}!")
    print(f"Loading {len(shard_files)} shards...")
    for sf in shard_files:
        shard_data = torch.load(sf)
        all_stories.extend(shard_data)
    print(f"Loaded {len(all_stories)} stories.")
    return all_stories

all_data = load_sharded_data(CONFIG["data_dir"])
# Separate train/val based on shard filenames
train_stories = [s for sf in glob.glob(os.path.join(CONFIG["data_dir"], "train_shard_*.pt")) 
                 for s in torch.load(sf)]
val_stories = [s for sf in glob.glob(os.path.join(CONFIG["data_dir"], "validation_shard_*.pt")) 
               for s in torch.load(sf)]
print(f"Train: {len(train_stories)}, Val: {len(val_stories)}")

# ============================================================================
# 3. BATCH CREATION
# ============================================================================
def create_2d_batch(stories, batch_size, device):
    if not stories:
        return None, None, None, None
    indices = torch.randint(0, len(stories), (batch_size,))
    batch_input = torch.full((batch_size, MAX_SENTENCES, MAX_SEQ_LEN_TOTAL), PAD_TOKEN_ID, dtype=torch.long, device=device)
    batch_labels = torch.full((batch_size, MAX_SENTENCES, MAX_SEQ_LEN_TOTAL), -100, dtype=torch.long, device=device)
    token_start_positions = torch.zeros((batch_size, MAX_SENTENCES), dtype=torch.long, device=device)
    compress_positions = torch.zeros((batch_size, MAX_SENTENCES), dtype=torch.long, device=device)

    for b, idx in enumerate(indices):
        story_sents = stories[idx]
        num_sents = min(len(story_sents), MAX_SENTENCES)
        for s_idx in range(num_sents):
            tokens = story_sents[s_idx][:MAX_TOKENS]
            token_len = len(tokens)
            start_pos = s_idx
            token_start_positions[b, s_idx] = start_pos
            batch_input[b, s_idx, start_pos:start_pos + token_len] = torch.tensor(tokens, dtype=torch.long, device=device)

            if s_idx > 0:
                batch_labels[b, s_idx, start_pos - 1] = tokens[0]
                if token_len > 1:
                    batch_labels[b, s_idx, start_pos:start_pos + token_len - 1] = torch.tensor(tokens[1:], dtype=torch.long, device=device)
            else:
                if token_len > 1:
                    batch_labels[b, s_idx, start_pos:start_pos + token_len - 1] = torch.tensor(tokens[1:], dtype=torch.long, device=device)

            compress_positions[b, s_idx] = start_pos + token_len - 1
    return batch_input, batch_labels, compress_positions, token_start_positions

def get_batch(split):
    data = train_stories if split == "train" else val_stories
    return create_2d_batch(data, CONFIG["batch_size"], CONFIG["device"])

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(CONFIG["eval_iters"])
        for k in range(CONFIG["eval_iters"]):
            xb, yb, cp, ts = get_batch(split)
            if xb is None:
                continue
            logits, loss = model(xb, yb, cp, ts)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ============================================================================
# 4. LOOKUP NORMALIZATION
# ============================================================================
def norm(x):
    """Functional RMSNorm with no learnable params"""
    return F.rms_norm(x, (x.size(-1),))

# ============================================================================
# 5. TRANSFORMER PRIMITIVES (ALL USE BILINEAR ATTENTION)
# ============================================================================
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 1).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x, freqs_cos, freqs_sin):
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
    """
    B, T, H, D = x.shape
    if T > freqs_cos.shape[0]:
        new_cos, new_sin = precompute_freqs_cis(D, T, theta=10000.0)
        freqs_cos = new_cos.to(x.device)
        freqs_sin = new_sin.to(x.device)
    
    # Extract magnitude (content/intensity) - KEY DIFFERENCE FROM ROPE!
    mu = F.softplus(x)  # r = magnitude = semantic content
    
    # Fetch phase (position)
    cos = freqs_cos[:T].view(1, T, 1, D).to(x.device)  # cos(θ)
    sin = freqs_sin[:T].view(1, T, 1, D).to(x.device)  # sin(θ)
    
    # Polar to Cartesian: z = r*e^(iθ) = r*cos(θ) + i*r*sin(θ)
    x_real = mu * cos  # Real part
    x_imag = mu * sin  # Imaginary part
    
    return torch.cat([x_real, x_imag], dim=-1)  # [B, T, H, 2*D]

class PackedSwiGLU(nn.Module):
    def __init__(self, dim, output_dim=None, expansion_factor=4/3):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        out = output_dim if output_dim is not None else dim
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, out, bias=False)
    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = torch.chunk(x12, 2, dim=-1)
        return self.w3(F.silu(x1) * x2)

def bilinear_scaled_dot_product_attention(query, key, value, interaction_matrix, attn_mask=None, dropout_p=0.0, is_causal=True, scale=None):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias
    query_transformed = torch.einsum("bhld,hde->bhle", query, interaction_matrix)
    query_activated = F.silu(query_transformed)
    attn_weight = query_activated @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=self.training)
    return attn_weight @ value

class BilinearAttention(nn.Module):
    """
    Bilinear Attention with PoPE (Polar Position Encoding).
    
    Used in ALL layers (encoder, decoder, mixer).
    
    Key features:
    1. PoPE instead of RoPE (magnitude = content, phase = position)
    2. Learnable interaction matrix W_bilinear
    3. Non-linear transformation: Score = SiLU(Q @ W_bi) @ K^T
    
    This allows each head to learn different logical relationships:
    - Head 1: Causality
    - Head 2: Contradiction
    - Head 3: Temporal ordering
    etc.
    """
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg["dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = self.dim // self.n_heads
        self.pope_dim = self.head_dim * 2  # PoPE doubles dimension (real + imag)
        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, self.dim, bias=False)
        self.wv = nn.Linear(self.dim, self.dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)
        
        # Learnable interaction tensor [Heads, PoPE_Dim, PoPE_Dim]
        self.W_bilinear = nn.Parameter(torch.Tensor(self.n_heads, self.pope_dim, self.pope_dim))
        
        # Initialize near identity (starts like dot product, learns complexity)
        nn.init.eye_(self.W_bilinear.view(-1, self.pope_dim, self.pope_dim).flatten(0, 1))
        with torch.no_grad():
            self.W_bilinear.add_(torch.randn_like(self.W_bilinear) * 0.02)
    
    def forward(self, x, freqs_cos, freqs_sin, mask=None, layer_past=None):
        B, T, C = x.shape
        
        # Linear projections
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        
        # Apply PoPE (Q and K become [B, T, H, 2*D])
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        # Transpose for attention: [B, H, T, 2*D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # V stays [B, H, T, D]
        
        # KV caching (for generation - optional)
        present = (k, v)
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            present = (k, v)
        
        # Bilinear attention with learned interaction matrix
        out = bilinear_scaled_dot_product_attention(
            q, k, v,
            interaction_matrix=self.W_bilinear,
            is_causal=True
        )
        
        # Output projection
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, C)), present

class TransformerBlock(nn.Module):
    """
    Transformer block using BilinearAttention (NOT standard attention).
    Used in: Encoder and Decoder layers.
    """
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg["dim"])
        self.attn = BilinearAttention(cfg)  # ← Uses bilinear, not standard!
        self.norm2 = nn.RMSNorm(cfg["dim"])
        self.mlp = PackedSwiGLU(cfg["dim"])
    
    def forward(self, x, freqs_cos, freqs_sin, mask=None):
        # BilinearAttention returns (output, present)
        # We ignore 'present' for encoder/decoder (no caching during training)
        attn_out, _ = self.attn(self.norm1(x), freqs_cos, freqs_sin, mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class MixerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg["dim"])
        self.attn = BilinearAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg["dim"])
        self.mlp = PackedSwiGLU(cfg["dim"])
    def forward(self, x, freqs_cos, freqs_sin, mask=None, layer_past=None):
        attn_out, present = self.attn(self.norm1(x), freqs_cos, freqs_sin, mask, layer_past)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, present

class ResidualGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = CONFIG["global_mixer_layers"]
        self.alpha = nn.Parameter(torch.zeros(self.num_layers, self.num_layers))
        self.norm = nn.RMSNorm(CONFIG["dim"])
        with torch.no_grad():
            for i in range(1, self.num_layers):
                self.alpha[i, i - 1] = 1.0
    def forward(self, current_layer_idx, past_outputs):
        if current_layer_idx == 0:
            return past_outputs[0]
        weights = self.alpha[current_layer_idx, :current_layer_idx + 1]
        stack = torch.stack(past_outputs, dim=0)
        w = weights.view(-1, 1, 1, 1)
        return self.norm((w * stack).sum(dim=0))
        
def get_relative_pe(global_cos, global_sin, context_len, total_len, device):
    # Context indices: 0, 1, ... s_idx-1
    ctx_ids = torch.arange(context_len, device=device)
    # Token indices: RESET to 0, 1, ... (total_len - context_len)
    tok_ids = torch.arange(total_len - context_len, device=device)
    
    indices = torch.cat([ctx_ids, tok_ids])
    
    # Gather from global buffers [MaxT, D] -> [1, TotalLen, 1, D]
    cos = global_cos[indices].unsqueeze(0).unsqueeze(2)
    sin = global_sin[indices].unsqueeze(0).unsqueeze(2)
    return cos, sin

# ============================================================================
# 6. LATENT PREDICTOR HOURGLASS TRANSFORMER (ALL BILINEAR ATTENTION)
# ============================================================================
class HourglassTransformer(nn.Module):
    """
    Latent Predictor with VAE-style architecture.
    
    KEY FEATURES:
    1. ALL layers use BilinearAttention (no standard attention)
    2. PoPE instead of RoPE (magnitude=content, phase=position)
    3. Lookup normalization after embedding
    4. Autoregressive decoder with encoder refinement
    
    Components:
    - Encoder: BilinearAttention (5 layers)
    - Mixer: BilinearAttention (20 layers) 
    - Decoder: BilinearAttention (5 layers)
    
    Total: 30 layers of BilinearAttention + PoPE
    """
    def __init__(self):
        super().__init__()
        dim = CONFIG["dim"]
        self.wte = nn.Embedding(CONFIG["vocab_size"], dim)
        self.local_encoder = nn.ModuleList([TransformerBlock(CONFIG) for _ in range(CONFIG["local_encoder_layers"])])
        self.global_mixer = nn.ModuleList([MixerBlock(CONFIG) for _ in range(CONFIG["global_mixer_layers"])])
        self.residual_gate = ResidualGate()
        self.local_decoder = nn.ModuleList([TransformerBlock(CONFIG) for _ in range(CONFIG["local_decoder_layers"])])
        self.ln_f = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, CONFIG["vocab_size"], bias=False)
        self.wte.weight = self.lm_head.weight
        fc_token, fs_token = precompute_freqs_cis(dim // CONFIG["n_heads"], MAX_SEQ_LEN_TOTAL + 128)
        self.register_buffer("freqs_cos_token", fc_token)
        self.register_buffer("freqs_sin_token", fs_token)
        fc_sent, fs_sent = precompute_freqs_cis(dim // CONFIG["n_heads"], MAX_SENTENCES + 8)
        self.register_buffer("freqs_cos_sent", fc_sent)
        self.register_buffer("freqs_sin_sent", fs_sent)
        self.start_context = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, idx, targets=None, compress_positions=None, token_start_positions=None):
        """
        CORRECTED LATENT PREDICTOR:
        
        Decoder Input: Past Encoder-Vectors + Current Mixer-Vector + Current Token-Vectors
        Decoder Output: Current Token-Vectors (autoregressive)
        
        Encoder Input: Past Encoder-Vectors + Current Mixer-Vector + Current Decoder Token-Vectors
        Encoder Output: Latest Past Encoder-Vector
        
        Mixer Input: Past encoder vectors
        Mixer Output: Current mixer vector
        """
        B, S, W = idx.shape
        device = idx.device
        dim = CONFIG["dim"]
        
        all_logits_list = []
        
        # Embed and normalize
        x_base = self.wte(idx.view(-1, W)).view(B, S, W, dim)
        x_base = norm(x_base)  # LOOKUP NORMALIZATION
        
        context_history = self.start_context.expand(B, 1, dim)
        all_logits_list = []
        all_z_pred = [] # <--- NEW
        all_z_real = [] # <--- NEW
    

        for s_idx in range(S):
            seq_mask = torch.triu(torch.ones(W, W, device=device, dtype=torch.bool), diagonal=1)
            
            # ═══════════════════════════════════════════════════════════
            # A. MIXER (PRIOR): z_pred = P(z_t | z_{<t})
            # ═══════════════════════════════════════════════════════════
            mix_len = context_history.size(1)
            mix_mask = torch.triu(torch.ones(mix_len, mix_len, device=device, dtype=torch.bool), diagonal=1)
            
            mixer_history = [context_history]
            for i, block in enumerate(self.global_mixer):
                block_input = self.residual_gate(i, mixer_history)
                block_out, _ = block(block_input, self.freqs_cos_sent, self.freqs_sin_sent, mix_mask)
                mixer_history.append(block_out)
            
            mix_out = mixer_history[-1]
            z_pred = mix_out[:, -1, :].unsqueeze(1)  # Current Mixer Vector
            all_z_pred.append(z_pred) # <--- SAVE IT

            

            # ═══════════════════════════════════════════════════════════
            # B. DECODER (LIKELIHOOD): P(x_t | z_pred)
            # Input: Past Encoder-Vectors + Current Mixer-Vector + Token-Vectors
            # ═══════════════════════════════════════════════════════════
            # Decoder uses CURRENT SENTENCE TOKENS (teacher forcing during training)
            decoder_x = x_base[:, s_idx, :, :].clone()
            
            # Inject past encoder vectors and current mixer vector
            # Layout: [z_0, z_1, ..., z_{s-2}, z_pred, tok_0, tok_1, ...]
            if s_idx > 0:
                if s_idx > 1:
                    # Inject past refined vectors (all except the last one)
                    decoder_x[:, :s_idx-1, :] = context_history[:, 1:s_idx, :]
                # Inject current mixer prediction in the last context slot
                decoder_x[:, s_idx-1, :] = z_pred.squeeze(1)
            
            current_cos, current_sin = get_relative_pe(
                self.freqs_cos_token, 
                self.freqs_sin_token, 
                context_len=s_idx, 
                total_len=W, 
                device=device
            )

            # Decoder processes autoregressively (can attend to its own tokens)
            dec_out = decoder_x
            for block in self.local_decoder:
                dec_out = block(dec_out, self.freqs_cos_token, self.freqs_sin_token, seq_mask)
            
            logits = self.lm_head(self.ln_f(dec_out))
            all_logits_list.append(logits)
            
            # ═══════════════════════════════════════════════════════════
            # C. ENCODER (POSTERIOR): z_real = Q(z_t | z_pred, x_t)
            # Input: Past Encoder-Vectors + Current Mixer-Vector + Decoder Token-Vectors
            # Output: Latest Past Encoder-Vector
            # ═══════════════════════════════════════════════════════════
            # Encoder uses SAME INPUT as decoder (during training)
            encoder_x = x_base[:, s_idx, :, :].clone()
            
            if s_idx > 0:
                if s_idx > 1:
                    encoder_x[:, :s_idx-1, :] = context_history[:, 1:s_idx, :]
                encoder_x[:, s_idx-1, :] = z_pred.squeeze(1)

            current_cos, current_sin = get_relative_pe(
                self.freqs_cos_token, 
                self.freqs_sin_token, 
                context_len=s_idx, 
                total_len=W, 
                device=device
            )

            enc_out = encoder_x
            for block in self.local_encoder:                
                # FIX: Pass the dynamic PE
                enc_out = block(enc_out, current_cos, current_sin, seq_mask)

            

            # Extract refined encoder vector
            c_indices = compress_positions[:, s_idx].view(B, 1, 1).expand(-1, 1, dim)
            z_real = torch.gather(enc_out, 1, c_indices).squeeze(1)
            all_z_real.append(z_real) # <--- SAVE IT
            # ═══════════════════════════════════════════════════════════
            # D. RECURRENCE: History ← History + [z_real]
            # ═══════════════════════════════════════════════════════════
            context_history = torch.cat([context_history, z_real.unsqueeze(1)], dim=1)
        
        final_logits = torch.stack(all_logits_list, dim=1)        
    
        # Stack the latent vectors
        final_z_pred = torch.stack(all_z_pred, dim=1) # Shape: [B, S, 1, D]
        final_z_real = torch.stack(all_z_real, dim=1) # Shape: [B, S, D]
            
        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                final_logits.view(-1, CONFIG["vocab_size"]),
                targets.view(-1),
                ignore_index=-100
            )
            latent_loss = F.mse_loss(final_z_pred.squeeze(2), final_z_real.detach())
        
            loss = lm_loss + 0.1 * latent_loss # Add weight factor

        
        return final_logits, loss
    
    # ========================================================================
    # 7. GENERATION (INFERENCE MODE)
    # ========================================================================
    @torch.no_grad()
    def generate(self, prompt="Once upon a time", max_sentences=15, max_tokens_per_sentence=47, temperature=0.8):
        """
        Generate text autoregressively.
        
        CRITICAL: During inference, encoder WAITS for decoder to finish!
        
        Flow for each sentence:
        1. Mixer predicts z_pred
        2. Decoder generates tokens autoregressively
        3. Encoder takes z_pred + generated tokens → z_real
        4. z_real added to history
        """
        device = CONFIG["device"]
        self.eval()
        
        # Tokenize prompt
        current_tokens = tokenizer.encode(prompt)
        full_text_ids = []
        context_history = self.start_context
        
        print(f"\n💬 Prompt: {prompt}")
        
        for s_idx in range(max_sentences):
            print(f"\n[Sentence {s_idx}]: ", end="", flush=True)
            
            # ═══════════════════════════════════════════════════════════
            # A. MIXER predicts next sentence vector
            # ═══════════════════════════════════════════════════════════
            mix_len = context_history.size(1)
            mix_mask = torch.triu(torch.ones(mix_len, mix_len, device=device, dtype=torch.bool), diagonal=1)
            
            mixer_history = [context_history]
            for i, block in enumerate(self.global_mixer):
                block_input = self.residual_gate(i, mixer_history)
                block_out, _ = block(block_input, self.freqs_cos_sent, self.freqs_sin_sent, mix_mask)
                mixer_history.append(block_out)
            
            mix_out = mixer_history[-1]
            z_pred = mix_out[:, -1, :].unsqueeze(1)
            
            # ═══════════════════════════════════════════════════════════
            # B. DECODER generates tokens autoregressively
            # ═══════════════════════════════════════════════════════════
            generated_tokens = []
            
            # Start with prompt tokens if first sentence
            if s_idx == 0 and current_tokens:
                generated_tokens = current_tokens.copy()
            
            while len(generated_tokens) < max_tokens_per_sentence:
                # Build input: [past_vectors, z_pred, generated_tokens_so_far]
                seq_len = s_idx + len(generated_tokens)
                if seq_len > MAX_SEQ_LEN_TOTAL:
                    break
                
                inp = torch.full((1, seq_len), PAD_TOKEN_ID, device=device)
                inp[0, s_idx:] = torch.tensor(generated_tokens, device=device)
                
                x = norm(self.wte(inp))  # Lookup normalization
                
                # Inject past vectors and z_pred
                if s_idx > 0:
                    if s_idx > 1:
                        x[:, :s_idx-1, :] = context_history[:, 1:s_idx, :]
                    x[:, s_idx-1, :] = z_pred.squeeze(1)
                
                # Decoder processes
                mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
                for block in self.local_decoder:
                    x = block(x, self.freqs_cos_token, self.freqs_sin_token, mask)
                
                logits = self.lm_head(self.ln_f(x))
                next_token_logits = logits[0, -1, :]
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                
                if next_id == COMPRESS_TOKEN_ID:
                    generated_tokens.append(next_id)
                    print(" <compress>", end="", flush=True)
                    break
                elif next_id == tokenizer.eos_token_id:
                    print(" [EOS]", end="", flush=True)
                    full_text_ids.extend(generated_tokens)
                    return tokenizer.decode(full_text_ids)
                else:
                    word = tokenizer.decode([next_id])
                    print(word, end="", flush=True)
                    generated_tokens.append(next_id)
            
            # Force compress if too long
            if generated_tokens[-1] != COMPRESS_TOKEN_ID:
                generated_tokens.append(COMPRESS_TOKEN_ID)
                print(" <compress-force>", end="", flush=True)
            
            full_text_ids.extend(generated_tokens)
            
            # ═══════════════════════════════════════════════════════════
            # C. ENCODER waits for decoder, then refines vector
            # ═══════════════════════════════════════════════════════════
            final_len = s_idx + len(generated_tokens)
            inp = torch.full((1, final_len), PAD_TOKEN_ID, device=device)
            inp[0, s_idx:] = torch.tensor(generated_tokens, device=device)
            
            x_enc = norm(self.wte(inp))
            
            if s_idx > 0:
                if s_idx > 1:
                    x_enc[:, :s_idx-1, :] = context_history[:, 1:s_idx, :]
                x_enc[:, s_idx-1, :] = z_pred.squeeze(1)
            
            mask = torch.triu(torch.ones(final_len, final_len, device=device, dtype=torch.bool), diagonal=1)
            for block in self.local_encoder:
                x_enc = block(x_enc, self.freqs_cos_token, self.freqs_sin_token, mask)
            
            local_compress = x_enc[:, -1, :]
            
            # ═══════════════════════════════════════════════════════════
            # D. UPDATE history with refined vector
            # ═══════════════════════════════════════════════════════════
            context_history = torch.cat([context_history, local_compress.unsqueeze(1)], dim=1)
            current_tokens = []  # Reset for next sentence
        
        self.train()
        return tokenizer.decode(full_text_ids)

# ============================================================================
# 8. TRAINING LOOP
# ============================================================================
def print_param_breakdown(model):
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = sum(p.numel() for p in model.wte.parameters())
    print(f"📊 Total: {total_params/1e6:.2f}M params ({embed_params/total_params:.1%} embeddings)")

torch.manual_seed(CONFIG["seed"])
model = HourglassTransformer().to(CONFIG["device"])
print_param_breakdown(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
history = {"steps": [], "train_loss": [], "val_loss": [], "ppl": []}

if CONFIG["checkpoint_path"] and os.path.exists(CONFIG["checkpoint_path"]):
    print(f"Loading checkpoint: {CONFIG['checkpoint_path']}")
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"])
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'history' in checkpoint:
        history = checkpoint['history']

print("🚀 Starting Training...")
start_time = time.time()

for step in range(CONFIG["max_iters"]):
    xb, yb, cp, ts = get_batch("train")
    if xb is None:
        break
    
    optimizer.zero_grad()
    logits, loss = model(xb, yb, cp, ts)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")
    
    if step % CONFIG["eval_interval"] == 0:
        metrics = estimate_loss(model)
        ppl = math.exp(metrics["val"])
        print(f"--- Eval Step {step}: Train {metrics['train']:.3f} | Val {metrics['val']:.3f} | PPL {ppl:.2f}")
        
        history["steps"].append(step)
        history["train_loss"].append(metrics["train"])
        history["val_loss"].append(metrics["val"])
        history["ppl"].append(ppl)
        
        # Generate sample
        if step > 0 and step % 100 == 0:
            print("\n🎨 Sample Generation:")
            model.generate("Once upon a time", max_sentences=3)
            print("\n")
        
        # Save checkpoint
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': CONFIG,
            'history': history,
        }, CONFIG["save_path"])

print(f"✅ Training Complete in {time.time()-start_time:.2f}s")

# Final generation
print("\n🎨 Final Generation:")
model.generate("Once upon a time", max_sentences=10)
print("\n")