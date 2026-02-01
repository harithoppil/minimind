"""
TempModel Inference Script

Standalone inference script for TempModel.
Can be run directly or copied into a notebook cell.

Usage:
    python inference.py
"""

import sys
import os
import torch
from transformers import AutoTokenizer

# --- CONFIGURATION ---
REPO_DIR = os.path.abspath("minimind") if os.path.exists("minimind") else os.path.abspath(".")

# Try multiple checkpoint paths
MODEL_PATHS = [
   
   
     os.path.join(REPO_DIR, "sft_minimind2_104m_tempmodel_512.pth"),  # SFT checkpoint
      os.path.join(REPO_DIR, "checkpoints", "pretrain_minimind2_104m_512_resume.pth"),  # Pretrain checkpoint
    os.path.join(REPO_DIR, "checkpoints", "tempmodel.pth"),
    os.path.join(REPO_DIR, "out", "tempmodel.pth"),
    "/content/out/tempmodel.pth",
]

MODEL_PATH = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Model config matching training: hidden_size=512, num_hidden_layers=6
MODEL_CONFIG = {
    "hidden_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "v_head_expansion": 2,  # Must match training checkpoint!
}

if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

try:
    from model.tempmodel import TempModelForCausalLM, TempModelConfig
except ImportError:
    sys.path.append(".")
    from model.tempmodel import TempModelForCausalLM, TempModelConfig


def load_model(model_path, device, vocab_size):
    """Load TempModel from checkpoint."""
    print(f"Loading Weights from {model_path} ...")

    config = TempModelConfig(
        hidden_size=MODEL_CONFIG["hidden_size"],
        num_hidden_layers=MODEL_CONFIG["num_hidden_layers"],
        num_attention_heads=MODEL_CONFIG["num_attention_heads"],
        num_key_value_heads=MODEL_CONFIG["num_key_value_heads"],
        n_routed_experts=MODEL_CONFIG["n_routed_experts"],
        n_shared_experts=MODEL_CONFIG["n_shared_experts"],
        num_experts_per_tok=MODEL_CONFIG["num_experts_per_tok"],
        v_head_expansion=MODEL_CONFIG["v_head_expansion"],
        vocab_size=vocab_size,
    )

    model = TempModelForCausalLM(config).to(device)


    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle training checkpoint format (has 'model' key) vs raw state_dict
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
            print(f"  Loaded from training checkpoint (epoch {checkpoint.get('epoch', '?')}, step {checkpoint.get('step', '?')})")
        else:
            state_dict = checkpoint
        
        # Convert float16 weights to float32 if needed
        state_dict = {k: v.float() if v.dtype == torch.float16 else v for k, v in state_dict.items()}
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Warning: Unexpected keys: {len(unexpected)}")
        print("Model loaded successfully!")
    else:
        print(f"No checkpoint found at {model_path}")
        print("Running with randomly initialized weights (for testing)")

    return model


def run_chat():
    print("=" * 50)
    print("TempModel Inference")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Model Path: {MODEL_PATH}")

    print("\nLoading GPT-2 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Apply the same patches as training
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    EOT = "<|endoftext|>"
    special_tokens_dict = {
        "bos_token": IM_START,
        "eos_token": IM_END,
        "pad_token": EOT,
        "additional_special_tokens": [IM_START, IM_END]
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

    vocab_size = len(tokenizer)
    print(f"   Vocab Size: {vocab_size}")

    model = load_model(MODEL_PATH, DEVICE, vocab_size)
    model.eval()

    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    print("\nRunning Sanity Check...")
    test_prompt = "write about how a law firm can use data to help clients"
    print(f"INPUT: {test_prompt}")

    inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.3
        )

    response = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"OUTPUT: {response}\n")
    print("-" * 40)

    while True:
        prompt = input("User (type 'exit' to quit): ")
        if prompt.lower() in ["exit", "quit"]:
            break

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=1.0,
                top_p=0.9,
                do_sample=True
            )
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"TempModel: {response}")


if __name__ == "__main__":
    run_chat()