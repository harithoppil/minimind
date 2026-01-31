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

MODEL_PATH = os.path.join(REPO_DIR, "checkpoints", "tempmodel.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(REPO_DIR, "out", "tempmodel.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/content/out/tempmodel.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIG = {
    "hidden_size": 768,
    "num_hidden_layers": 16,
    "num_attention_heads": 12,
    "num_key_value_heads": 4,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
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
        vocab_size=vocab_size,
    )

    model = TempModelForCausalLM(config).to(device)

    if model.model.embed_tokens.num_embeddings != vocab_size:
        model.model.embed_tokens = torch.nn.Embedding(vocab_size, MODEL_CONFIG["hidden_size"]).to(device)
        model.lm_head = torch.nn.Linear(MODEL_CONFIG["hidden_size"], vocab_size, bias=False).to(device)
        model.model.embed_tokens.weight = model.lm_head.weight

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print("Model loaded successfully!")
    else:
        print(f"No checkpoint found at {model_path}")
        print("Running with randomly initialized weights (for testing)")

    return model


def run_chat():
    print("=" * 50)
    print("TempModel Inference")
    print("=" * 50)

    print("\nLoading GPT-2 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Apply the same patches as training
    IM_START = "\u003c|im_start|\u003e"
    IM_END = "\u003c|im_end|\u003e"
    EOT = "\u003c|endoftext|\u003e"

    special_tokens_dict = {
        "bos_token": IM_START,
        "eos_token": IM_END,
        "pad_token": EOT,
        "additional_special_tokens": [IM_START, IM_END]
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # ChatML Template
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'" + IM_START + "' + message['role'] + '\\n' + message['content'] + '" + IM_END + "' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '" + IM_START + "assistant\\n' }}"
        "{% endif %}"
    )

    vocab_size = len(tokenizer)
    print(f"   Vocab Size: {vocab_size}")

    model = load_model(MODEL_PATH, DEVICE, vocab_size)
    model.eval()

    # --- SANITY CHECK ---
    print("\nRunning Sanity Check...")
    test_prompt = "hi"
    print(f"INPUT: {test_prompt}")

    inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

    response = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"OUTPUT: {response}\n")
    print("-" * 40)

    # Interactive Loop
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
