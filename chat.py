import sys
import os
import torch
from transformers import AutoTokenizer

# --- CONFIGURATION ---
# 1. Path to your Repo (Assumes running from repo root or /content)
REPO_DIR = (
    os.path.abspath("minimind") if os.path.exists("minimind") else os.path.abspath(".")
)

# 2. Path to the FINAL weights (Try /out first)
MODEL_PATH = os.path.join(REPO_DIR, "checkpoints", "sft_minimind2_104m_768.pth")
if not os.path.exists(MODEL_PATH):
    # Fallback to current directory or ../out
    MODEL_PATH = "sft_minimind2_104m_768.pth"

# 3. Device
DEVICE = "mps" if torch.cuda.is_available() else "cpu"

# ---------------------
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)
try:
    from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
except ImportError:
    sys.path.append(".")
    from model.model_minimind import MiniMindForCausalLM, MiniMindConfig


def run_chat():
    print(f"🔧 Loading GPT-2 Tokenizer (and patching)...")
    # ALWAYS load GPT-2 base to ensure vocab matches your checkpoint
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Apply the EXACT same patches as training
    special_tokens_dict = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # ChatML Template
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

    vocab_size = len(tokenizer)  # Should be 50259
    print(f"   Vocab Size: {vocab_size}")

    print(f"⚖️  Loading Weights from {MODEL_PATH} ...")
    config = MiniMindConfig(
        hidden_size=768, num_hidden_layers=16, vocab_size=vocab_size, use_moe=False
    )
    model = MiniMindForCausalLM(config).to(DEVICE)

    # Resize embeddings to match 50259
    model.model.embed_tokens = torch.nn.Embedding(vocab_size, 768).to(DEVICE)
    model.lm_head = torch.nn.Linear(768, vocab_size, bias=False).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        # strict=False is safer for manual resizing
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Error: File not found: {MODEL_PATH}")
        return

    model.eval()

    # --- SANITY CHECK ---
    print("\n🔎 Running Sanity Check...")
    # Pretrained models might not follow instructions well, so we test completion
    test_prompt = "hi"
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
        )

    response = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"OUTPUT: {response}\n")
    print("-" * 40)

    # Interactive Loop
    while True:
        prompt = input("👤 User (type 'exit' to quit): ")
        if prompt.lower() in ["exit", "quit"]:
            break

        # Note: Pretrained base models are NOT chat models.
        # Using raw prompt completion usually works better than chat templates until SFT.
        # But we use the chat template here since that's your goal.
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        response = tokenizer.decode(
            out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"🤖 MiniMind: {response}")


if __name__ == "__main__":
    run_chat()
