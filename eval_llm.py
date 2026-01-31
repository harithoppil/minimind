import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    # Use gpt2 tokenizer matching training
    if args.load_from == 'model':
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.load_from)
        
    special_tokens_dict = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Set ChatML template
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling,
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        state_dict = torch.load(ckp, map_location=args.device)
        # Handle potential vocab size mismatch if loading old checkpoint
        if state_dict['lm_head.weight'].shape[0] != model.lm_head.weight.shape[0]:
            print(f"Warning: Vocab size mismatch. Loaded: {state_dict['lm_head.weight'].shape[0]}, Expected: {model.lm_head.weight.shape[0]}")
            # Filter out mismatching layers for partial load
            state_dict = {k: v for k, v in state_dict.items() if v.shape == model.state_dict()[k].shape}
            
        model.load_state_dict(state_dict, strict=False)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        
    # Resize embeddings if needed (for AutoModel cases or partial loads)
    model.resize_token_embeddings(len(tokenizer))
    
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind Model Inference and Chat")
    parser.add_argument('--load_from', default='model', type=str, help="Model load path (model=native torch weights, other path=transformers format)")
    parser.add_argument('--save_dir', default='out', type=str, help="Model weights directory")
    parser.add_argument('--weight', default='full_sft', type=str, help="Weight name prefix (pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo)")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA weight name (None means not used, options: lora_identity, lora_medical)")
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden layer dimension (512=Small-26M, 640=MoE-145M, 768=Base-104M)")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers (Small/MoE=8, Base=16)")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Whether to use MoE architecture (0=no, 1=yes)")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="Enable RoPE position encoding extrapolation (4x, only solves position encoding problem)")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="Maximum generation length (note: this is not the actual long text capability)")
    parser.add_argument('--temperature', default=0.85, type=float, help="Generation temperature, controls randomness (0-1, higher is more random)")
    parser.add_argument('--top_p', default=0.85, type=float, help="Nucleus sampling threshold (0-1)")
    parser.add_argument('--historys', default=0, type=int, help="Number of historical dialogue turns to carry (must be even, 0 means no history)")
    parser.add_argument('--show_speed', default=1, type=int, help="Display decode speed (tokens/s)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Running device")
    args = parser.parse_args()
    
    prompts = [
        'What are your special skills?',
        'Why is the sky blue?',
        'Please write a Python function to calculate the Fibonacci sequence',
        'Explain the basic process of "photosynthesis"',
        'If it rains tomorrow, how should I go out?',
        'Compare the pros and cons of cats and dogs as pets',
        'Explain what machine learning is',
        'Recommend some Chinese cuisine'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] Auto test\n[1] Manual input\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(2026) # or setup_seed(random.randint(0, 2048))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason': templates["enable_thinking"] = True # Only used for Reason model
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()