import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
    init_model,
)

warnings.filterwarnings("ignore")


# Custom Critic model, inherits from MiniMindLM
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # Replace lm_head with a linear layer that outputs a single value
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Use base model to get hidden states
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        hidden_states = self.model.norm(outputs[0])
        # Use value_head to get value estimate
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """Integrate all reward functions to calculate total reward"""

    def reasoning_model_reward(rewards):
        # 1. Format reward (only used when training reasoning model)
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [
            re.match(pattern2, response, re.S) for response in responses
        ]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 2. Mark reward (to prevent strict reward sparsity, only used when training reasoning model)
        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)

    # Format reward
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # Use reward model to calculate reward for entire response
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [
                {"role": role, "content": content.strip()} for role, content in matches
            ]

            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            scale = 3.0
            score = max(min(score, scale), -scale)

            # When args.reasoning=1, also calculate reward for <answer> content
            if args.reasoning == 1:
                answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # Calculate reward separately for answer content
                    tmp_chat = messages + [
                        {"role": "assistant", "content": answer_content}
                    ]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(
    epoch,
    loader,
    iters,
    old_actor_model,
    ref_model,
    actor_scheduler,
    critic_scheduler,
    reward_model,
    reward_tokenizer,
    start_step=0,
    wandb=None,
):
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            padding_side="left",
        ).to(
            args.device
        )  # input_ids: [B, P], attention_mask: [B, P]
        prompt_length = enc.input_ids.shape[1]

        with torch.no_grad():
            # DDP model needs to use .module to access generate method
            model_for_gen = (
                actor_model.module
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model
            )
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )  # [B, P+R]

        responses_text = [
            tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True)
            for i in range(len(prompts))
        ]
        rewards = calculate_rewards(
            prompts, responses_text, reward_model, reward_tokenizer
        )  # [B]

        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        values_seq = critic_model(
            input_ids=gen_out, attention_mask=full_mask
        )  # [B, P+R]
        last_indices = (
            full_mask * torch.arange(full_mask.size(1), device=gen_out.device)
        ).argmax(dim=1)
        values = values_seq[
            torch.arange(values_seq.size(0), device=values_seq.device), last_indices
        ]  # [B]
        advantages = rewards - values.detach()  # [B]

        with autocast_ctx:
            res = actor_model(input_ids=gen_out, attention_mask=full_mask)
            logits = res.logits  # [B, P+R, V]
            aux_loss = (
                res.aux_loss
                if lm_config.use_moe
                else torch.tensor(0.0, device=args.device)
            )

        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        logp_tokens = (
            F.log_softmax(logits[:, :-1], dim=-1)
            .gather(2, labels.unsqueeze(-1))
            .squeeze(-1)
        )  # [B, P+R-1]
        seq_len = gen_out.size(1) - 1
        resp_mask = (
            torch.arange(seq_len, device=gen_out.device).unsqueeze(0)
            >= prompt_length - 1
        )
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

        with torch.no_grad():
            old_logits = old_actor_model(
                input_ids=gen_out, attention_mask=full_mask
            ).logits  # [B, P+R, V]
            old_logp_tokens = (
                F.log_softmax(old_logits[:, :-1], dim=-1)
                .gather(2, labels.unsqueeze(-1))
                .squeeze(-1)
            )  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]

            ref_logits = ref_model(
                input_ids=gen_out, attention_mask=full_mask
            ).logits  # [B, P+R, V]
            ref_logp_tokens = (
                F.log_softmax(ref_logits[:, :-1], dim=-1)
                .gather(2, labels.unsqueeze(-1))
                .squeeze(-1)
            )  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        kl = (actor_logp - old_logp).mean()  # scalar
        kl_ref = (actor_logp - ref_logp).mean()  # scalar
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        surr1 = ratio * advantages  # [B]
        surr2 = (
            torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
            * advantages
        )  # [B]
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar
        value_loss = F.mse_loss(values, rewards)  # scalar
        loss = (
            policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss
        ) / args.accumulation_steps  # scalar
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        if is_main_process():
            response_ids = gen_out[:, enc.input_ids.shape[1] :]
            is_eos = response_ids == tokenizer.eos_token_id
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(
                has_eos,
                eos_indices + 1,
                torch.tensor(response_ids.shape[1], device=is_eos.device),
            )
            avg_len = lengths.float().mean()

            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            current_aux_loss = aux_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]["lr"]
            critic_lr = critic_optimizer.param_groups[0]["lr"]

            if wandb is not None:
                wandb.log(
                    {
                        "actor_loss": actor_loss_val,
                        "critic_loss": critic_loss_val,
                        "aux_loss": current_aux_loss,
                        "reward": reward_val,
                        "kl": kl_val,
                        "kl_ref": kl_ref_val,
                        "avg_response_len": avg_len_val,
                        "actor_lr": actor_lr,
                    }
                )

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, "
                f"Reward: {reward_val:.4f}, KL: {kl_val:.4f}, KL_ref: {kl_ref_val:.4f}, "
                f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}"
            )

        if (step + 1) % args.update_old_actor_freq == 0:
            raw_actor = (
                actor_model.module
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model
            )
            raw_actor = getattr(raw_actor, "_orig_mod", raw_actor)
            state_dict = raw_actor.state_dict()
            old_actor_model.load_state_dict(
                {k: v.detach().cpu() for k, v in state_dict.items()}
            )
            old_actor_model.to(args.device)

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            raw_actor = (
                actor_model.module
                if isinstance(actor_model, DistributedDataParallel)
                else actor_model
            )
            raw_actor = getattr(raw_actor, "_orig_mod", raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)

            # Use lm_checkpoint to save complete state (including critic)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=actor_model,
                optimizer=actor_optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="checkpoints",
                scheduler=actor_scheduler,
                critic_model=critic_model,
                critic_optimizer=critic_optimizer,
                critic_scheduler=critic_scheduler,
            )
            actor_model.train()
            del actor_state

        del (
            enc,
            gen_out,
            responses_text,
            rewards,
            full_mask,
            values_seq,
            values,
            advantages,
        )
        del (
            logits,
            labels,
            logp_tokens,
            final_mask,
            actor_logp,
            old_logits,
            old_logp,
            ref_logits,
            ref_logp,
        )
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MiniMind PPO (Proximal Policy Optimization)"
    )
    parser.add_argument("--save_dir", type=str, default="out", help="Model save directory")
    parser.add_argument(
        "--save_weight", default="ppo_actor", type=str, help="Prefix for saved weight files"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor learning rate")
    parser.add_argument(
        "--critic_learning_rate", type=float, default=8e-8, help="Critic learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=8, help="Data loader worker threads")
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--log_interval", type=int, default=1, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=10, help="Model save interval")
    parser.add_argument("--hidden_size", default=512, type=int, help="Hidden layer dimension")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="Number of hidden layers")
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="Whether to use MoE architecture (0=no, 1=yes)",
    )
    parser.add_argument("--max_seq_len", default=66, type=int, help="Maximum prompt length")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="Maximum generation length")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/rlaif-mini.jsonl",
        help="RLAIF data path",
    )
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO clipping parameter")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL divergence penalty coefficient")
    parser.add_argument(
        "--reasoning",
        type=int,
        default=1,
        choices=[0, 1],
        help="Reasoning model type (0=regular model, 1=reasoning model)",
    )
    parser.add_argument(
        "--update_old_actor_freq", type=int, default=4, help="Frequency to update old_actor_model"
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="../internlm2-1_8b-reward",
        help="Reward model path",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="Whether to auto-detect and resume training (0=no, 1=yes)",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="MiniMind-PPO", help="wandb project name"
    )
    parser.add_argument(
        "--use_compile",
        default=0,
        type=int,
        choices=[0, 1],
        help="Whether to use torch.compile acceleration (0=no, 1=yes)",
    )
    args = parser.parse_args()

    # ========== 1. Initialize environment and random seed ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. Configure directories, model parameters, check checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="checkpoints")
        if args.from_resume == 1
        else None
    )

    # ========== 3. Set up mixed precision ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. Configure wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. Initialize models and data ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    # Actor model
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger("torch.compile enabled")
    # Old Actor model
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    # Reference model
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Critic model
    moe_suffix = "_moe" if lm_config.use_moe else ""
    ckp = f"{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    # Reward model
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_path, trust_remote_code=True
    )
    # Data and optimizers
    train_ds = RLAIFDataset(
        args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len)
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(
        critic_model.parameters(), lr=args.critic_learning_rate
    )
    loader_for_count = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler
    )
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(
        actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10
    )
    critic_scheduler = CosineAnnealingLR(
        critic_optimizer,
        T_max=total_optimizer_steps,
        eta_min=args.critic_learning_rate / 10,
    )

    # ========== 6. Resume state from checkpoint ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data["model"])
        critic_model.load_state_dict(ckp_data["critic_model"])
        actor_optimizer.load_state_dict(ckp_data["optimizer"])
        critic_optimizer.load_state_dict(ckp_data["critic_optimizer"])
        actor_scheduler.load_state_dict(ckp_data["scheduler"])
        critic_scheduler.load_state_dict(ckp_data["critic_scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ========== 7. Wrap model with DDP ==========
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)

    # ========== 8. Start training ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(
            train_sampler or indices, args.batch_size, skip
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if skip > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: Skipping first {start_step} steps, starting from step {start_step + 1}"
            )
            ppo_train_epoch(
                epoch,
                loader,
                len(loader) + skip,
                old_actor_model,
                ref_model,
                actor_scheduler,
                critic_scheduler,
                reward_model,
                reward_tokenizer,
                start_step,
                wandb,
            )
        else:
            ppo_train_epoch(
                epoch,
                loader,
                len(loader),
                old_actor_model,
                ref_model,
                actor_scheduler,
                critic_scheduler,
                reward_model,
                reward_tokenizer,
                0,
                wandb,
            )

    # ========== 9. Cleanup distributed processes ==========
    if dist.is_initialized():
        dist.destroy_process_group()
