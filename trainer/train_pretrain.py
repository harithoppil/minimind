import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from model.tempmodel import TempModelConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)
from trainer.logicgrad import LogicGrad, create_dual_optimizer

warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            
            # Step LogicGrad if using dual optimizers
            # LogicGrad doesn't use scaler (manages its own precision)
            if optimizer_logic is not None:
                optimizer_logic.step()
                optimizer_logic.zero_grad(set_to_none=True)

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "logits_loss": current_logits_loss,
                        "aux_loss": current_aux_loss,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="checkpoints",
            )
            model.train()
            del state_dict

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="out", help="Model save directory")
    parser.add_argument(
        "--save_weight", default="pretrain", type=str, help="Prefix for saved weight files"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs (recommend 1 for zero or 2-6 for full training)"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=8, help="Data loader worker threads")
    parser.add_argument(
        "--accumulation_steps", type=int, default=8, help="Gradient accumulation steps"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model save interval")
    parser.add_argument("--hidden_size", default=512, type=int, help="Hidden layer dimension")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="Number of hidden layers")
    parser.add_argument("--num_attention_heads", default=4, type=int, help="Number of attention heads")
    parser.add_argument(
        "--max_seq_len",
        default=340,
        type=int,
        help="Maximum truncation length for training (Chinese: 1 token ≈ 1.5-1.7 characters)",
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="Whether to use MoE architecture (0=no, 1=yes)",
    )
    parser.add_argument(
        "--use_tempmodel",
        default=0,
        type=int,
        choices=[0, 1],
        help="Whether to use TempModel architecture (0=no MiniMind, 1=yes TempModel)",
    )
    parser.add_argument(
        "--n_routed_experts",
        default=4,
        type=int,
        help="Number of routed experts for TempModel",
    )
    parser.add_argument(
        "--n_shared_experts",
        default=1,
        type=int,
        help="Number of shared experts for TempModel",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        default=2,
        type=int,
        help="Number of experts per token for TempModel",
    )
    parser.add_argument(
        "--v_head_expansion",
        default=2,
        type=int,
        help="Expansion factor for V-head MLP (default 2)",
    )
    parser.add_argument(
        "--tempmodel_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="Whether TempModel uses MoE attention experts (0=dense, 1=MoE)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/pretrain_hq.jsonl",
        help="Pretraining data path",
    )
    parser.add_argument(
        "--from_weight",
        default="none",
        type=str,
        help="Base weight to train from, 'none' to train from scratch",
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
        "--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb project name"
    )
    parser.add_argument(
        "--use_compile",
        default=0,
        type=int,
        choices=[0, 1],
        help="Whether to use torch.compile acceleration (0=no, 1=yes)",
    )
    parser.add_argument(
        "--use_logicgrad",
        default=0,
        type=int,
        choices=[0, 1],
        help="Use LogicGrad optimizer for W_bilinear matrices (0=AdamW only, 1=dual optimizer)",
    )
    parser.add_argument(
        "--logic_lr",
        default=0.05,
        type=float,
        help="Learning rate for LogicGrad optimizer (only used if --use_logicgrad=1)",
    )
    args = parser.parse_args()

    # ========== 1. Initialize environment and random seed ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. Configure directories, model parameters, check checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    if args.use_tempmodel:
        lm_config = TempModelConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            use_moe=bool(args.tempmodel_moe),
            n_routed_experts=args.n_routed_experts,
            n_shared_experts=args.n_shared_experts,
            num_experts_per_tok=args.num_experts_per_tok,
            v_head_expansion=args.v_head_expansion,
        )
        mode_str = "MoE" if args.tempmodel_moe else "Dense"
        Logger(f"Using TempModel ({mode_str}) - hidden={args.hidden_size}, layers={args.num_hidden_layers}")
    else:
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
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. Define model, data, optimizer ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # ========== 6. Resume model weights from checkpoint (BEFORE compile!) ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=True)
        Logger("Resumed model weights from checkpoint")
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
    
    # Compile AFTER loading checkpoint
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    
    # Create optimizer(s) based on --use_logicgrad flag
    if args.use_logicgrad == 1 and args.use_tempmodel == 1:
        optimizer_logic, optimizer = create_dual_optimizer(
            model, adam_lr=args.learning_rate, logic_lr=args.logic_lr
        )
        Logger(f"Using dual optimizer: LogicGrad (lr={args.logic_lr}) + AdamW (lr={args.learning_rate})")
    else:
        optimizer_logic = None
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Resume optimizer/scaler state (after optimizer is created)
    if ckp_data:
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        # Note: LogicGrad state is not saved/restored (momentum buffer)

    # ========== 7. Wrap model with DDP ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

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
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. Cleanup distributed processes ==========
    if dist.is_initialized():
        dist.destroy_process_group()
