"""
OLMo-3 32B Fine-Tuning Script — OOM FIX v6

ROOT CAUSE: accelerate 1.5.2 TRANSFORMER_BASED_WRAP silently fails to
resolve Olmo3DecoderLayer, so the entire 32B model becomes ONE FSDP unit.
During forward, FSDP all-gathers ALL 64GB of params onto each GPU → OOM.

FIX: YAML now uses SIZE_BASED_WRAP + CPU offloading.
This script adds FSDP unit counting to verify per-layer wrapping works.
"""

import argparse
import csv
import gc
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class MalwareAnalysisDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                self.samples.append(record)
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]
        conversations = record["conversations"]
        system_msg = conversations[0]["content"]
        user_msg = conversations[1]["content"]
        assistant_msg = conversations[2]["content"]

        if "<binary>" in user_msg and "</binary>" in user_msg:
            start = user_msg.index("<binary>") + len("<binary>")
            end = user_msg.index("</binary>")
            binary_hex = user_msg[start:end]
            question = user_msg[end + len("</binary>"):]
            reserved_chars = (self.max_length - 200) * 2
            if len(binary_hex) > reserved_chars:
                binary_hex = binary_hex[:reserved_chars]
            user_msg = f"<binary>{binary_hex}</binary>{question}"

        full_text = (
            f"<|system|>\n{system_msg}\n"
            f"<|user|>\n{user_msg}\n"
            f"<|assistant|>\n{assistant_msg}<|endoftext|>"
        )

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class MetricsTracker:
    def __init__(self, output_dir: str, gpu_type: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_type = gpu_type
        self.step_metrics = []
        self.start_time = time.time()
        self.csv_path = self.output_dir / f"training_metrics_{gpu_type}.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "step", "epoch", "loss", "learning_rate",
            "tokens_per_sec", "samples_per_sec",
            "step_time_ms", "forward_time_ms", "backward_time_ms",
            "gpu_mem_allocated_gb", "gpu_mem_reserved_gb", "gpu_mem_peak_gb",
            "wall_clock_sec",
        ])

    def log_step(self, metrics: dict):
        self.step_metrics.append(metrics)
        self.csv_writer.writerow([
            metrics.get("step", 0), metrics.get("epoch", 0),
            metrics.get("loss", 0.0), metrics.get("learning_rate", 0.0),
            metrics.get("tokens_per_sec", 0.0), metrics.get("samples_per_sec", 0.0),
            metrics.get("step_time_ms", 0.0), metrics.get("forward_time_ms", 0.0),
            metrics.get("backward_time_ms", 0.0),
            metrics.get("gpu_mem_allocated_gb", 0.0), metrics.get("gpu_mem_reserved_gb", 0.0),
            metrics.get("gpu_mem_peak_gb", 0.0), metrics.get("wall_clock_sec", 0.0),
        ])
        self.csv_file.flush()

    def save_summary(self, total_steps: int, total_tokens: int):
        elapsed = time.time() - self.start_time
        losses = [m["loss"] for m in self.step_metrics if "loss" in m]
        step_times = [m["step_time_ms"] for m in self.step_metrics if "step_time_ms" in m]
        tokens_per_sec = [m["tokens_per_sec"] for m in self.step_metrics if "tokens_per_sec" in m]
        summary = {
            "gpu_type": self.gpu_type,
            "timestamp": datetime.utcnow().isoformat(),
            "total_steps": total_steps,
            "total_tokens_processed": total_tokens,
            "total_wall_time_sec": elapsed,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "final_loss": losses[-1] if losses else 0,
            "avg_step_time_ms": sum(step_times) / len(step_times) if step_times else 0,
            "avg_tokens_per_sec_per_gpu": sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0,
            "peak_gpu_mem_gb": max((m.get("gpu_mem_peak_gb", 0) for m in self.step_metrics), default=0),
        }
        summary_path = self.output_dir / f"training_summary_{self.gpu_type}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
        return summary

    def close(self):
        self.csv_file.close()


def get_gpu_memory_stats():
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_mem_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "gpu_mem_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def log_memory(tag, is_main):
    if is_main and torch.cuda.is_available():
        m = get_gpu_memory_stats()
        logger.info(f"[MEM {tag}] allocated={m['gpu_mem_allocated_gb']}GB "
                     f"reserved={m['gpu_mem_reserved_gb']}GB peak={m['gpu_mem_peak_gb']}GB")


def count_fsdp_units(model):
    """Count how many FSDP-wrapped modules exist in the model."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    count = 0
    for module in model.modules():
        if isinstance(module, FSDP):
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="OLMo-3 32B Fine-Tuning with FSDP")
    parser.add_argument("--model_name", type=str, default="allenai/Olmo-3-1125-32B")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/home/outputs/olmo3-finetune")
    parser.add_argument("--gpu_type", type=str, required=True, choices=["h100", "b200"])
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--profile_steps", type=int, default=5)
    parser.add_argument("--profile_start_step", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None,
        mixed_precision="bf16",
    )

    set_seed(args.seed)
    is_main = accelerator.is_main_process

    if is_main:
        logger.info("=" * 70)
        logger.info(f"OLMo-3 32B Fine-Tuning — GPU Type: {args.gpu_type.upper()} — OOM FIX v6")
        logger.info(f"  Model:        {args.model_name}")
        logger.info(f"  Data:         {args.data_path}")
        logger.info(f"  Max Length:   {args.max_length}")
        logger.info(f"  Batch Size:   {args.batch_size} (per device)")
        logger.info(f"  Grad Accum:   {args.gradient_accumulation_steps}")
        logger.info(f"  Effective BS: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
        logger.info(f"  Epochs:       {args.num_epochs}")
        logger.info(f"  LR:           {args.learning_rate}")
        logger.info(f"  Num GPUs:     {accelerator.num_processes}")
        logger.info(f"  Profiling:    {args.profile_steps} steps starting at step {args.profile_start_step}")
        logger.info("=" * 70)

    # ---- Tokenizer ----
    if is_main:
        logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model ----
    if is_main:
        logger.info("Loading model...")

    log_memory("before-model-load", is_main)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_cache=False,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model loaded: {total_params / 1e9:.1f}B total, {trainable_params / 1e9:.1f}B trainable")

    gc.collect()
    torch.cuda.empty_cache()
    log_memory("after-model-load-cleanup", is_main)

    # ---- Dataset ----
    if is_main:
        logger.info("Loading dataset...")
    dataset = MalwareAnalysisDataset(args.data_path, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1,
    )

    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )

    if is_main:
        logger.info(f"Total training steps: {total_steps}")

    # ---- FSDP wrapping via accelerate ----
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    gc.collect()
    torch.cuda.empty_cache()
    log_memory("after-fsdp-wrap", is_main)

    # ---- DIAGNOSTIC: Count FSDP units ----
    # If this shows 1 → entire model is one unit (BAD, causes OOM)
    # If this shows 65+ → per-layer wrapping works (GOOD)
    fsdp_units = count_fsdp_units(model)
    if is_main:
        logger.info(f"[DIAGNOSTIC] FSDP units in model: {fsdp_units}")
        if fsdp_units <= 2:
            logger.warning("WARNING: Only {fsdp_units} FSDP units! Per-layer wrapping may have failed!")
            logger.warning("The entire model may be all-gathered at once, using ~64GB GPU memory.")
        else:
            logger.info(f"Per-layer FSDP wrapping confirmed ({fsdp_units} units)")

    log_memory("before-training", is_main)

    # ---- Metrics ----
    metrics_tracker = MetricsTracker(args.output_dir, args.gpu_type) if is_main else None

    # ---- PyTorch Profiler ----
    profiler = None
    prof_dir = os.path.join(args.output_dir, f"profiler_traces_{args.gpu_type}")
    if args.profile_steps > 0 and is_main:
        os.makedirs(prof_dir, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=max(0, args.profile_start_step - 1), warmup=1,
                active=args.profile_steps, repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_dir),
            record_shapes=True, profile_memory=True, with_stack=True, with_flops=True,
        )

    if is_main:
        torch.cuda.reset_peak_memory_stats()

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    global_step = 0
    total_tokens = 0
    model.train()

    if profiler:
        profiler.start()

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()

            fwd_start = time.time()
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                fwd_time = (time.time() - fwd_start) * 1000

                bwd_start = time.time()
                accelerator.backward(loss)
                bwd_time = (time.time() - bwd_start) * 1000

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step_time = (time.time() - step_start) * 1000
            num_tokens = batch["attention_mask"].sum().item()
            total_tokens += num_tokens

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                epoch_steps += 1
                loss_val = accelerator.gather(loss.detach()).mean().item()
                epoch_loss += loss_val

                if is_main and global_step % args.log_every == 0:
                    mem = get_gpu_memory_stats()
                    tokens_per_sec = num_tokens / (step_time / 1000) if step_time > 0 else 0
                    samples_per_sec = args.batch_size / (step_time / 1000) if step_time > 0 else 0

                    step_metrics = {
                        "step": global_step, "epoch": epoch,
                        "loss": round(loss_val, 6),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "tokens_per_sec": round(tokens_per_sec, 1),
                        "samples_per_sec": round(samples_per_sec, 4),
                        "step_time_ms": round(step_time, 1),
                        "forward_time_ms": round(fwd_time, 1),
                        "backward_time_ms": round(bwd_time, 1),
                        "wall_clock_sec": round(time.time() - metrics_tracker.start_time, 1),
                        **mem,
                    }
                    metrics_tracker.log_step(step_metrics)

                    logger.info(
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {loss_val:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Tok/s: {tokens_per_sec:.0f} | "
                        f"Step: {step_time:.0f}ms (fwd:{fwd_time:.0f} bwd:{bwd_time:.0f}) | "
                        f"Mem: {mem['gpu_mem_allocated_gb']:.1f}/{mem['gpu_mem_peak_gb']:.1f}GB"
                    )

                if is_main and global_step % args.save_every == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(ckpt_dir)
                    logger.info(f"Checkpoint saved to {ckpt_dir}")

            if profiler:
                profiler.step()

            if args.max_steps > 0 and global_step >= args.max_steps:
                if is_main:
                    logger.info(f"Reached --max_steps={args.max_steps}, stopping.")
                break

        if is_main and epoch_steps > 0:
            logger.info(f"Epoch {epoch} complete | Avg Loss: {epoch_loss / epoch_steps:.4f}")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # ============================================================
    # CLEANUP
    # ============================================================
    if profiler:
        profiler.stop()
        if is_main:
            logger.info(f"Profiler traces saved to {prof_dir}/")

    if is_main:
        logger.info("Saving final model...")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if is_main:
        unwrapped_model.save_pretrained(
            os.path.join(args.output_dir, "final_model"),
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))

    if is_main:
        summary = metrics_tracker.save_summary(global_step, total_tokens)
        metrics_tracker.close()
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  GPU Type:              {args.gpu_type.upper()}")
        logger.info(f"  Total Steps:           {global_step}")
        logger.info(f"  Total Tokens:          {total_tokens:,}")
        logger.info(f"  Total Time:            {summary['total_wall_time_sec']:.1f}s")
        logger.info(f"  Avg Tokens/sec/GPU:    {summary['avg_tokens_per_sec_per_gpu']:.1f}")
        logger.info(f"  Avg Step Time:         {summary['avg_step_time_ms']:.1f}ms")
        logger.info(f"  Final Loss:            {summary['final_loss']:.4f}")
        logger.info(f"  Peak GPU Memory:       {summary['peak_gpu_mem_gb']:.2f}GB")
        logger.info(f"  Profiler Traces:       {prof_dir}/")
        logger.info(f"  Metrics CSV:           {metrics_tracker.csv_path}")
        logger.info("=" * 70)

    accelerator.end_training()


if __name__ == "__main__":
    main()
