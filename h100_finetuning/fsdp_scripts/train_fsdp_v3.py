"""
OLMo-3 32B Fine-Tuning — Native PyTorch FSDP + TCPXO (v3)

No DeepSpeed. No Accelerate. No NeMo.
Just PyTorch FSDP + transformers for model loading.
"""

import argparse
import functools
import json
import logging
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def log_memory(tag, device=0):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        logger.info(f"[MEM {tag}] allocated={alloc:.1f}GB reserved={reserved:.1f}GB")


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_decoder_layer_class(model):
    """Dynamically find the transformer decoder layer class."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is not None and len(layers) > 0:
        cls = type(layers[0])
        if is_main():
            logger.info(f"Detected decoder layer class: {cls.__name__}")
        return cls
    raise RuntimeError("Could not detect decoder layer class.")


class MalwareAnalysisDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        if is_main():
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
            prompt_overhead = len(self.tokenizer.encode(
                f"System: {system_msg}\nUser: <binary></binary>{question}\nAssistant: {assistant_msg}",
                add_special_tokens=False
            ))
            hex_budget = max(200, self.max_length - prompt_overhead - 50)
            hex_tokens = self.tokenizer.encode(binary_hex, add_special_tokens=False)
            if len(hex_tokens) > hex_budget:
                truncated = self.tokenizer.decode(hex_tokens[:hex_budget])
                user_msg = f"<binary>{truncated}</binary>{question}"

        text = f"System: {system_msg}\nUser: {user_msg}\nAssistant: {assistant_msg}"
        encoded = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="allenai/Olmo-3-1125-32B")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--cpu_offload", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        logger.info("=" * 60)
        logger.info("OLMo-3 32B Fine-Tuning — Native PyTorch FSDP v3")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Data: {args.data_path}")
        logger.info(f"World size: {world_size}")
        logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
        logger.info(f"GPUs per node: {torch.cuda.device_count()}")
        logger.info("=" * 60)

    log_memory("before-model-load", local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        use_cache=False, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if global_rank == 0:
        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        logger.info(f"Model loaded: {param_count:.1f}B parameters")

    decoder_layer_cls = get_decoder_layer_class(model)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={decoder_layer_cls},
    )

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16,
    )

    cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None

    model = FSDP(
        model, auto_wrap_policy=auto_wrap_policy, mixed_precision=bf16_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD, cpu_offload=cpu_offload,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE, device_id=local_rank,
        limit_all_gathers=True, use_orig_params=True,
    )

    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: isinstance(submodule, decoder_layer_cls),
    )

    dataset = MalwareAnalysisDataset(args.data_path, tokenizer, args.max_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps

    model.train()
    global_step = 0
    total_tokens = 0
    step_times = []
    accumulated_loss = 0.0
    accumulated = 0

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()
            accumulated += 1
            tokens_in_batch = attention_mask.sum().item()
            total_tokens += tokens_in_batch * world_size

            if accumulated >= args.gradient_accumulation_steps:
                model.clip_grad_norm_(1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                step_time = time.time() - step_start
                step_times.append(step_time * args.gradient_accumulation_steps)

                if global_rank == 0 and global_step % args.log_every == 0:
                    tps = (tokens_in_batch * world_size * args.gradient_accumulation_steps) / step_times[-1]
                    avg_tps = total_tokens / sum(step_times) if step_times else 0
                    alloc_gb = torch.cuda.memory_allocated(device) / 1e9
                    logger.info(
                        f"Step {global_step}/{total_steps} | Loss: {accumulated_loss:.4f} "
                        f"| TPS: {tps:.0f} (avg: {avg_tps:.0f}) "
                        f"| GPU mem: {alloc_gb:.1f}GB | Step time: {step_times[-1]:.2f}s"
                    )

                accumulated_loss = 0.0
                accumulated = 0
                dist.barrier()

    if global_rank == 0:
        total_time = sum(step_times)
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        logger.info("TRAINING COMPLETE")
        logger.info(f"Average TPS: {avg_tps:.0f}")
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated(device) / 1e9:.1f}GB")
        metrics = {
            "total_steps": global_step, "total_tokens": total_tokens,
            "total_time_seconds": total_time, "average_tokens_per_second": avg_tps,
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            "world_size": world_size, "model": args.model_name,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "framework": "PyTorch FSDP (native)",
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
