import argparse
import contextlib
import logging
import time
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import NanoLlama
from utils import *


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="tiny", choices=["small", "tiny", "micro", "nano"], help="Model size.") # ok.
    parser.add_argument("--init_mode", type=str, default="gpt-2", choices=["gpt-2", "olmo", "small"], help="Weight init type.") # ok.
    parser.add_argument("--vocab_size", type=int, default=20_000, help="Vocabulary size.") # ok.
    parser.add_argument("--context_size", type=int, default=256, help="Number of tokens to process at a time.") # ok.
    parser.add_argument("--tokens_per_batch", type=int, default=4096, help="Number of tokens per batch.") # ok.

    parser.add_argument("--max_num_steps", type=int, default=100_000, help="Max number of training steps.") # ok.
    parser.add_argument("--num_steps", type=int, default=10_000, help="Number of training steps in this run.")
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Max learning rate.") # ok.
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Min learning rate.") # ok.
    parser.add_argument("--warmup_steps", type=int, default=2_000, help="Number of warmup steps.") # ok.
    
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.") # ok.  
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="AdamW betas.") # ok.
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping.") # ok.
    parser.add_argument("--grad_acc", type=int, default=4, help="Gradient accumulation steps.") # ok.
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for ffwd and attention.") # ok.
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Training precision.") # ok.

    parser.add_argument("--log_interval", type=int, default=5, help="Log interval.") # ok.
    parser.add_argument("--eval_interval", type=int, default=1_000, help="Evaluation interval.") # ok.
    parser.add_argument("--save_interval", type=int, default=5_000, help="Save interval. -1 means save only once at the end.") # ok.

    parser.add_argument("--train_dir", type=str, required=True, help="Directory with training files.") # ok.
    parser.add_argument("--dev_dir", type=str, required=True, help="Directory with dev file.") # ok.
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Where to save the model.") # ok.
    parser.add_argument("--log_dir", type=str, default="logs", help="Where to save logs.") # ok
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint to load.") # ok.
    parser.add_argument("--seed", type=int, default=None, help="Random seed.") # ok.
    parser.add_argument("--comment", type=str, default=None, help="Additional comment for Tensorboard.") # ok.
    parser.add_argument("--use_cpu", action="store_true", help="Deliberately use CPU instead of GPU for debug purposes.") # ok.

    args = parser.parse_args()
    model_size = get_model_size(args.model_size)
    args_dict = vars(args)
    args_dict.update(model_size)
    args_dict["device"] = "cuda" if torch.cuda.is_available() and not args_dict.pop("use_cpu") else "cpu"
    args_dict["batch_size"] = int(args.tokens_per_batch / args.context_size)
    if args_dict["precision"] == "fp16" and args_dict["device"] == "cuda":
        args_dict["precision"] = torch.float16
    elif args_dict["precision"] == "bf16" and args_dict["device"] == "cuda" and torch.cuda.is_bf16_supported():
        args_dict["precision"] = torch.bfloat16
    else:
        args_dict["precision"] = torch.float32
    return args_dict


def get_train_batch(files, batch_size, context_size, device):
    """Returns a batch of train data."""
    file = np.random.choice(files)
    data = np.load(file, mmap_mode="r")
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+context_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+context_size+1]).astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


def get_dev_batches(file, batch_size, context_size, device):
    """Returns a batch of dev data."""
    data = np.load(file, mmap_mode="r")
    indexes = torch.arange(0, len(data) -  context_size, context_size)
    for index in range(len(indexes) // batch_size):
        ix = indexes[index*batch_size:(index+1)*batch_size]
        x = torch.stack([torch.from_numpy((data[i:i+context_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+context_size+1]).astype(np.int64)) for i in ix])
        if device == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y


ACTIVATIONS = {}
def get_activation(name):
    """Forward hook to get activations of each layer."""
    def hook(model, input, output):
        ACTIVATIONS[name] = output.detach()
    return hook


def start_training(args):
    """Start pre-training."""
    global ACTIVATIONS

    # Pop arguments that won't be logged to TensorBoard.
    device = args.pop("device")
    seed = args.pop("seed")
    log_interval = args.pop("log_interval")
    eval_interval = args.pop("eval_interval")
    save_interval = args.pop("save_interval")
    precision = args.pop("precision")
    train_dir = args.pop("train_dir")
    dev_dir = args.pop("dev_dir")
    checkpoint = args.pop("checkpoint")
    comment = args.pop("comment")
    model_dir = os.path.join(args.pop("model_dir"), get_run_name(args["model_size"]))
    log_dir = os.path.join(args.pop("log_dir"), get_run_name(args["model_size"]))
    
    # Logging and maybe seeding.
    set_logging()
    maybe_seed_everything(seed)

    # Model, criterion and optimizer.
    model = NanoLlama(
        vocab_size=args["vocab_size"],
        context_size=args["context_size"],
        dim=args["dim"],
        num_heads=args["num_heads"],
        num_layers=args["num_layers"],
        attn_dropout=args["dropout"],
        ffwd_dropout=args["dropout"],
        resid_dropout=args["dropout"],
        init_mode=args["init_mode"]
    ).to(device)

    # Register hooks for logging activations.
    for name, param in model.named_parameters():
        model.register_forward_hook(get_activation(name))

    criterion = nn.CrossEntropyLoss()

    optimizer = model.configure_optimizer(args["weight_decay"], args["max_lr"], args["betas"])
    logging.info(f"Model has {get_num_params(model):,} parameters ({get_num_params(model, exclude_embedds=True):,} non-embeddings).")
    logging.info(f"Total of {args['grad_acc'] * args['tokens_per_batch']:,} tokens per optimizer step.")

    # Set up TensorBoard.
    writer = SummaryWriter(log_dir + "_" + comment if comment else log_dir)
    writer.add_text(
        "hyperparams",
        "|param|value|\n|-|-|\n%s" % ("\n".join(f"|{key}|{value}|" for key, value in args.items()))
    )

    writer_tps = []
    writer_losses = []
    writer_grad_norms = []

    # Get files to sample from.
    train_files = [os.path.join(train_dir, f.name) for f in os.scandir(train_dir) if f.is_file()]
    dev_file = [os.path.join(dev_dir, f.name) for f in os.scandir(dev_dir) if f.is_file()][0]
    logging.info(f"Found {len(train_files)} train files.")

    # Set up grad scaler if using float16.
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == torch.float16))
    ctx = contextlib.nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=precision)
    logging.info(f"Grad scaler enabled={scaler.is_enabled()}. Training in {str(precision).split('.')[1]} precision.")

    # Model Flop Utilization (MFU).
    device_name = torch.cuda.get_device_name() if device == "cuda" else None
    logging.info(f"Using {device_name if device_name is not None else 'CPU'}")
    # Running MFU as in Karpathy's NanoGPT.
    running_mfu = -1.0

    # Load checkpoint if provided.
    if checkpoint:
        max_steps, step, processed_tokens = load_checkpoint(checkpoint, model, optimizer)
        logging.info(f"Loading checkpoint from {checkpoint}.")
    else:
        max_steps = args["max_num_steps"]
        step = 0
        processed_tokens = 0
        logging.info("No checkpoint provided. Starting from scratch.")

    # Adjust number of steps if necessary.
    if step + args["num_steps"] > max_steps:
        logging.info(f"Training for {max_steps - step} steps.")
        args["num_steps"] = max_steps - step

    # Training loop.
    model.train()
    for step in tqdm(range(step + 1, step + args["num_steps"] + 1), desc="Training", total=args["num_steps"], ncols=100):
        
        # Adjust learning rate.
        lr = get_lr(args["max_lr"], args["min_lr"], args["warmup_steps"], args["max_num_steps"], 0, step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward & backward pass + measuring time.
        t1 = time.time()
        x, y = get_train_batch(train_files, args["batch_size"], args["context_size"], device)
        with ctx:
            logits = model(x)
            loss = criterion(logits.transpose(1, 2), y) / args["grad_acc"]
        scaler.scale(loss).backward()
        t2 = time.time()
        dt = t2 - t1

        # Get gradient norms for Tensorboard.
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_scale = scaler.get_scale()
                writer_grad_norms.append(torch.norm(param.grad / grad_scale, p=2).item())
        
        # Collect statistics to log later.
        writer_losses.append(loss.item() * args["grad_acc"])
        writer_tps.append(dt)
        processed_tokens += args["tokens_per_batch"]
        mfu = calculate_mfu(args, args["batch_size"], dt, device_name, precision)
        running_mfu = mfu if running_mfu == -1 else 0.9 * running_mfu + 0.1 * mfu

        # Gradient clipping and optimizer update.
        if step % args["grad_acc"] == 0 or step == max_steps:
            if args["clip_grad"] != 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args["clip_grad"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Logging.
        if step % log_interval == 0:
            writer.add_scalar("pt_losses/train_loss", np.mean(writer_losses), processed_tokens)
            writer.add_scalar("pt_losses/perplexity", np.exp(np.mean(writer_losses)), processed_tokens)
            writer.add_scalar("pt_charts/grad_norm", np.sum(writer_grad_norms), processed_tokens)
            writer.add_scalar("pt_charts/mfu", running_mfu, processed_tokens)
            writer.add_scalar("pt_charts/tps", args["tokens_per_batch"] / (np.mean(writer_tps)), processed_tokens)
            writer.add_scalar("pt_charts/acts", sum(v.sum().abs() for v in ACTIVATIONS.values()), processed_tokens)
            writer.add_scalar("pt_charts/lr", lr, processed_tokens)
            writer.add_scalar("pt_charts/loss_scale", scaler.get_scale(), processed_tokens)
            writer_tps = []
            writer_losses = []
            writer_grad_norms = []
            ACTIVATIONS = {}

        # Evaluation.
        if step % eval_interval == 0:
            model.eval()
            total_dev_loss = []

            with torch.no_grad():
                for batch in get_dev_batches(dev_file, args["batch_size"], args["context_size"], device):
                    x, y = batch
                    logits = model(x)
                    loss = criterion(logits.transpose(1, 2), y).item()
                    total_dev_loss.append(loss)

            writer.add_scalar("pt_losses/total_dev_loss", np.mean(total_dev_loss), processed_tokens)
            model.train()

        # Checkpointing.
        if save_interval != -1 and step % save_interval == 0:
            save_checkpoint(
                path=os.path.join(model_dir, f"model_{step}.pt"), 
                model=model, 
                optimizer=optimizer, 
                max_steps=args["max_num_steps"], 
                step=step,
                processed_tokens=processed_tokens
            )
        
    save_checkpoint(
        path=os.path.join(model_dir, f"model.pt"), 
        model=model, 
        optimizer=optimizer, 
        max_steps=args["max_num_steps"], 
        step=step,
        processed_tokens=processed_tokens
    )
    

def main():
    args = parse_args()
    start_training(args)


if __name__ == "__main__":
    main()