import argparse
import contextlib
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import NanoLlama
from utils import *


class SFTDataset(Dataset):
    """Each instance in dataset is padded to pre-defined length and contains special tokens.

    Tokens of each sample: 
    [TITLE] <Example title here> [CONTEXT] <Text of the article> [END-OF-TEXT] [PADDING] [PADDING] ...
    """
    def __init__(self, data_path, device):
        super().__init__()
        self.data = np.load(data_path)
        self.device = device

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x, y = self.data[index, :-1], self.data[index, 1:]
        x = torch.from_numpy(x).to(torch.long).to(self.device)
        y = torch.from_numpy(y).to(torch.long).to(self.device)
        return x, y
    

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="tiny", choices=["small", "tiny", "micro", "nano"], help="Model size.") # ok.
    parser.add_argument("--vocab_size", type=int, default=10_000, help="Vocabulary size.") # ok.
    parser.add_argument("--batch_size", type=int, default=32, help="Number of examples per batch.") # ok.

    parser.add_argument("--epochs", type=int, default=1, help="Total number of epochs.") # ok.
    parser.add_argument("--max_lr", type=float, default=3e-5, help="Max learning rate.") # ok.
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Min learning rate.") # ok.
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps.") # ok.
    parser.add_argument("--anneal_to_zero", type=int, default=100, help="Decrease learning rate to 0 for last steps.") # ok.
    
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.") # ok.  
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999), help="AdamW betas.") # ok.
    parser.add_argument("--clip_grad", type=float, default=0.0, help="Gradient clipping.") # ok.
    parser.add_argument("--grad_acc", type=int, default=1, help="Gradient accumulation steps.") # ok.
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for ffwd and attention.") # ok.
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Training precision.") # ok.

    parser.add_argument("--log_interval", type=int, default=5, help="Log interval.") # ok.

    parser.add_argument("--train_path", type=str, required=True, help="Directory with training files.") # ok.
    parser.add_argument("--dev_path", type=str, required=True, help="Directory with dev file.") # ok.
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Where to save the model.") # ok.
    parser.add_argument("--log_dir", type=str, default="logs", help="Where to save logs.") # ok
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to a checkpoint to load.") # ok.
    parser.add_argument("--seed", type=int, default=None, help="Random seed.") # ok.
    parser.add_argument("--comment", type=str, default=None, help="Additional comment for Tensorboard.") # ok.
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU.") # ok.

    args = parser.parse_args()
    model_size = get_model_size(args.model_size)
    args_dict = vars(args)
    args_dict.update(model_size)
    args_dict["device"] = "cuda" if torch.cuda.is_available() and not args_dict.pop("use_cpu") else "cpu"
    if args_dict["precision"] == "fp16" and args_dict["device"] == "cuda":
        args_dict["precision"] = torch.float16
    elif args_dict["precision"] == "bf16" and args_dict["device"] == "cuda" and torch.cuda.is_bf16_supported():
        args_dict["precision"] = torch.bfloat16
    else:
        args_dict["precision"] = torch.float32
    return args_dict


def start_training(args):
    """Start fine-tuning."""

    # Pop arguments that won't be logged to TensorBoard.
    device = args.pop("device")
    seed = args.pop("seed")
    log_interval = args.pop("log_interval")
    precision = args.pop("precision")
    train_path = args.pop("train_path")
    dev_path = args.pop("dev_path")
    checkpoint = args.pop("checkpoint")
    comment = args.pop("comment")
    model_dir = os.path.join(args.pop("model_dir"), get_run_name(args["model_size"]))
    log_dir = os.path.join(args.pop("log_dir"), get_run_name(args["model_size"]))

    # Logging and seeding.
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
        resid_dropout=args["dropout"]
    ).to(device)
    _ = load_checkpoint(checkpoint, model)

    criterion = nn.CrossEntropyLoss()

    optimizer = model.configure_optimizer(args["weight_decay"], args["max_lr"], args["betas"])
    logging.info(f"Model has {get_num_params(model):,} parameters ({get_num_params(model, exclude_embedds=True):,} non-embeddings).")

    # Set up TensorBoard.
    writer = SummaryWriter(log_dir + "_" + comment if comment else log_dir)
    writer.add_text(
        "hyperparams",
        "|param|value|\n|-|-|\n%s" % ("\n".join(f"|{key}|{value}|" for key, value in args.items()))
    )

    # Set up dataloaders.
    train_ds = SFTDataset(train_path, device)
    dev_ds = SFTDataset(dev_path, device)
    # For some reason training crashes when num_workers=os.cpu_count() 
    train_loader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=args["batch_size"], shuffle=False, num_workers=0)
    logging.info(f"Total of {len(train_ds):,} train samples.")
    logging.info(f"Total of {len(dev_ds):,} dev samples.")

    # Set up grad scaler if using float16.
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == torch.float16))
    ctx = contextlib.nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=precision)
    logging.info(f"Grad scaler enabled={scaler.is_enabled()}. Training in {str(precision).split('.')[1]} precision.")

    # Training loop.
    for epoch in range(args["epochs"]):
        
        # Training part.
        train_epoch_loss = 0
        train_step_loss = []
        model.train()
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}",ncols=100):

            # Adjust learning rate.
            lr = get_lr(args["max_lr"], args["min_lr"], args["warmup_steps"], args["epochs"] * len(train_loader), args["anneal_to_zero"], step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward & backward pass.
            x, y = batch
            with ctx:
                logits = model(x)
                # todo: Zero-out loss for prompt.
                loss = criterion(logits.transpose(1, 2), y) / args["grad_acc"]
            scaler.scale(loss).backward()

            # Collect statistics to log later.
            unscaled_loss = loss.item() * args["grad_acc"]
            train_epoch_loss += unscaled_loss * args["grad_acc"] / len(train_loader)
            train_step_loss.append(unscaled_loss * args["grad_acc"])

            # Gradient clipping and optimizer update.
            if step % args["grad_acc"] == 0 or step == len(train_loader) - 1:
                if args["clip_grad"] != 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args["clip_grad"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Logging.
            if step % log_interval == 0:
                tb_step = len(train_loader) * epoch + step
                writer.add_scalar("sft_losses/train_step_loss", np.mean(train_step_loss), tb_step)
                writer.add_scalar("sft_charts/lr", lr, tb_step)
                writer.add_scalar("sft_charts/loss_scale", scaler.get_scale(), tb_step)
                train_step_loss = []
                
        writer.add_scalar("sft_losses/train_epoch_loss", train_epoch_loss, epoch + 1)

        # Evaluation part.
        with torch.no_grad():
            dev_epoch_loss = 0
            model.eval()
            for step, batch in tqdm(enumerate(dev_loader), total=len(dev_loader), desc=f"Epoch {epoch}",ncols=100):
                x, y = batch
                logits = model(x)
                loss = criterion(logits.transpose(1, 2), y)
                dev_epoch_loss += loss.item() / len(dev_loader)
            writer.add_scalar("sft_losses/total_dev_loss", dev_epoch_loss, epoch + 1)

        # Save checkpoint.
        checkpoint_name = f"epoch-{epoch + 1}_train-{train_epoch_loss:.4f}_dev-{dev_epoch_loss:.4f}.pt"
        save_checkpoint(
            path=os.path.join(model_dir, checkpoint_name),
            model=model,
        )


def main():
    args = parse_args()
    start_training(args)


if __name__ == "__main__":
    main()