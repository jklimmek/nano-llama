import argparse
import contextlib
import logging
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import NanoLlama
from utils import *


class DPODataset(Dataset):
    """Each instance in dataset is padded to pre-defined length and contains special tokens.

    Tokens of each sample: 
    [TITLE] <Example title here> [CONTEXT] <Accepted article tokens> [END-OF-TEXT] [PADDING] [PADDING] ...
    [TITLE] <Example title here> [CONTEXT] <Rejected article tokens> [END-OF-TEXT] [PADDING] [PADDING] ...
    """
    def __init__(self, data_path, device):
        super().__init__()
        self.data = np.load(data_path)
        self.device = device

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        accepted, rejected = torch.from_numpy(self.data[index]).to(torch.long).to(self.device).chunk(2, dim=-1)
        # print()
        # print(accepted.shape)
        # print()
        x_accepted, y_accepted = accepted[:-1], accepted[1:]
        x_rejected, y_rejected = rejected[:-1], rejected[1:]
        return x_accepted, y_accepted, x_rejected, y_rejected
    

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="tiny", choices=["small", "tiny", "micro", "nano"], help="Model size.") # ok.
    parser.add_argument("--vocab_size", type=int, default=10_000, help="Vocabulary size.") # ok.
    parser.add_argument("--context_size", type=int, default=256, help="Number of tokens to process at a time.") # ok.
    parser.add_argument("--batch_size", type=int, default=16, help="Number of examples per batch.") # ok.

    parser.add_argument("--epochs", type=int, default=1, help="Total number of epochs.") # ok.
    parser.add_argument("--max_lr", type=float, default=3e-6, help="Max learning rate.") # ok.
    parser.add_argument("--min_lr", type=float, default=3e-6, help="Min learning rate.") # ok.
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps.") # ok.
    parser.add_argument("--anneal_to_zero", type=int, default=100, help="Decrease learning rate to 0 for last steps.") # ok.

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.") # ok.  
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for DPO.") # ok.  
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999), help="AdamW betas.") # ok.
    parser.add_argument("--clip_grad", type=float, default=0.0, help="Gradient clipping.") # ok.
    parser.add_argument("--grad_acc", type=int, default=1, help="Gradient accumulation steps.") # ok.
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for ffwd and attention.") # ok.
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Training precision.") # ok.

    parser.add_argument("--log_interval", type=int, default=5, help="Log interval.") # ok.

    parser.add_argument("--train_path", type=str, required=True, help="Directory with training files.") # ok.
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


def preference_loss(
        policy_logprobs_accepted, 
        policy_logprobs_rejected, 
        reference_logprobs_accepted, 
        reference_logprobs_rejected,
        beta
    ):
    """Vanilla DPO loss according to Bradley-Terry preference model."""
    log_ratio_w = policy_logprobs_accepted - reference_logprobs_accepted
    log_ratio_l = policy_logprobs_rejected - reference_logprobs_rejected
    dpo_loss = -F.logsigmoid(beta * log_ratio_w - beta * log_ratio_l).mean()
    accepted_rewards = (beta * log_ratio_w).mean().detach()
    rejected_rewards = (beta * log_ratio_l).mean().detach()
    return dpo_loss, accepted_rewards, rejected_rewards


def start_training(args):
    """Start DPO training."""

    # Pop arguments that won't be logged to TensorBoard.
    device = args.pop("device")
    seed = args.pop("seed")
    log_interval = args.pop("log_interval")
    precision = args.pop("precision")
    train_path = args.pop("train_path")
    checkpoint = args.pop("checkpoint")
    comment = args.pop("comment")
    model_dir = os.path.join(args.pop("model_dir"), get_run_name(args["model_size"]))
    log_dir = os.path.join(args.pop("log_dir"), get_run_name(args["model_size"]))

    # Logging and seeding.
    set_logging()
    maybe_seed_everything(seed)

    # Models and optimizer.
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
    ref_model = copy.deepcopy(model)

    optimizer = model.configure_optimizer(args["weight_decay"], args["max_lr"], args["betas"])
    logging.info(f"Model has {get_num_params(model):,} parameters ({get_num_params(model, exclude_embedds=True):,} non-embeddings).")

    # Set up TensorBoard.
    writer = SummaryWriter(log_dir + "_" + comment if comment else log_dir)
    writer.add_text(
        "hyperparams",
        "|param|value|\n|-|-|\n%s" % ("\n".join(f"|{key}|{value}|" for key, value in args.items()))
    )

    # Set up dataloader.
    # Trained models are too small for DPO to work properly, so I skip validation part.
    train_ds = DPODataset(train_path, device)
    train_loader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True, num_workers=0)
    logging.info(f"Total of {len(train_ds):,} train samples.")

    # Set up grad scaler if using float16.
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == torch.float16))
    ctx = contextlib.nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=precision)
    logging.info(f"Grad scaler enabled={scaler.is_enabled()}. Training in {str(precision).split('.')[1]} precision.")

    # Training loop.
    for epoch in range(args["epochs"]):
        
        train_epoch_loss = 0
        train_step_loss = []
        train_step_acc_rews = []
        train_step_rej_rews = []
        model.train()
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}",ncols=100):

            # Adjust learning rate.
            lr = get_lr(args["max_lr"], args["min_lr"], args["warmup_steps"], args["epochs"] * len(train_loader), args["anneal_to_zero"], step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward & backward pass.
            x_accepted, y_accepted, x_rejected, y_rejected = batch
            # print()
            # print(x_accepted.shape)
            # print()
            with ctx:
                # Query the models.
                logprobs_accepted = model(x_accepted).log_softmax(-1)
                logprobs_rejected = model(x_rejected).log_softmax(-1)
                ref_logprobs_accepted = ref_model(x_accepted).log_softmax(-1)
                ref_logprobs_rejected = ref_model(x_rejected).log_softmax(-1)

                # Get selected logprobs.
                logprob_accepted = logprobs_accepted.gather(2, y_accepted.unsqueeze(2)).squeeze(-1)
                logprob_rejected = logprobs_rejected.gather(2, y_rejected.unsqueeze(2)).squeeze(-1)
                ref_logprob_accepted = ref_logprobs_accepted.gather(2, y_accepted.unsqueeze(2)).squeeze(-1)
                ref_logprob_rejected = ref_logprobs_rejected.gather(2, y_rejected.unsqueeze(2)).squeeze(-1)

                # Calculate the loss and the rewards.
                # todo: Zero-out loss for prompt.
                dpo_loss, accepted_rewards, rejected_rewards = \
                    preference_loss(
                        logprob_accepted, logprob_rejected, ref_logprob_accepted, ref_logprob_rejected, args["beta"]
                    )

            scaler.scale(dpo_loss / args["grad_acc"]).backward()

            # Collect statistics to log later.
            loss_item = dpo_loss.item()
            train_step_loss.append(loss_item)
            train_epoch_loss += loss_item / len(train_loader)
            train_step_acc_rews.append(accepted_rewards.item())
            train_step_rej_rews.append(rejected_rewards.item())

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
                writer.add_scalar("dpo_losses/train_step_loss", np.mean(train_step_loss), tb_step)
                writer.add_scalar("dpo_charts/lr", lr, tb_step)
                writer.add_scalar("dpo_charts/train_step_acc_rews", np.mean(train_step_acc_rews), tb_step)
                writer.add_scalar("dpo_charts/train_step_rej_rews", np.mean(train_step_rej_rews), tb_step)
                train_step_loss = []
                train_step_acc_rews = []
                train_step_rej_rews = []

        # Save checkpoint.
        checkpoint_name = f"epoch-{epoch + 1}_loss-{train_epoch_loss:.4f}.pt"
        save_checkpoint(
            path=os.path.join(model_dir, checkpoint_name),
            model=model,
        )


def main():
    args = parse_args()
    start_training(args)


if __name__ == "__main__":
    main()