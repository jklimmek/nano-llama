import os
import datetime
import logging
import torch
import numpy as np


def set_logging():
    """Configures logging module."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s", 
        level=logging.INFO,
        datefmt="%H:%M:%S"
    )


def get_model_size(size):
    """Returns model size hyperparameters."""
    # 56,6M non-embedding params.
    if size == "small":
        return dict(dim=768, num_heads=12, num_layers=8)
    # 18,8M non-embedding params.
    elif size == "tiny":
        return dict(dim=512, num_heads=8, num_layers=6)
    # 10,6M non-embedding params.
    elif size == "micro":
        return dict(dim=384, num_heads=6, num_layers=6)
    # 3,1M non-embedding params.
    elif size == "nano":
        return dict(dim=256, num_heads=4, num_layers=4)
    raise ValueError(f"Unknown model size: {size}")
    

def get_lr(max_lr, min_lr, warmup_steps, max_steps, anneal_to_zero, step):
    """Returns learning rate for the given step."""
    # Linearly increase lr for warm up period.
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * step / warmup_steps
    # Cosine decay.
    elif step < max_steps - anneal_to_zero:
        return min_lr + (max_lr - min_lr) / 2 * (1 + np.cos((step - warmup_steps) / (max_steps - anneal_to_zero - warmup_steps) * np.pi))
    # Linearly decrease lr towards zero.
    else:
        return min_lr * (max_steps - step) / anneal_to_zero
    

def get_num_params(model, exclude_embedds=False):
    """Returns the number of parameters in the model."""
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedds:
        num_params -= model.embeddings.weight.numel()
    return num_params
    

def get_run_name(size):
    """Create run's name for Tensorboard."""
    timestamp = datetime.datetime.now().strftime("%b-%d__%H-%M-%S")
    folder_name = f"{size}__{timestamp}"
    return folder_name


def maybe_seed_everything(seed=None):
    """Seeds random number generators."""
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def save_checkpoint(path, model, optimizer=None, max_steps=None, step=None, processed_tokens=None):
    """Saves a model checkpoint."""
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "max_steps": max_steps,
        "step": step,
        "processed_tokens": processed_tokens
    }, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    """Loads a model checkpoint."""
    checkpoint = torch.load(path, map_location=torch.device(map_location))
    model.load_state_dict(checkpoint["model"])
    if optimizer and checkpoint["optimizer"]:
        optimizer.load_state_dict(checkpoint["optimizer"])
    max_steps = checkpoint.get("max_steps", None)
    step = checkpoint.get("step", None)
    processed_tokens = checkpoint.get("processed_tokens", None)
    return max_steps, step, processed_tokens


def calculate_mfu(config, seq_per_iter, dt, device_name, precision):
    """Calculates model flop utilization based on Chinchilla paper."""
    ctx = config["context_size"]
    vocab = config["vocab_size"]
    dim = config["dim"]
    nheads = config["num_heads"]
    nlayers = config["num_layers"]
    swiglu_dim = int(2/3 * 4 * dim // 8 * 8)

    # Some operations like normalization, activations, or ALiBi are skipped.
    # This approach seems more accurate that 6*N*D and predicts ~32% more FLOPs for `tiny` model.
    embbedings = 2*ctx*vocab*dim
    att_qkv = 2*ctx*3*dim*dim
    att_qk_logits = 2*ctx*ctx*dim
    att_softmax = 3*nheads*ctx*ctx
    att_reduction = 2*ctx*ctx*dim
    att_proj = 2*ctx*dim*dim
    swiglu = 2*3*ctx*dim*swiglu_dim
    logits = 2*ctx*dim*vocab

    fwd = embbedings+nlayers*(att_qkv+att_qk_logits+att_softmax+att_reduction+att_proj+att_proj+swiglu)+logits
    bwd = 2 * fwd

    # Testing the code was done on my poor GTX 1660, model trained using Google's T4 in half precision.
    device_flops = {
        "Tesla T4 NVIDIA":  {torch.float32: 8.1e12, torch.bfloat16: 65.1e12, torch.float16: 65.1e12},
        "GeForce GTX 1660": {torch.float32: 5.0e12, torch.bfloat16: 10.1e12, torch.float16: 10.1e12}
    }

    C = fwd + bwd
    D = seq_per_iter * (1.0 / dt)
    P = device_flops.get(device_name, {}).get(precision, None)
    mfu = C * D / P if P else -1.0
    return mfu