# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv (3.12.10)
#     language: python
#     name: python3
# ---

# %%
import sys
import torch
import transformers

print("Python:")
print(sys.executable)

print("\nPyTorch:")
print(torch.__version__)

print("\nCUDA:")
print(torch.cuda.is_available())
print(torch.version.cuda)

print("\nGPU:")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

print("\nTransformers:")
print(transformers.__version__)

# %%
from create_model import create_model
import numpy as np
import time
from pathlib import Path

# %%
ALL_TOKENS = 500_000_000
BATCH_SIZE = 32
SEQ_LEN = 256

TRAIN_TOKENS = 500_000_000

MAX_LR = 1e-3
MIN_LR = 1e-4
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1

EVAL_EVERY = 250
EVAL_BATCHES = 50
MODEL_SIZE = "3M"
MODEL_NAME = f"{MODEL_SIZE}_model_on_{TRAIN_TOKENS // 1000000}M_tokens"
MODEL_DIR = Path("models")

# %%
train_tokens = np.memmap(
    "data/train_500M.bin",
    dtype=np.uint16,
    mode="r",
    shape=(TRAIN_TOKENS,),
)

val_tokens = np.memmap(
    "data/validation/validation_5M.bin",
    dtype=np.uint16,
    mode="r",
    shape=(5_000_000,),
)
test_tokens = np.memmap(
    "data/test/test_5M.bin",
    dtype=np.uint16,
    mode="r",
    shape=(5_000_000,),
)

# %%
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = create_model(MODEL_SIZE)
model = model.to(device)

print(device)


# %%
def get_batch(
    tokens,
    batch_size,
    seq_len,
    device,
):
    max_start = len(tokens) - seq_len - 1

    starts = np.random.randint(
        0,
        max_start,
        size=batch_size,
    )

    x = np.stack([
        tokens[i:i + seq_len]
        for i in starts
    ])

    y = np.stack([
        tokens[i + 1:i + seq_len + 1]
        for i in starts
    ])

    x = torch.from_numpy(
        x.astype(np.int64)
    )

    y = torch.from_numpy(
        y.astype(np.int64)
    )

    x = x.to(device)
    y = y.to(device)

    return x, y


# %%
import math

tokens_per_step = BATCH_SIZE * SEQ_LEN

max_steps = math.ceil(
    TRAIN_TOKENS / tokens_per_step
)

print("Tokens per step:", tokens_per_step)
print("Max steps:", max_steps)


# %%
@torch.no_grad()
def evaluate(
    model,
    tokens,
    batch_size,
    seq_len,
    device,
    num_batches=50,
):
    model.eval()

    losses = []

    for _ in range(num_batches):
        x, y = get_batch(
            tokens=tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )

        outputs = model(
            input_ids=x,
            labels=x,
        )

        losses.append(
            outputs.loss.item()
        )

    model.train()

    return float(np.mean(losses))


# %%
def get_lr(step, max_steps):
    if step < WARMUP_STEPS:
        return MAX_LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, max_steps - WARMUP_STEPS)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


# %%
def train(model, train_tokens, val_tokens, device):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=MAX_LR,
        betas=(0.9, 0.95),
    )

    tokens_per_step = BATCH_SIZE * SEQ_LEN
    max_steps = math.ceil(TRAIN_TOKENS / tokens_per_step)
    save_steps = {
        math.ceil(tokens / tokens_per_step): tokens
        for tokens in [10_000_000, 100_000_000]
    }
    best_val_loss = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.train()
    start_time = time.time()

    for step in range(1, max_steps + 1):

        x, y = get_batch(
            tokens=train_tokens,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            device=device,
        )

        lr = get_lr(step, max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        outputs = model(input_ids=x, labels=x)
        loss = outputs.loss

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if step % 50 == 0:
            tokens_seen = step * tokens_per_step
            elapsed = time.time() - start_time
            print(
                f"step={step:5d} "
                f"tokens={tokens_seen / 1e6:8.2f}M "
                f"loss={loss.item():.4f} "
                f"lr={lr:.2e} "
                f"gnorm={grad_norm:.2f} "
                f"time={elapsed:.1f}s"
            )

        if step % EVAL_EVERY == 0 or step == max_steps or step in save_steps:

            val_loss = evaluate(
                model=model,
                tokens=val_tokens,
                batch_size=BATCH_SIZE,
                seq_len=SEQ_LEN,
                device=device,
                num_batches=EVAL_BATCHES,
            )

            perplexity = math.exp(min(val_loss, 20))

            print(
                f"VALIDATION | step={step} "
                f"loss={val_loss:.4f} ppl={perplexity:.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                    },
                    MODEL_DIR / f"{MODEL_NAME}.pt",
                )

                print("Saved best model.")

            if step in save_steps:
                tokens = save_steps[step]
                name = f"{MODEL_SIZE}_model_on_{tokens // 1000000}M_tokens"

                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                    },
                    MODEL_DIR / f"{name}.pt",
                )

                print(f"Saved model on {tokens // 1000000}M tokens.")


# %%
train(
    model=model,
    train_tokens=train_tokens,
    val_tokens=val_tokens,
    device=device,
)

# %%
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = create_model(MODEL_SIZE)

checkpoint = torch.load(
    MODEL_DIR / f"{MODEL_NAME}.pt",
    map_location=device,
)

model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

print("Loaded checkpoint")
print("Step:", checkpoint["step"])
print("Validation loss:", checkpoint["val_loss"])

# %%
import math


@torch.no_grad()
def evaluate_test(
    model,
    tokens,
    batch_size,
    seq_len,
    device,
):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    starts = range(
        0,
        len(tokens) - seq_len - 1,
        seq_len,
    )

    batch_x = []
    batch_y = []

    for start in starts:

        x = tokens[
            start:start + seq_len
        ]

        y = tokens[
            start + 1:start + seq_len + 1
        ]

        batch_x.append(x)
        batch_y.append(y)

        if len(batch_x) == batch_size:

            x = torch.from_numpy(
                np.stack(batch_x).astype(np.int64)
            ).to(device)

            y = torch.from_numpy(
                np.stack(batch_y).astype(np.int64)
            ).to(device)

            outputs = model(
                input_ids=x,
                labels=x,
            )

            loss = outputs.loss

            n = y.numel()

            total_loss += loss.item() * n
            total_tokens += n

            batch_x.clear()
            batch_y.clear()

    # leftover batch
    if batch_x:
        x = torch.from_numpy(
            np.stack(batch_x).astype(np.int64)
        ).to(device)

        y = torch.from_numpy(
            np.stack(batch_y).astype(np.int64)
        ).to(device)

        outputs = model(
            input_ids=x,
            labels=x,
        )

        loss = outputs.loss

        n = y.numel()

        total_loss += loss.item() * n
        total_tokens += n

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


# %%
test_loss, test_ppl = evaluate_test(
    model=model,
    tokens=test_tokens,
    batch_size=32,
    seq_len=256,
    device=device,
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test perplexity: {test_ppl:.2f}")

# %%
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)

print(tokenizer.get_vocab_size())

# %%
prompt = "Хвороба"

encoded = tokenizer.encode(prompt)

input_ids = torch.tensor(
    [encoded.ids],
    dtype=torch.long,
    device=device,
)
with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,

        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,

        eos_token_id=2,
        pad_token_id=3,
    )
    generated_text = tokenizer.decode(
    output_ids[0].tolist()
)

print(generated_text)
