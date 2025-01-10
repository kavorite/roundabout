import os
from itertools import islice

import awkward as ak
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from adamw_bf16 import AdamWBF16
from safetensors.torch import save_model
from wav_decoder import decode_wav

from model import AudioDecoder

model = AudioDecoder(input_dim=2206).to(torch.bfloat16).to("cuda")
optimizer = AdamWBF16(model.parameters(), lr=1e-3, cautious=True)
# Set up one cycle learning rate schedule
total_steps = 10000  # Adjust based on your expected number of steps
chunk_size = model.input_dim * 10 * 8  # 8 seconds


def pad_to_multiple_of(a, divisor, axis=-1):
    padding = [(0, 0)] * a.ndim
    padding[axis] = (0, -a.shape[axis] % divisor)
    return np.pad(a, padding)


def batches():
    if not os.path.exists("gtzan"):
        import subprocess as sp

        sp.run("fetch.sh")

    df = pl.scan_parquet("gtzan/*.parquet").select(
        decode_wav(pl.col("audio").struct.field("bytes")).alias("audio"), "genre"
    )
    loaded = ak.from_arrow(df.collect().to_arrow())
    padded = [
        pad_to_multiple_of(ak.to_numpy(a), chunk_size).reshape(-1, chunk_size)
        for a in loaded["audio"]
    ]
    tk_ids = np.repeat(np.arange(len(padded)), [a.shape[-2] for a in padded])
    genres = np.repeat(ak.to_numpy(loaded["genre"]), [a.shape[-2] for a in padded])
    padded = np.concatenate(padded, axis=-2)
    cursor = 0
    while True:
        chunk = {
            "tk_id": tk_ids[cursor : cursor + chunk_size],
            "audio": padded[cursor : cursor + chunk_size],
            "genre": genres[cursor : cursor + chunk_size],
        }
        yield {k: torch.from_numpy(ak.to_numpy(v)[None, ...]) for k, v in chunk.items()}

        cursor += chunk_size
        cursor %= sum(len(x) for x in loaded["audio"])
        yield chunk


def main():
    max_lr = 1e-3
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1e4,
        anneal_strategy="cos",
    )

    running_loss = 0.0
    loss_beta = 1 - 1 / 128

    for step, batch in islice(enumerate(batches()), total_steps):
        optimizer.zero_grad()
        logits = model(batch["audio"].to(torch.bfloat16).to("cuda"), batch["tk_id"])
        loss = F.cross_entropy(
            logits, F.one_hot(batch["genre"].to(device="cuda", dtype=torch.bfloat16))
        ).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            display_loss = loss.item()
            if step == 0:
                running_loss = display_loss
            else:
                attenuation = 1 - (1 - loss_beta) ** (step + 1)
                running_loss = (
                    loss_beta * running_loss + (1 - loss_beta) * display_loss
                ) / attenuation
        print(f"Step {step}: {running_loss:.3g}" + 32 * " ", end="\r")

    save_model(model, "model.safetensors")
    print("Saved to model.safetensors")


if __name__ == "__main__":
    main()
