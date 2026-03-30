import random

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm
from learning_drifting.utils import visualization



def set_seed(seed: int) -> None:
    """Set the seed for the random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def train_with_eval(
    global_step,
    log_every,
    model,
    loss,
    output_dir,
    dataset,
    visualize,
    plot_num_samples=100_000,
    noise_dim=2,
):
    step = global_step + 1

    if step == 1 or step % log_every == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        

        tqdm.write(
            f"| step: {step:6d} "
            f"| loss: {loss.item():10.8f} |"
        )

        model.eval()
        ckpt_dir = output_dir.parent / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"ckpt_step_{step:06d}.pth")

        if visualize:
            visualization.plot_drifting_samples(
                model=model,
                dataset=dataset,
                output_dir=output_dir,   # per_step 폴더
                filename=f"samples_step_{step:06d}.png",
                num_samples=plot_num_samples,
                noise_dim=noise_dim,
            )

        model.train()

def train_only(global_step, log_every, model, loss, output_dir):
    if (global_step + 1) % log_every == 0:
        tqdm.write(
            f"| step: {global_step + 1:6d} "
            f"| loss: {loss.item():8.4f} |"
        )
    torch.save(model.state_dict(), output_dir / "ckpt.pth")