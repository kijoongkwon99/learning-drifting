from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import torch
import tyro
from tqdm.auto import tqdm

from learning_drifting.train.prepare_dataset import TOY_DATASETS, ToyDatasetName, SyntheticDataset
from learning_drifting.utils.utils import set_seed, train_with_eval, train_only
from learning_drifting.utils import visualization

from learning_drifting.methods import Drifting
from learning_drifting.models import MlpModel

METHODS = {
    "vanilla_drifting": Drifting,
}

MODELS = {
    "mlp": MlpModel,
}

@dataclass
class ScriptArguments:
    dataset: ToyDatasetName = "logo"
    output_dir: Path = Path("../../../outputs")

    method: Literal["vanilla_drifting"] = "vanilla_drifting"
    model: Literal["mlp"] = "mlp"

    learning_rate: float = 1e-3
    batch_size: int = 4096
    iterations: int = 20000
    log_every: int = 200
    hidden_dim: int = 512
    seed: int = 74

    train_with_eval: bool = True
    visualize: bool = True

    plot_num_samples: int = 1_000_000
    
def build_model(args: ScriptArguments, dataset: SyntheticDataset) -> torch.nn.Module:
    if args.model == "mlp":
        return MlpModel(
            dim = dataset.dim,
            hidden_dim=args.hidden_dim,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

def build_method(args: ScriptArguments) -> torch.nn.Module:
    if args.method == "vanilla_drifting":
        return Drifting(normalize="xy")

    else:
        raise ValueError(f"Unkown method: {args.method}")

def main(args:ScriptArguments) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    output_dir = args.output_dir / args.method / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device:   {device}")
    print(f"Dataset:        {args.dataset}")
    print(f"Method:         {args.method}")
    print(f"Model:          {args.model}")

    dataset: SyntheticDataset = TOY_DATASETS[args.dataset](device=device)

    model = build_model(args, dataset).to(device)
    method = build_method(args).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    losses: list[float] = []
    model.train()

    for global_step in tqdm(range(args.iterations), desc="Training", dynamic_ncols=True):
        
        # pos = sampler(args.batch_size).to(device)
        x_1 = dataset.sample(args.batch_size).to(device)
        out = method(model=model, pos=x_1)

        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if args.train_with_eval:
            train_with_eval(
                global_step=global_step,
                log_every=args.log_every,
                model=model,
                loss=loss,
                output_dir=output_dir / "per_steps" / "png",
                dataset=dataset,
                visualize=args.visualize,
                plot_num_samples=args.plot_num_samples,
                noise_dim=dataset.dim,   
            )
        else:
            train_only(
                global_step=global_step,
                log_every=args.log_every,
                model=model,
                loss=loss,
                output_dir=output_dir,
            )

    visualization.plot_loss_curve(
        losses=losses,
        output_path=output_dir / "losses.png",
    )

    
    # snapshot_steps = [0] + list(range(args.log_every, args.iterations + 1, args.log_every))

    if args.visualize:
        visualization.plot_training_snapshots(
            model=model,
            dataset=dataset,
            checkpoint_steps=[1, 200, 400, 600, 800, 1000, 1600, 2000], #10000, 20000],
            checkpoint_dir=output_dir / "per_steps" / "ckpt",  
            output_path=output_dir / "training_snapshots.png",
            num_samples=args.plot_num_samples,
            noise_dim=dataset.dim,
        )


if __name__ == "__main__":
    main(tyro.cli(ScriptArguments))