from typing import Literal

from learning_drifting.train.prepare_dataset.toy_datasets import (
    DatasetCheckerboard,
    DatasetInvertocat,
    DatasetMixture,
    DatasetMoons,
    DatasetSiggraph,
    SyntheticDataset,
    DatasetLogo,
)

ToyDatasetName = Literal["moons", "mixture", "siggraph", "checkerboard", "invertocat", "logo"]

TOY_DATASETS: dict[str, type[SyntheticDataset]] = {
    "moons": DatasetMoons,
    "mixture": DatasetMixture,
    "siggraph": DatasetSiggraph,
    "checkerboard": DatasetCheckerboard,
    "invertocat": DatasetInvertocat,
    "logo": DatasetLogo,
}
