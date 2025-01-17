##########################################################################################
# Imports
##########################################################################################
import itertools
import warnings

warnings.filterwarnings("ignore")
from torch import nn

from training import train_model
from training import test_model
from sb3_contrib.ppo_mask import MaskablePPO


##########################################################################################
# Grid search
##########################################################################################

GRID = {
    "algorithm_class": [MaskablePPO],
    "gamma": [0.99],
    "learning_rate": [0.0003],
    "normalize_env": [True],
    "activation_fn": [nn.LeakyReLU],
    "net_arch": [
        [256, 256],
        # 은닉층의 뉴런 개수
    ],
}


def hyperparam_generator(grid: dict[str, list]):
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))

    for combination in combinations:
        yield dict(zip(keys, combination))


##########################################################################################
# Training loop
##########################################################################################
def train_models(model_path: str):
    for hyperparams in hyperparam_generator(GRID):
        train_model(model_path=model_path, verbose=0, **hyperparams)


##########################################################################################
# Main
##########################################################################################
if __name__ == "__main__":
    model_path = "OCP_test"
    model = train_models(model_path)
    test_model("OCP_test.zip", n_episodes=1)