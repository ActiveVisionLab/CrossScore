from functools import partial
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parents[1]))
from utils.check_config import check_metric_prediction_config


class RegressionLayer(torch.nn.Module):
    def __init__(self, metric_type, metric_min, metric_max, pow_factor="default"):
        """
        Make a regression layer based on the metric configuration.
        Use power_factor to help predict very small numbers.
        """
        super().__init__()

        check_metric_prediction_config(metric_type, metric_min, metric_max)
        self.metric_type = metric_type
        self.metric_min = metric_min
        self.metric_max = metric_max

        self.activation_fn = self._get_activation_fn()
        self.pow_fn = self._get_pow_fn(pow_factor)

    def forward(self, x):
        x = self.activation_fn(x)
        x = self.pow_fn(x)
        return x

    def _get_activation_fn(self):
        if self.metric_min == -1:
            activation_fn = torch.nn.Tanh()
        elif self.metric_min == 0:
            activation_fn = torch.nn.Sigmoid()
        else:
            raise ValueError(f"metric_min={self.metric_min} not supported")
        return activation_fn

    def _get_pow_fn(self, p):
        # define a lookup table for default power factor
        pow_default_table = {
            "ssim": 1,
            "mae": 2,
            "mse": 4,
        }

        # only apply power fn for a non-negative score value range
        if self.metric_min == 0:
            if p == "default":
                # use default power factor from the look up table
                p = pow_default_table[self.metric_type]
            else:
                pass  # use the provided power factor
        else:
            p = 1

        if float(p) == 1.0:
            pow_fn = torch.nn.Identity()
        else:
            pow_fn = partial(torch.pow, exponent=p)
        return pow_fn


if __name__ == "__main__":
    for metric_type in ["ssim", "mae", "mse"]:
        for metric_min in [-1, 0]:
            for p in ["some_typo", "default", 0.1, 1, 1.5, 5]:
                print(f"--------")
                print(f"metric_type: {metric_type}, metric_min: {metric_min}, pow_factor: {p}")
                try:
                    l = RegressionLayer(
                        metric_type=metric_type,
                        metric_min=metric_min,
                        metric_max=1,
                        pow_factor=p,
                    )
                    print(f"activation_fn: {l.activation_fn}")
                    print(f"pow_fn: {l.pow_fn}")
                except Exception as e:
                    print(f"Error: {e}")
