from abc import ABC, abstractmethod
import torch
import numpy as np
from .metric import correlation


class MetricLogger(ABC):
    def __init__(self, max_length):
        self.storage = []
        self.max_length = max_length

    @torch.no_grad()
    def update(self, x):
        if self.max_length is not None and len(self) >= self.max_length:
            self.reset()
        self.storage.append(x)

    def reset(self):
        self.storage.clear()

    def __len__(self):
        return len(self.storage)

    @abstractmethod
    def compute(self):
        raise NotImplementedError


class MetricLoggerScalar(MetricLogger):
    @torch.no_grad()
    def compute(self, aggregation_fn=torch.mean):
        tmp = torch.stack(self.storage)
        result = aggregation_fn(tmp)
        return result


class MetricLoggerHistogram(MetricLogger):
    @torch.no_grad()
    def compute(self, bins=10, range=None):
        tmp = torch.cat(self.storage).cpu().numpy()
        result = np.histogram(tmp, bins=bins, range=range)
        return result


class MetricLoggerCorrelation(MetricLoggerScalar):
    @torch.no_grad()
    def update(self, a, b):
        corr = correlation(a, b)
        super().update(corr)


class MetricLoggerImg(MetricLogger):
    @torch.no_grad()
    def compute(self):
        return self.storage
