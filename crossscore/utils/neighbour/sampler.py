from abc import ABC, abstractmethod
import numpy as np


class SampleBase(ABC):
    def __init__(self, N_sample):
        self.N_sample = N_sample

    @abstractmethod
    def sample(self):
        pass


class SamplerRandom(SampleBase):
    def __init__(self, N_sample, deterministic):
        self.deterministic = deterministic
        super().__init__(N_sample)

    def sample(self, query, ref_list):
        num_ref = len(ref_list)
        if self.N_sample > num_ref:
            # pad empty_image placeholders if ref list < N_sample
            num_empty = self.N_sample - num_ref
            placeholder = ["empty_image"] * num_empty
            result = ref_list + placeholder
            result = np.random.permutation(result).tolist()
        else:
            result = []

            if self.deterministic:
                samples = ref_list[: self.N_sample]
            else:
                samples = np.random.choice(ref_list, self.N_sample, replace=False).tolist()
            result.extend(samples)
        return result


class SamplerFactory:
    def __init__(
        self,
        strategy_name,
        N_sample,
        deterministic,
        **kwargs,
    ):
        self.N_sample = N_sample
        self.deterministic = deterministic

        if strategy_name == "random":
            self.sampler = SamplerRandom(
                N_sample=self.N_sample,
                deterministic=self.deterministic,
            )
        else:
            raise NotImplementedError

    def __call__(self, query, ref_list):
        return self.sampler.sample(query, ref_list)
