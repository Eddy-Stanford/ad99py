from typing import Literal
from .ad99 import AlexanderDunkerton1999
import numpy as np


class AlexanderDunkerton1999Stochastic(AlexanderDunkerton1999):
    """
    Alexander-Dunkerton 1999 model with stochastic intermittency.
    Note that the same input call will provide different outputs due to the stochasticity.
    Be warned, is this class is run using `map_blocks` in `dask`, the results may vary for the same seed
    due to the order that blocks are processed in. 

    Make sure to use a seed PER block in this case to ensure reproducibility. 
    """

    def __init__(self,Fs0,Fs0_sigma, *args, rng=None, seed=None, distribution:Literal['lognormal','normal','uniform']='lognormal', **kwargs):
        super().__init__(Fs0=Fs0,*args, **kwargs)
        self.Fs0_sigma = Fs0_sigma
        self.distribution = distribution
        if rng is None and seed is None:
            raise ValueError("Either rng or seed must be provided.")
        if rng is None:
            self.rng = np.random.default_rng(seed)
        self.rng = rng  # Random number generator, please seed for consistency.

    def sample_fs0(self):
        if self.distribution == 'lognormal':
            sigma = np.sqrt(np.log( (self.Fs0_sigma/self.Fs0)**2 + 1))
            mu = np.log(self.Fs0) - 0.5*sigma**2

            return self.rng.lognormal(mean=mu,sigma=sigma)

        elif self.distribution == 'normal':
            return np.abs(self.rng.normal(loc=self.Fs0,scale=self.Fs0_sigma)) # must be positive
        elif self.distribution == 'uniform':
            low = np.max(self.Fs0 - self.Fs0_sigma*(12**0.5)/2,0)
            high = self.Fs0 + self.Fs0_sigma*(12**0.5)/2
            return self.rng.uniform(low=low,high=high)
        else:
            raise ValueError("distribution must be one of ['lognormal','normal','uniform']")

    def intermittency(self, rho_source, u=None, lat=None):
        base_intermittency = self.dc / ( # intermittency without Fs0 scaling 
            rho_source
            * np.sum(np.abs(self.source_spectrum(self.cp, u, lat=lat) * self.dc))
        )
        Fs0 = self.sample_fs0()
        return Fs0 * base_intermittency
    

