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

    def __init__(self,Fs0,Fs0_sigma, *args, rng=None, seed=None, Fs0_meaning:Literal['mean','median']='mean', **kwargs):
        super().__init__(Fs0=Fs0,*args, **kwargs)
        self.Fs0_sigma = Fs0_sigma
        if Fs0_meaning == 'mean':
            self.Fs0_mu = np.log(Fs0) - 0.5 * Fs0_sigma**2
        elif Fs0_meaning == 'median':
            self.Fs0_mu = np.log(Fs0)
        else:
            raise ValueError("Fs0_meaning must be 'mean' or 'median'")
        if rng is None and seed is None:
            raise ValueError("Either rng or seed must be provided.")
        if rng is None:
            self.rng = np.random.default_rng(seed)
        self.rng = rng  # Random number generator, please seed for consistency.


    def intermittency(self, rho_source, u=None, lat=None):
        base_intermittency = self.dc / ( # intermittency without Fs0 scaling 
            rho_source
            * np.sum(np.abs(self.source_spectrum(self.cp, u, lat=lat) * self.dc))
        )
        Fs0 = self.rng.lognormal(mean=self.Fs0_mu, sigma=self.Fs0_sigma)
        return Fs0 * base_intermittency
    

