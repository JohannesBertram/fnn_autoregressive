import torch
from .parameters import Parameter
from .modules import Module


# -------------- Position Prototype --------------


class Position(Module):
    """Position Module"""

    def _init(self, units):
        """
        Parameters
        ----------
        units : int
            number of units (U)
        """
        raise NotImplementedError()

    def sample(self, batch_size=1):
        """
        Parameters
        ----------
        batch_size : int
            batch size (N)

        Returns
        -------
        Tensor
            [N, U, 2], 2D spatial positions
        """
        raise NotImplementedError()

    @property
    def mean(self):
        """
        Returns
        -------
        Tensor
            [U, 2], 2D spatial positions
        """
        raise NotImplementedError()


# -------------- Position Types --------------


class Gaussian(Position):
    def __init__(self, init_std=0.4):
        """
        Parameters
        ----------
        init_std : float
            initial stddev
        """
        super().__init__()
        self.init_std = float(init_std)

    def _init(self, units):
        """
        Parameters
        ----------
        units : int
            number of units (U)
        """
        self.units = int(units)

        self.mu = Parameter(torch.zeros(units, 2))
        self.mu.scale = self.units
        self.mu.decay = False
        self.mu.norm_dim = 1

        self.sigma = Parameter(torch.eye(2).repeat(units, 1, 1))
        self.sigma.decay = self.units
        self.sigma.decay = False
        self.sigma.norm_dim = [1, 2]

        self._restart()
        self._reset()

    def _restart(self):
        with torch.no_grad():
            self.sigma.copy_(torch.eye(2).mul(self.init_std))

    def _reset(self):
        self._position = None

    def sample(self, batch_size=1):
        """
        Parameters
        ----------
        batch_size : int
            batch size (N)

        Returns
        -------
        Tensor
            [N, U, 2], 2D spatial positions
        """
        if self._position is None:
            x = self.mu.repeat(batch_size, 1, 1)
            x = x + torch.einsum("U C D , N U D -> N U C", self.sigma, torch.randn_like(x))
            self._position = x

        else:
            assert batch_size == self._position.size(0)

        return self._position

    @property
    def mean(self):
        """
        Returns
        -------
        Tensor
            [U, 2], 2D spatial positions
        """
        return self.mu
