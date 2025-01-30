import torch
import numpy as np
from itertools import repeat
from .modules import Module


# -------------- Network Base --------------


class Network(Module):
    """Network"""

    def _init(self, stimuli, perspectives, modulations, units, streams):
        """
        Parameters
        ----------
        stimuli : int
            stimulus channels (C)
        perspectives : int
            perspective features (P)
        modulations : int
            modulation features (M)
        units : int
            number of units (U)
        streams : int
            number of streams (S)
        """
        raise NotImplementedError()

    @property
    def default_perspective(self):
        """
        Returns
        -------
        1D array
            [P] -- default perspective value
        """
        raise NotImplementedError()

    @property
    def default_modulation(self):
        """
        Returns
        -------
        1D array
            [M] -- default modulation value
        """
        raise NotImplementedError()

    def forward(self, stimulus, perspective, modulation, stream=None):
        """
        Parameters
        ----------
        stimulus : ND Tensor
            [N, C, ...] -- stimulus frame
        perspective : 2D Tensor
            [N, P] -- perspective frame
        modulation : 2D Tensor
            [N, M] -- modulation frame
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        2D Tensor
            [N, U] -- response frame
        """
        raise NotImplementedError()

    def loss(self, stimulus, perspective, modulation, unit, stream=None):
        """
        Parameters
        ----------
        stimulus : ND Tensor
            [N, C, ...] -- stimulus frame
        perspective : 2D Tensor
            [N, P] -- perspective frame
        modulation : 2D Tensor
            [N, M] -- modulation frame
        unit : 2D Tensor
            [N, U] --  unit frame
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, U] -- loss frame
        """
        raise NotImplementedError()

    def generate_response(
        self,
        stimuli,
        perspectives=None,
        modulations=None,
        training=False,
        reset=True,
    ):
        """
        Parameters
        ----------
        stimuli : Iterable[ND array]
            T x [...]
        perspectives : Iterable[1D|2D array] | None
            T x [P] (singular) | T x [N, P] (batch)
        modulations : Iterable[1D|2D array] | None
            T x [M] (singular) | T x [N, M] (batch)
        training : bool
            training or inference mode
        reset : bool
            reset or continue state

        Yields
        ------
        either 1D array
            [U] (singular input, training=False)
        or 1D Tensor
            [U] (singular input, training=True)
        or 2D array
            [N, U] (batch input, training=False)
        or 2D Tensor
            [N, U] (batch input, training=True)
        """
        raise NotImplementedError()

    def generate_loss(
        self,
        units,
        stimuli,
        perspectives=None,
        modulations=None,
        stream=None,
        training=False,
        reset=True,
    ):
        """
        Parameters
        ----------
        units : Iterable[1D|2D array]
            T x [U] (singular) | T x [N, U] (batch)
        stimuli : Iterable[ND array]
            T x [...]
        perspectives : Iterable[1D|2D array] | None
            T x [P] (singular) | T x [N, P] (batch)
        modulations : Iterable[1D|2D array] | None
            T x [M] (singular) | T x [N, M] (batch)
        stream : int | None
            specific stream (int) or all streams (None)
        training : bool
            training or inference mode
        reset : bool
            reset or continue state

        Yields
        ------
        either 1D array
            [U] (singular input, training=False)
        or 1D Tensor
            [U] (singular input, training=True)
        or 2D array
            [N, U] (batch input, training=False)
        or 2D Tensor
            [N, U] (batch input, training=True)
        """
        raise NotImplementedError()


# -------------- Network Types --------------


class Visual(Network):
    """Visual Network"""

    def __init__(self, core, perspective, modulation, readout, reduce, unit):
        """
        Parameters
        ----------
        core : fnn.model.cores.Core
            core model
        perspective : fnn.model.perspectives.Perspective
            perspective model
        modulation : fnn.model.modulations.Modulation
            modulation model
        readout : fnn.model.readouts.Readout
            readout model
        reduce : fnn.model.reductions.Reduce
            stream reduction
        unit : fnn.model.units.Unit
            neuronal unit
        """
        super().__init__()
        self.core = core
        self.perspective = perspective
        self.modulation = modulation
        self.readout = readout
        self.reduce = reduce
        self.unit = unit

    def _init(self, stimuli, perspectives, modulations, units, streams):
        """
        Parameters
        ----------
        stimuli : int
            stimulus channels
        perspectives : int
            perspective features
        modulations : int
            modulation features
        units : int
            number of units
        streams : int
            number of streams
        """
        self.stimuli = int(stimuli)
        self.perspectives = int(perspectives)
        self.modulations = int(modulations)
        self.units = int(units)
        self.streams = int(streams)

        self.perspective._init(
            stimuli=stimuli,
            perspectives=self.perspectives,
        )
        self.modulation._init(
            modulations=self.modulations,
            streams=self.streams,
        )
        self.core._init(
            perspectives=self.perspective.channels,
            modulations=self.modulation.features,
            streams=self.streams,
        )
        self.readout._init(
            cores=self.core.channels,
            readouts=self.unit.readouts,
            units=self.units,
            streams=self.streams,
        )
        self.reduce._init(
            dim=[1],
            keepdim=False,
        )

    @property
    def default_perspective(self):
        """
        Returns
        -------
        1D array
            [P] -- default perspective value
        """
        return np.zeros([self.perspectives])

    @property
    def default_modulation(self):
        """
        Returns
        -------
        1D array
            [M] -- default modulation value
        """
        return np.zeros([self.modulations])

    def _raw(self, stimulus, perspective, modulation, stream=None, periphery="dark"):
        """
        Parameters
        ----------
        stimulus : 4D Tensor
            [N, C, H, W] -- stimulus frame
        perspective : 2D Tensor
            [N, P] -- perspective frame
        modulations : 2D Tensor
            [N, M] -- modulation frame
        stream : int | None
            specific stream (int) or all streams (None)
        periphery : str
            "dark" | "extend"

        Returns
        -------
        3D Tensor
            [N, U, R] -- raw output
        """
        if periphery == "dark":
            perspective = self.perspective(
                stimulus=stimulus,
                perspective=perspective,
                pad_mode="zeros",
            )
        elif periphery == "extend":
            perspective = self.perspective(
                stimulus=stimulus,
                perspective=perspective,
                pad_mode="replicate",
            )
        else:
            raise ValueError(f"Invalid periphery -- {periphery}")

        if stream is None:
            perspective = perspective.repeat(1, self.streams, 1, 1)
            modulation = modulation.repeat(1, self.streams)

        modulation = self.modulation(
            modulation=modulation,
            stream=stream,
        )
        core = self.core(
            perspective=perspective,
            modulation=modulation,
            stream=stream,
        )
        readout = self.readout(
            core=core,
            stream=stream,
        )
        if stream is None:
            return self.reduce(readout)
        else:
            return readout

    def forward(self, stimulus, perspective, modulation, stream=None, periphery="dark"):
        """
        Parameters
        ----------
        stimulus : 4D Tensor
            [N, C, H, W] -- stimulus frame
        perspective : 2D Tensor
            [N, P] -- perspective frame
        modulations : 2D Tensor
            [N, M] -- modulation frame
        stream : int | None
            specific stream (int) or all streams (None)
        periphery : str
            "dark" | "extend"

        Returns
        -------
        2D Tensor
            [N, U] -- response frame
        """
        r = self._raw(
            stimulus=stimulus,
            perspective=perspective,
            modulation=modulation,
            stream=stream,
            periphery=periphery,
        )
        return self.unit(readout=r)

    def loss(self, stimulus, perspective, modulation, unit, stream=None):
        """
        Parameters
        ----------
        stimulus : 4D Tensor
            [N, C, H, W] -- stimulus frame
        perspective : 2D Tensor
            [N, P] -- perspective frame
        modulations : 2D Tensor
            [N, M] -- modulation frame
        unit : 2D Tensor
            [N, U] --  unit frame
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        2D Tensor
            [N, U] -- loss frame
        """
        r = self._raw(
            stimulus=stimulus,
            perspective=perspective,
            modulation=modulation,
            stream=stream,
        )
        return self.unit.loss(readout=r, unit=unit)

    def to_tensor(self, stimulus, perspective=None, modulation=None):
        """
        Parameters
        ----------
        stimulus : 2D|3D|4D array
            [H, W] | [H, W, C] | [N, H, W, C]
        perspective : 1D|2D array | None
            [P] | [N, P]
        modulations : 1D|2D array | None
            [M] | [N, M]

        Returns
        -------
        4D Tensor
            [N, C, H, W]
        2D Tensor
            [N, P]
        2D Tensor
            [N, M]
        bool
            squeeze response batch dim
        """
        assert stimulus.dtype == np.uint8
        stimulus = stimulus / 255

        if stimulus.ndim == 2:
            stimulus = stimulus[None, None]
            squeeze = True
            N = 1
        elif stimulus.ndim == 3:
            stimulus = np.einsum("H W C -> C H W", stimulus)[None]
            squeeze = True
            N = 1
        else:
            stimulus = np.einsum("N H W C -> N C H W", stimulus)
            squeeze = False
            N = stimulus.shape[0]

        if perspective is None:
            perspective = self.default_perspective[None]
        elif perspective.ndim == 1:
            perspective = perspective[None]
        else:
            squeeze = False
            N = max(N, perspective.shape[0])

        if modulation is None:
            modulation = self.default_modulation[None]
        elif modulation.ndim == 1:
            modulation = modulation[None]
        else:
            squeeze = False
            N = max(N, modulation.shape[0])

        device = self.device
        tensor = lambda x: torch.tensor(x, dtype=torch.float, device=device)

        stimulus = tensor(stimulus).expand(N, -1, -1, -1)
        perspective = tensor(perspective).expand(N, -1)
        modulation = tensor(modulation).expand(N, -1)

        return stimulus, perspective, modulation, squeeze

    def generate_response(
        self,
        stimuli,
        perspectives=None,
        modulations=None,
        training=False,
        reset=True,
    ):
        """
        Parameters
        ----------
        stimuli : Iterable[2D|3D|4D array]
            T x [H, W] (singular) | T x [H, W, C] (singular) | T x [N, H, W, C] (batch) --- dtype=uint8
        perspectives : Iterable[1D|2D array] | None
            T x [P] (singular) | T x [N, P] (batch) --- dtype=float
        modulations : Iterable[1D|2D array] | None
            T x [M] (singular) | T x [N, M] (batch) --- dtype=float
        training : bool
            training or inference mode
        reset : bool
            reset or continue state

        Yields
        ------
        either 1D array
            [U] (singular input, training=False)
        or 1D Tensor
            [U] (singular input, training=True)
        or 2D array
            [N, U] (batch input, training=False)
        or 2D Tensor
            [N, U] (batch input, training=True)
        """
        if reset:
            self.reset()

        if perspectives is None:
            perspectives = repeat(None)

        if modulations is None:
            modulations = repeat(None)

        with self.train_context(training):

            for stimulus, perspective, modulation in zip(
                stimuli, perspectives, modulations
            ):

                *tensors, squeeze = self.to_tensor(stimulus, perspective, modulation)

                response = self(*tensors)
                if squeeze:
                    response = response.squeeze(0)

                if training:
                    yield response
                else:
                    yield response.cpu().numpy()

    def generate_loss(
        self,
        units,
        stimuli,
        perspectives=None,
        modulations=None,
        stream=None,
        training=False,
        reset=True,
    ):
        """
        Parameters
        ----------
        units : Iterable[1D|2D array]
            T x [U] (singular) | T x [N, U] (batch) -- dtype=float
        stimuli : Iterable[2D|3D|4D array]
            T x [H, W] (singular) | T x [H, W, C] (singular) | T x [N, H, W, C] (batch) --- dtype=uint8
        perspectives : Iterable[1D|2D array] | None
            T x [P] (singular) | T x [N, P] (batch) --- dtype=float
        modulations : Iterable[1D|2D array] | None
            T x [M] (singular) | T x [N, M] (batch) --- dtype=float
        stream : int | None
            specific stream (int) or all streams (None)
        training : bool
            training or inference mode
        reset : bool
            reset or continue state

        Yields
        ------
        either 1D array
            [U] (singular input, training=False)
        or 1D Tensor
            [U] (singular input, training=True)
        or 2D array
            [N, U] (batch input, training=False)
        or 2D Tensor
            [N, U] (batch input, training=True)
        """
        if reset:
            self.reset()

        if perspectives is None:
            perspectives = repeat(None)

        if modulations is None:
            modulations = repeat(None)

        with self.train_context(training):

            device = self.device

            for unit, stimulus, perspective, modulation in zip(
                units, stimuli, perspectives, modulations
            ):

                unit = torch.tensor(unit, dtype=torch.float, device=device)
                if unit.ndim == 1:
                    unit = unit[None]
                    squeeze = True
                else:
                    squeeze = False

                *tensors, _squeeze = self.to_tensor(stimulus, perspective, modulation)
                assert squeeze == _squeeze

                loss = self.loss(*tensors, unit=unit, stream=stream)
                if squeeze:
                    loss = loss.squeeze(0)

                if training:
                    yield loss
                else:
                    yield loss.cpu().numpy()

    def predict(self, stimuli, perspectives=None, modulations=None):
        """
        Parameters
        ----------
        stimuli : Iterable[2D|3D|4D array]
            T x [H, W] (singular) | T x [H, W, C] (singular) | T x [N, H, W, C] (batch) --- dtype=uint8
        perspectives : Iterable[1D|2D array] | None
            T x [P] (singular) | T x [N, P] (batch) --- dtype=float
        modulations : Iterable[1D|2D array] | None
            T x [M] (singular) | T x [N, M] (batch) --- dtype=float
        training : bool
            training or inference mode
        reset : bool
            reset or continue state

        Returns
        -------
        2D array | 3D array
            [T, U] (singular input) | [T, N, U] (batch input) -- dtype=float
        """
        response = self.generate_response(stimuli, perspectives, modulations)
        return np.array([*response])
