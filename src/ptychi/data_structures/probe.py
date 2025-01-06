from typing import Optional, Union, TYPE_CHECKING
import logging
import os

import numpy as np
import tifffile
import torch
from torch import Tensor

import ptychi.image_proc as ip
import ptychi.maths as pmath
import ptychi.data_structures.base as ds
import ptychi.data_structures.object as object
import ptychi.data_structures.opr_mode_weights as oprweights
import ptychi.api.enums as enums
from ptychi.propagate import FourierPropagator, WavefieldPropagator

if TYPE_CHECKING:
    import ptychi.api as api

logger = logging.getLogger(__name__)


class Probe(ds.ReconstructParameter):
    # TODO: eigenmode_update_relaxation is only used for LSQML. We should create dataclasses
    # to contain additional options for ReconstructParameter classes, and subclass them for specific
    # reconstruction algorithms - for example, ProbeOptions -> LSQMLProbeOptions.
    options: "api.options.base.ProbeOptions"

    def __init__(
        self,
        name: str = "probe",
        options: "api.options.base.ProbeOptions" = None,
        *args,
        **kwargs,
    ):
        """
        Represents the probe function in a tensor of shape
            `(n_opr_modes, n_modes, h, w)`
        where:
        - n_opr_modes is the number of mutually coherent probe modes used in orthogonal
          probe relaxation (OPR).
        - n_modes is the number of mutually incoherent probe modes.
        """
        super().__init__(*args, name=name, options=options, is_complex=True, **kwargs)
        if len(self.shape) != 4:
            raise ValueError("Probe tensor must be of shape (n_opr_modes, n_modes, h, w).")

        self.probe_power = options.power_constraint.target_power
        self.probe_power_constraint_stride = options.power_constraint.stride
        self.orthogonalize_incoherent_modes = options.orthogonalize_incoherent_modes.enabled
        self.orthogonalize_incoherent_modes_stride = options.orthogonalize_incoherent_modes.stride
        self.orthogonalize_incoherent_modes_method = options.orthogonalize_incoherent_modes.method

    def shift(self, shifts: Tensor):
        """
        Generate shifted probe.

        Parameters
        ----------
        shifts : Tensor
            A tensor of shape (2,) or (N, 2) giving the shifts in pixels.
            If a (N, 2)-shaped tensor is given, a batch of shifted probes are generated.

        Returns
        -------
        shifted_probe : Tensor
            Shifted probe.
        """
        if shifts.ndim == 1:
            probe_straightened = self.tensor.complex().view(-1, *self.shape[-2:])
            shifted_probe = ip.fourier_shift(
                probe_straightened, shifts[None, :].repeat([[probe_straightened.shape[0], 1, 1]])
            )
            shifted_probe = shifted_probe.view(*self.shape)
        else:
            n_shifts = shifts.shape[0]
            n_images_each_probe = self.shape[0] * self.shape[1]
            probe_straightened = self.tensor.complex().view(n_images_each_probe, *self.shape[-2:])
            probe_straightened = probe_straightened.repeat(n_shifts, 1, 1)
            shifts = shifts.repeat_interleave(n_images_each_probe, dim=0)
            shifted_probe = ip.fourier_shift(probe_straightened, shifts)
            shifted_probe = shifted_probe.reshape(n_shifts, *self.shape)
        return shifted_probe

    @property
    def n_modes(self):
        return self.tensor.shape[1]

    @property
    def n_opr_modes(self):
        return self.tensor.shape[0]

    @property
    def has_multiple_opr_modes(self):
        return self.n_opr_modes > 1

    @property
    def has_multiple_incoherent_modes(self):
        return self.n_modes > 1

    def get_mode(self, mode: int, keepdim: bool = False):
        mode = self.tensor.complex()[:, mode]
        if keepdim:
            mode = mode.unsqueeze(1)
        return mode

    def get_opr_mode(self, mode: int, keepdim: bool = False):
        mode = self.tensor.complex()[mode]
        if keepdim:
            mode = mode.unsqueeze(0)
        return mode

    def get_mode_and_opr_mode(self, mode: int, opr_mode: int):
        return self.tensor.complex()[opr_mode, mode]

    def get_spatial_shape(self):
        return self.shape[-2:]

    def get_all_mode_intensity(
        self,
        opr_mode: Optional[int] = 0,
        weights: Optional[Union[Tensor, "ds.ReconstructParameter"]] = None,
    ) -> Tensor:
        """
        Get the intensity of all probe modes.

        Parameters
        ----------
        opr_mode : Optional[int]
            the OPR mode. If this is not None, only the intensity of the chosen
            OPR mode is calculated. Otherwise, it calculates the intensity of the weighted sum
            of all OPR modes. In that case, `weights` must be given.
        weights : Optional[Union[Tensor, ReconstructParameter]]
            a (n_opr_modes,) tensor giving the weights of OPR modes.

        Returns
        -------
        intensity : Tensor
            The summed intensity of all probe modes.
        """
        if isinstance(weights, oprweights.OPRModeWeights):
            weights = weights.data
        if opr_mode is not None:
            p = self.data[opr_mode]
        else:
            p = (self.data * weights[None, :, :, :]).sum(0)
        return torch.sum((p.abs()) ** 2, dim=0)

    def get_unique_probes(
        self, weights: Union[Tensor, "ds.ReconstructParameter"], mode_to_apply: Optional[int] = None
    ) -> Tensor:
        """
        Parameters
        ----------
        weights : Tensor
            A tensor giving the weights of the eigenmodes.
        mode_to_apply : int, optional
            The incoherent mode for which OPR modes should be applied. The data
            for other modes will be set to the value of the first OPR mode. If None,
            OPR correction will be done to all incoherent modes.

        Returns
        -------
        unique_probe : Tensor
            A tensor of unique probes. If weights.ndim == 2, the shape is (n_points, n_modes, h, w).
            If weights.ndim == 1, the shape is (n_modes, h, w).
        """
        if isinstance(weights, oprweights.OPRModeWeights):
            weights = weights.data

        p_orig = None
        if mode_to_apply is not None:
            p_orig = self.data.clone()
            p = p_orig[:, [mode_to_apply], :, :]
        else:
            p = self.data.clone()
        if weights.ndim == 1:
            unique_probe = p * weights[:, None, None, None]
            unique_probe = unique_probe.sum(0)
        else:
            unique_probe = p[None, ...] * weights[:, :, None, None, None]
            unique_probe = unique_probe.sum(1)

        # If OPR is only applied on one incoherent mode, add in the rest of the modes.
        if mode_to_apply is not None:
            if weights.ndim == 1:
                # Shape of unique_probe:     (1, h, w)
                p_orig[0, [mode_to_apply], :, :] = unique_probe
                unique_probe = p_orig[0, ...]
            else:
                # Shape of unique_probe:     (n_points, 1, h, w)
                p_orig = p_orig[0:1, ...].repeat(weights.shape[0], 1, 1, 1)
                p_orig[:, [mode_to_apply], :, :] = unique_probe
                unique_probe = p_orig
        return unique_probe

    def _get_probe_mode_slicer(self, mode_index=None):
        if mode_index is None:
            return slice(None)
        else:
            return slice(mode_index, mode_index + 1)

    def incoherent_mode_orthogonality_constraint_enabled(self, current_epoch: int) -> bool:
        if (
            self.has_multiple_incoherent_modes
            and self.orthogonalize_incoherent_modes
            and current_epoch >= self.optimization_plan.start
            and (current_epoch - self.optimization_plan.start)
            % self.orthogonalize_incoherent_modes_stride
            == 0
        ):
            return True
        else:
            return False

    def constrain_incoherent_modes_orthogonality(self):
        """Orthogonalize the incoherent probe modes for the first OPR mode."""
        probe = self.data

        norm_first_mode_orig = pmath.norm(probe[0, 0], dim=(-2, -1))

        if self.orthogonalize_incoherent_modes_method == "gs":
            func = pmath.orthogonalize_gs
        elif self.orthogonalize_incoherent_modes_method == "svd":
            func = pmath.orthogonalize_svd
        else:
            raise NotImplementedError(
                f"Orthogonalization method {self.orthogonalize_incoherent_modes_method} "
                "is not supported."
            )
        probe[0] = func(
            probe[0],
            dim=(-2, -1),
            group_dim=0,
        )

        # Restore norm.
        norm_first_mode_new = pmath.norm(probe[0, 0], dim=(-2, -1))
        probe = probe * norm_first_mode_orig / norm_first_mode_new

        self.set_data(probe)

    def opr_mode_orthogonalization_enabled(self, current_epoch: int) -> bool:
        return (
            self.optimization_enabled(current_epoch)
            and self.has_multiple_opr_modes
            and self.options.orthogonalize_opr_modes.enabled
            and self.options.orthogonalize_opr_modes.optimization_plan.is_enabled(current_epoch)
        )

    def constrain_opr_mode_orthogonality(
        self, weights: Union[Tensor, ds.ReconstructParameter], eps=1e-5
    ):
        """
        Add the following constraints to variable probe weights

        1. Remove outliars from weights
        2. Enforce orthogonality once per epoch
        3. Sort the variable probes by their total energy
        4. Normalize the variable probes so the energy is contained in the weight

        Adapted from Tike (https://github.com/AdvancedPhotonSource/tike). The implementation
        in Tike assumes a separately stored variable probe eigenmodes; here we use the
        PtychoShelves convention and regard the second and following OPR modes as eigenmodes.

        Also, this function assumes that OPR correction is only applied to the first
        incoherent mode when mixed state probe is used, as this is what PtychoShelves does.
        OPR modes of other incoherent modes are ignored, for now.

        Parameters
        ----------
        weights : Tensor
            A (n_points, n_opr_modes) tensor of weights.
        :param weights: a (n_points, n_opr_modes) tensor of weights.

        Returns
        -------
        Tensor
            Normalized and sorted OPR mode weights.
        """
        if isinstance(weights, oprweights.OPRModeWeights):
            weights = weights.data

        # The main mode of the probe is the first OPR mode, while the
        # variable part of the probe is the second and following OPR modes.
        # The main mode should not change during orthogonalization, but the
        # variable OPR modes should all be orthogonal to it.
        probe = self.data

        # TODO: remove outliars by polynomial fitting (remove_variable_probe_ambiguities.m)

        # Normalize eigenmodes and adjust weights.
        eigenmodes = probe[1:, ...]
        vnorm = pmath.mnorm(eigenmodes, dim=(-2, -1), keepdims=True)
        eigenmodes /= vnorm + eps
        # Shape of weights:      (n_points, n_opr_modes).
        # Currently, only the first incoherent mode has OPR modes, and the
        # stored weights are for that mode.
        weights[:, 1:] = weights[:, 1:] * vnorm[:, 0, 0, 0]

        # Orthogonalize variable probes. With Gram-Schmidt, the first
        # OPR mode (i.e., the main mode) should not change during orthogonalization.
        probe = pmath.orthogonalize_gs(
            probe,
            dim=(-2, -1),
            group_dim=0,
        )

        if False:
            # Compute the energies of variable OPR modes (i.e., the second and following)
            # in order to sort probes by energy.
            # Shape of power:         (n_opr_modes - 1,).
            power = pmath.norm(weights[..., 1:], dim=0) ** 2

            # Sort the probes by energy
            sorted = torch.argsort(-power)
            weights[:, 1:] = weights[:, sorted + 1]
            # Apply only to the first incoherent mode.
            probe[1:, 0, :, :] = probe[sorted + 1, 0, :, :]

        # Remove outliars from variable probe weights.
        aevol = torch.abs(weights)
        weights = torch.minimum(
            aevol,
            1.5
            * torch.quantile(
                aevol,
                0.95,
                dim=0,
                keepdims=True,
            ).type(weights.dtype),
        ) * torch.sign(weights)

        # Update stored data.
        self.set_data(probe)
        return weights

    def probe_power_constraint_enabled(self, current_epoch: int) -> bool:
        if (
            self.probe_power > 0.0
            and current_epoch >= self.optimization_plan.start
            and (current_epoch - self.optimization_plan.start) % self.probe_power_constraint_stride
            == 0
        ):
            return True
        else:
            return False

    def constrain_probe_power(
        self,
        object_: "object.Object",
        opr_mode_weights: Union[Tensor, "oprweights.OPRModeWeights"],
        propagator: Optional[WavefieldPropagator] = None,
    ) -> None:
        if self.probe_power <= 0.0:
            return

        if isinstance(opr_mode_weights, oprweights.OPRModeWeights):
            opr_mode_weights = opr_mode_weights.data

        if propagator is None:
            propagator = FourierPropagator()

        # Shape of probe_composed:        (n_modes, h, w)
        if self.has_multiple_opr_modes:
            avg_weights = opr_mode_weights.mean(dim=0)
            probe_composed = self.get_unique_probes(avg_weights, mode_to_apply=0)
        else:
            probe_composed = self.get_opr_mode(0)

        # TODO: use propagator for forward simulation
        propagated_probe = propagator.propagate_forward(probe_composed)
        propagated_probe_power = torch.sum(propagated_probe.abs() ** 2)
        power_correction = torch.sqrt(self.probe_power / propagated_probe_power)

        self.set_data(self.data * power_correction)
        object_.set_data(object_.data / power_correction)

        logger.info("Probe and object scaled by {}.".format(power_correction))

    def constrain_support(self):
        """
        Apply probe support constraint through a mask generated by thresholding the
        blurred probe magnitude.
        """
        data = self.data
        mask = ip.gaussian_filter(data, sigma=3, size=5).abs()
        thresh = (
            mask.max(-1, keepdim=True).values.max(-2, keepdim=True).values
            * self.options.support_constraint.threshold
        )
        mask = torch.where(mask > thresh, 1.0, 0.0)
        mask = ip.gaussian_filter(mask, sigma=2, size=3).abs()
        data = data * mask
        self.set_data(data)

    def support_constraint_enabled(self, current_epoch: int) -> bool:
        if (
            self.options.support_constraint.enabled
            and current_epoch >= self.optimization_plan.start
            and (current_epoch - self.optimization_plan.start)
            % self.options.support_constraint.stride
            == 0
        ):
            return True
        else:
            return False

    def post_update_hook(self) -> None:
        super().post_update_hook()

    def normalize_eigenmodes(self):
        """
        Normalize all eigenmodes (the second and following OPR modes) such that each of them
        has a squared norm equal to the number of pixels in the probe.
        """
        if not self.has_multiple_opr_modes:
            return
        eigen_modes = self.data[1:, ...]
        for i_opr_mode in range(eigen_modes.shape[0]):
            for i_mode in range(eigen_modes.shape[1]):
                eigen_modes[i_opr_mode, i_mode, :, :] /= (
                    pmath.mnorm(eigen_modes[i_opr_mode, i_mode, :, :], dim=(-2, -1)) + 1e-8
                )

        new_data = self.data
        new_data[1:, ...] = eigen_modes
        self.set_data(new_data)

    def save_tiff(self, path: str):
        """
        Save the probe's magnitude and phase as 2 TIFF files. Each file contains
        an array of tiles, where the rows correspond to incoherent probe modes
        and columns correspond to OPR modes.

        Parameters
        ----------
        path : str
            Path to save. "_phase" and "_mag" will be appended to the filename.
        """
        fname = os.path.splitext(path)[0]
        mag_img = np.empty([self.shape[3] * self.shape[1], self.shape[2] * self.shape[0]])
        phase_img = np.empty([self.shape[3] * self.shape[1], self.shape[2] * self.shape[0]])
        data = self.data
        for i_mode in range(self.shape[1]):
            for i_opr_mode in range(self.shape[0]):
                mag_img[
                    i_mode * self.shape[3] : (i_mode + 1) * self.shape[3],
                    i_opr_mode * self.shape[2] : (i_opr_mode + 1) * self.shape[2],
                ] = data[i_opr_mode, i_mode, :, :].abs().detach().cpu().numpy()
                phase_img[
                    i_mode * self.shape[3] : (i_mode + 1) * self.shape[3],
                    i_opr_mode * self.shape[2] : (i_opr_mode + 1) * self.shape[2],
                ] = torch.angle(data[i_opr_mode, i_mode, :, :]).detach().cpu().numpy()
        tifffile.imsave(fname + "_mag.tif", mag_img)
        tifffile.imsave(fname + "_phase.tif", phase_img)
