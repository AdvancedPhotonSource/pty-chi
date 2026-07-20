# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from pydantic import model_validator
from pydantic.dataclasses import dataclass

import ptychi.api.options.base as base


__all__ = [
    "WorkflowOptions",
    "ProgressiveResolutionWorkflowOptions",
    "MultiscanSharedObjectWorkflowOptions",
]


@dataclass
class WorkflowOptions(base.Options):
    """Base class for options that configure a workflow."""


@dataclass
class ProgressiveResolutionWorkflowOptions(WorkflowOptions):
    """Options for a progressive-resolution reconstruction workflow."""

    num_resolution_levels: int
    """The number of resolution levels, including the full-resolution level."""

    num_epochs_all_levels: list[int]
    """The number of reconstruction epochs to run at each resolution level."""

    @model_validator(mode="after")
    def _validate_resolution_levels(self):
        if self.num_resolution_levels <= 0:
            raise ValueError("`num_resolution_levels` must be greater than 0.")
        if len(self.num_epochs_all_levels) != self.num_resolution_levels:
            raise ValueError(
                "`num_epochs_all_levels` must contain one value for each resolution level."
            )
        if any(num_epochs <= 0 for num_epochs in self.num_epochs_all_levels):
            raise ValueError("All values in `num_epochs_all_levels` must be greater than 0.")
        return self


@dataclass
class MultiscanSharedObjectWorkflowOptions(WorkflowOptions):
    """Options for a multiscan reconstruction with a shared object."""

    num_outer_epochs: int
    """The number of complete passes over all scans."""

    num_inner_epochs: int
    """The number of epochs to run for each scan in each pass."""

    @model_validator(mode="after")
    def _validate_epochs(self):
        if self.num_outer_epochs <= 0:
            raise ValueError("`num_outer_epochs` must be greater than 0.")
        if self.num_inner_epochs <= 0:
            raise ValueError("`num_inner_epochs` must be greater than 0.")
        return self
