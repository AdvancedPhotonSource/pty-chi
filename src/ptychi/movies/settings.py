# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import dataclasses
from enum import StrEnum, auto
from typing import Sequence, Optional
import cv2


class MovieFileTypes(StrEnum):
    GIF = auto()
    MP4 = auto()


class MovieSubjectTypes(StrEnum):
    OBJECT = auto()
    PROBE = auto()


class ProcessFunctionType(StrEnum):
    PHASE = auto()
    MAGNITUDE = auto()


class ProbePlotTypes(StrEnum):
    INCOHERENT_SUM = auto()
    SEPERATE_MODES = auto()


@dataclasses.dataclass
class SnapshotSettings:
    stride: int = 1
    "The number of epochs between each frame acquisition"

    scale: int = 4
    "How much each acquired frame is downsampled by"


@dataclasses.dataclass
class MovieFileSettings:
    "Settings related to look and playback of the movie file."

    fps: int = 5
    "Frames per second in the saved movie file"

    high_contrast: bool = False

    colormap: int = cv2.COLORMAP_BONE

    compress: bool = True

    upper_bound: Optional[float] = None

    lower_bound: Optional[float] = None


@dataclasses.dataclass
class ObjectMovieSettings:
    process_function: ProcessFunctionType = ProcessFunctionType.PHASE

    slice_index: int = 0
    "Index of the object to select"

    snapshot: SnapshotSettings = dataclasses.field(default_factory=SnapshotSettings)

    movie_file: MovieFileSettings = dataclasses.field(default_factory=MovieFileSettings)

    save_intermediate_data_to_hdf5: bool = False


@dataclasses.dataclass
class ProbeMovieSettings:
    process_function: ProcessFunctionType = ProcessFunctionType.MAGNITUDE

    plot_type: ProbePlotTypes = ProbePlotTypes.INCOHERENT_SUM
    """
    Determines what type of plot to make for the probe. You can choose 
    between plotting the total intensity (INCOHERENT_SUM) and plotting
    each mode seperately (SEPERATE_MODES).
    """

    mode_indices: Optional[Sequence[int]] = None
    """
    Indices of modes to be selected for plotting. This is only used 
    when `plot_type` is `SEPERATE_MODES`.
    """

    snapshot: SnapshotSettings = dataclasses.field(default_factory=SnapshotSettings)

    movie_file: MovieFileSettings = dataclasses.field(default_factory=MovieFileSettings)

    save_intermediate_data_to_hdf5: bool = False
