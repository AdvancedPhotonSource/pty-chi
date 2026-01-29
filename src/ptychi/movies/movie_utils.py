# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional
import numpy as np

from .settings import (
    MovieFileSettings,
    MovieFileTypes,
    PlotTypes,
    PositionsMovieSettings,
)
from .mappings import prepare_movie_subject, movie_setting_types
from .io import append_array_to_h5, save_movie_to_file
import os
import h5py

import ptychi.reconstructors.base as base

dataset_name = "frames"


class MovieBuilder:
    current_frame: int = 0
    frames: np.ndarray = None
    epochs: list[int] = []

    def __init__(
        self,
        settings: movie_setting_types,
        folder: str,
        movie_name: str,
    ):
        self.settings = settings
        self.folder = folder
        self.movie_name = movie_name
        if self.settings.save_intermediate_data_to_hdf5:
            self.hdf5_file_path = os.path.join(self.folder, self.movie_name + ".hdf5")
        else:
            self.hdf5_file_path = None

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

    def save_frame_enabled_this_epoch(self, current_epoch: int):
        if current_epoch % self.settings.snapshot.stride == 0:
            return True
        else:
            return False

    def record_frame(self, movie_frame_array: np.ndarray):
        if self.settings.save_intermediate_data_to_hdf5:
            append_array_to_h5(
                array=movie_frame_array,
                file_path=self.hdf5_file_path,
                create_new_file=self.current_frame == 0,
            )
        else:
            if self.current_frame == 0:
                self.frames = movie_frame_array[None]
            else:
                self.frames = np.append(self.frames, movie_frame_array[None], axis=0)
        self.epochs += [self.current_frame * self.settings.snapshot.stride]
        self.current_frame += 1

    def create_movie(
        self, file_type: MovieFileTypes, movie_file_settings: Optional[MovieFileSettings] = None
    ):
        if movie_file_settings is None:
            movie_file_settings = self.settings.movie_file
        output_path = os.path.join(self.folder, self.movie_name + "." + file_type)
        if self.settings.save_intermediate_data_to_hdf5:
            with h5py.File(self.hdf5_file_path) as F:
                frames = F[dataset_name][:]
        else:
            frames = self.frames
            if isinstance(self.settings, PositionsMovieSettings):
                image_type = PlotTypes.LINE_PLOT
            else:
                image_type = PlotTypes.IMAGE

            # save arrays as images
            save_movie_to_file(
                array=frames,
                file_type=file_type,
                output_path=output_path,
                fps=self.settings.movie_file.fps,
                enhance_contrast=self.settings.movie_file.high_contrast,
                colormap=self.settings.movie_file.colormap,
                titles=[str(i) for i in self.epochs],
                compress=self.settings.movie_file.compress,
                upper_bound=self.settings.movie_file.upper_bound,
                lower_bound=self.settings.movie_file.lower_bound,
                image_type=image_type,
            )

    def reset(self):
        self.current_frame = 0
        self.epochs = []
        self.frames = None


class MoviesManager:
    """Class that handles the updating of `MovieBuilder` instances."""

    def __init__(self):
        self.movie_builders: dict[str, MovieBuilder] = {}

    def add_movie_builder(
        self,
        settings: movie_setting_types,
        folder: str,
        movie_name: str,
    ):
        self.movie_builders[movie_name] = MovieBuilder(settings, folder, movie_name)

    def record_all_frames(self, reconstructor: "base.Reconstructor"):
        for movie_builder in self.movie_builders.values():
            if movie_builder.save_frame_enabled_this_epoch(reconstructor.current_epoch):
                movie_frame_array = prepare_movie_subject(reconstructor, movie_builder.settings)
                movie_builder.record_frame(movie_frame_array)

    def create_all_movies(self, file_type: MovieFileTypes):
        for movie_builder in self.movie_builders.values():
            movie_builder.create_movie(file_type)

    def reset_all(self):
        for movie_builder in self.movie_builders.values():
            movie_builder.reset()

    def delete_intermediate_movie_files(self):
        for movie_builder in self.movie_builders.values():
            file_path = movie_builder.hdf5_file_path
            if file_path is not None and os.path.exists(file_path):
                os.remove(file_path)
                print(f"Intermediate file {os.path.basename(file_path)} has been deleted.")
