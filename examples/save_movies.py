import ptychi.movies as movies
if not movies.MOVIES_INSTALLED:
    raise ImportError("Dependencies needed by ptychi.movies are not installed!")

import sys
import os
import torch
import numpy as np
import cv2
import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size
from ptychi.global_settings import get_default_complex_dtype


# Add the folder containing "tests/" to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.test_utils import TungstenDataTester

from ptychi.movies.settings import (
    ProbePlotTypes,
    ProcessFunctionType,
    ProbeMovieSettings,
    ObjectMovieSettings,
    MovieFileTypes,
    SnapshotSettings,
    MovieFileSettings,
)

movie_folder = input("Enter a folder to save movies to:\n")

### Movie setup ####
# Enable movies
movies.api.toggle_movies(True)
### Initialize movie builders ###
# The scale determines the level of downsampling
scale = 1
# 1) Add movie builder for total probe intensity
probe_sum_movie_name = "probe_total_intensity"
movies.api.add_new_movie_builder(
    settings=ProbeMovieSettings(snapshot=SnapshotSettings(scale=scale)),
    folder=movie_folder,
    movie_name=probe_sum_movie_name,
)
# You can change settings after initializing the MovieBuilder object by
# using the get_movie_builder_settings function to retrieve the settings
probe_movie_file_settings = movies.api.get_movie_builder_settings(probe_sum_movie_name).movie_file
probe_movie_file_settings.colormap = cv2.COLORMAP_VIRIDIS

# 2) Add movie builder for showing all probe mode magnitudes
probe_modes_movie_name = "probe_modes_magnitude"
movies.api.add_new_movie_builder(
    settings=ProbeMovieSettings(
        plot_type=ProbePlotTypes.SEPERATE_MODES,
        snapshot=SnapshotSettings(scale=scale),
        movie_file=MovieFileSettings(colormap=cv2.COLORMAP_VIRIDIS),
    ),
    folder=movie_folder,
    movie_name=probe_modes_movie_name,
)

# 3) Add movie builder for showing all probe mode phases
probe_modes_movie_name = "probe_modes_phase"
movies.api.add_new_movie_builder(
    settings=ProbeMovieSettings(
        process_function=ProcessFunctionType.PHASE,
        plot_type=ProbePlotTypes.SEPERATE_MODES,
        snapshot=SnapshotSettings(scale=scale),
        movie_file=MovieFileSettings(colormap=cv2.COLORMAP_JET),
    ),
    folder=movie_folder,
    movie_name=probe_modes_movie_name,
)

# 4) Add movie builder for object phase (this is the default for object movie settings)
movies.api.add_new_movie_builder(
    settings=ObjectMovieSettings(snapshot=SnapshotSettings(scale=scale)),
    folder=movie_folder,
    movie_name="reconstructed_object_phase",
)

# 5) Add movie builder for object magnitude
# Enabling save_intermediate_data_to_hdf5 means that each frame of the
# movie will *not* be saved to a variable in python, instead it will be
# appended to a dataset in a hdf5 each iteration. This is useful if you
# know the array of saved frames will be very large and therefore would take
# up a lot of memory.
movies.api.add_new_movie_builder(
    settings=ObjectMovieSettings(
        process_function=ProcessFunctionType.MAGNITUDE,
        snapshot=SnapshotSettings(scale=scale),
        save_intermediate_data_to_hdf5=True,
    ),
    folder=movie_folder,
    movie_name="reconstructed_object_magnitude",
)

### Run ptychography reconstruction ###
demo_tester = TungstenDataTester()
demo_tester.load_tungsten_data()
data, probe, pixel_size_m, positions_px = demo_tester.load_tungsten_data(additional_opr_modes=0)

options = api.DMOptions()
options.data_options.data = data

options.object_options.initial_guess = torch.ones(
    [1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=100)],
    dtype=get_default_complex_dtype(),
)
options.object_options.pixel_size_m = pixel_size_m
options.object_options.optimizable = True

options.probe_options.initial_guess = probe
options.probe_options.power_constraint.probe_power = np.sum(np.max(data, axis=-3), axis=(-2, -1))
options.probe_options.power_constraint.enabled = True
options.probe_options.optimizable = True

options.probe_position_options.position_x_px = positions_px[:, 1]
options.probe_position_options.position_y_px = positions_px[:, 0]
options.probe_position_options.optimizable = True

options.reconstructor_options.num_epochs = 20
options.reconstructor_options.chunk_length = 100

task = PtychographyTask(options)
task.run()


### Generate movie files ###
movies.api.create_all_movies(MovieFileTypes.GIF)
# You don't need to use the StrEnum, you could just use: movies.api.create_all_movies("gif")
# Check MovieFileTypes to see what file formats are available

### Delete intermediate movie files after generating movies ###
movies.api.delete_intermediate_movie_files()
