# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

from typing import Optional
from .movie_utils import MoviesManager
from .mappings import movie_setting_types
from .settings import MovieFileTypes, MovieFileSettings
from ptychi.timing.timer_utils import timer
import ptychi.reconstructors.base as base

MOVIES_MANAGER = MoviesManager()
ENABLE_MOVIES = False


def toggle_movies(enable: bool):
    global ENABLE_MOVIES
    ENABLE_MOVIES = enable


def clear_movie_globals():
    global MOVIES_MANAGER
    MOVIES_MANAGER = MoviesManager()


def add_new_movie_builder(settings: movie_setting_types, folder: str, movie_name: str):
    global MOVIES_MANAGER
    if MOVIES_MANAGER is None:
        MOVIES_MANAGER = MoviesManager()
    MOVIES_MANAGER.add_movie_builder(settings, folder, movie_name)


@timer()
def update_movie_builders(reconstructor: "base.Reconstructor"):
    global MOVIES_MANAGER
    MOVIES_MANAGER.record_all_frames(reconstructor)


def create_all_movies(file_type: MovieFileTypes):
    global MOVIES_MANAGER
    MOVIES_MANAGER.create_all_movies(file_type)


def create_movie(
    movie_name: str,
    file_type: MovieFileTypes,
    movie_file_settings: Optional[MovieFileSettings] = None,
):
    if not check_if_movie_builder_exists(movie_name):
        return
    global MOVIES_MANAGER
    MOVIES_MANAGER.movie_builders[movie_name].create_movie(file_type, movie_file_settings)


def get_movie_builder_settings(movie_name: str) -> movie_setting_types:
    if not check_if_movie_builder_exists(movie_name):
        return
    global MOVIES_MANAGER
    return MOVIES_MANAGER.movie_builders[movie_name].settings


def reset_movie_builders():
    global MOVIES_MANAGER
    MOVIES_MANAGER.reset_all()


def delete_intermediate_movie_files():
    global MOVIES_MANAGER
    MOVIES_MANAGER.delete_intermediate_movie_files()


def check_if_movie_builder_exists(movie_name: str) -> bool:
    global MOVIES_MANAGER
    if movie_name not in MOVIES_MANAGER.movie_builders.keys():
        print(f"{movie_name} is not found")
        return False
    else:
        return True
