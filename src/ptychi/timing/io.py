# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import os
import datetime
from typing import Union
import h5py
import numpy as np
import socket
import subprocess

# from ptychi.timer_utils import ADVANCED_TIME_DICT, ELAPSED_TIME_DICT, toggle_timer
import ptychi.timing.timer_utils as timer_utils


def get_timestamp_for_timing_files() -> str:
    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    date_string = current_datetime.strftime("%Y-%m-%d")
    time_string = current_datetime.strftime("%H-%M-%S")
    return timestamp, date_string, time_string


def save_timing_data_to_unique_file_path(name: str, folder: str):
    timing_results_folder = os.path.join(folder, name)
    if not os.path.exists(timing_results_folder):
        os.makedirs(timing_results_folder)

    if "TIMESTAMP" in os.environ:
        timestamp, date_string, time_string = (
            os.environ["TIMESTAMP"],
            os.environ["DATE_STRING"],
            os.environ["TIME_STRING"],
        )
    else:
        timestamp, date_string, time_string = get_timestamp_for_timing_files()

    unique_file_name = "timing_results_" + str(timestamp) + ".h5"
    file_path = os.path.join(timing_results_folder, unique_file_name)

    with h5py.File(file_path, "w") as F:
        F.attrs["name"] = name
        F.attrs["host"] = socket.gethostname().split(".")[0]
        F.attrs["commit"] = get_current_commit_hash()
        F.attrs["branch"] = get_current_branch_name()
        F.attrs["date"] = date_string
        F.attrs["time"] = time_string
        insert_timing_dict_into_h5_object(
            timer_utils.ELAPSED_TIME_DICT, F.create_group("elapsed_time_dict")
        )
        insert_timing_dict_into_h5_object(
            timer_utils.ADVANCED_TIME_DICT, F.create_group("advanced_time_dict")
        )


def read_timing_data_file(file_path: str) -> dict:
    def recursively_convert_h5_group(group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                # Recursively process groups
                result[key] = recursively_convert_h5_group(item)
            elif isinstance(item, h5py.Dataset):
                # Convert datasets to numpy arrays or lists
                result[key] = item[()]  # Read the dataset
            else:
                raise ValueError(f"Unknown item type for key {key}: {type(item)}")
        return result

    with h5py.File(file_path, "r") as h5_file:
        d = recursively_convert_h5_group(h5_file)
        # Get attributes
        d["attributes"] = {}
        for k in h5_file.attrs.keys():
            d["attributes"][k] = h5_file.attrs[k]
        return d


def insert_timing_dict_into_h5_object(d: dict, h5_object: Union[h5py.Group, h5py.File]):
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively handle dicts
            insert_timing_dict_into_h5_object(value, h5_object.create_group(key))
        elif isinstance(value, np.ndarray):
            h5_object.create_dataset(key, data=value)
        else:
            raise ValueError("Data type not supported")


def get_current_commit_hash() -> str:
    # Get the shortened commit hash
    commit_hash = (
        subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.STDOUT,
        )
        .strip()
        .decode("utf-8")
    )
    return commit_hash


def get_current_branch_name() -> str:
    # Get the current branch name
    branch_name = (
        subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.STDOUT,
        )
        .strip()
        .decode("utf-8")
    )

    return branch_name
