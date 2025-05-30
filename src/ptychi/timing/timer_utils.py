# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import time
from functools import wraps
from typing import Callable, TypeVar, Optional, List, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import defaultdict

from ptychi.device import AcceleratorModuleWrapper


# Type variables to retain function signatures
T = TypeVar("T", bound=Callable)

ENABLE_TIMING = False
"Global flag to enable or disable timing."
ELAPSED_TIME_DICT: dict[str, np.ndarray] = defaultdict(lambda: np.array([]))
"""
A dictionary containing numpy arrays of the measured execution times of 
each timed function.
"""
ADVANCED_TIME_DICT: dict[str, Union[np.ndarray, dict]] = defaultdict(lambda: {})
"""
A nested dictionary, where each level of the dictionary contains
1) a key, value pair ("time": np.ndarray) that contains all measured
execution times for that function and 2) zero or more key-value 
pairs (function_name: dict) where function_name refers to the name
of each functions called in the function currently being times.

Note that only functions with the `timer` decorator will show up
in `ADVANCED_TIME_DICT`.
"""
CURRENT_DICT_REFERENCE = ADVANCED_TIME_DICT  # Initialized to the top level of ADVANCED_TIME_DICT
"""
A reference to the level of `ADVANCED_TIME_DICT` that corresponds to 
the function currently being executed.
"""
TIMING_OVERHEAD_ARRAY = np.array([])
"""
A numpy array containing measurements of how long it takes to execute 
functions in the `timer` decorator.
"""
default_label_font_size = 8


def toggle_timer(enable: bool):
    """
    Toggle the global ENABLE_TIMING flag.

    Parameters
    ----------
    enable : bool
        If True, enable timing. If False, disable timing.
    """
    global ENABLE_TIMING
    ENABLE_TIMING = enable


def timer(enabled: bool = True, override_with_name: Optional[str] = None):
    """
    Decorator to time a function's execution time and the execution time of the timed code
    within that function. This function is enabled or disabled depending on the state of the
    global ENABLE_TIMING flag.

    The results of the timer function will be recorded in `ELAPSED_TIME_DICT` and
    `ADVANCED_TIME_DICT`.

    Parameters
    ----------
    enabled : bool, optional
        Whether timing is enabled for the decorated function. Default is True.
    override_with_name : str, optional
        Custom name to use for the function in the timing dictionary. If not
        specified, the function name is automatically generated.

    Returns
    -------
    Callable
        The wrapped function.
    """

    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enabled and globals().get("ENABLE_TIMING", False):
                # Measure the overhead from running the timer function
                measure_overhead_start_1 = time.time()
                if override_with_name is None:
                    function_name = func.__qualname__
                else:
                    function_name = override_with_name
                saved_dict_reference = update_current_dict_reference(function_name)
                overhead_time_1 = time.time() - measure_overhead_start_1

                # Measure function execution time
                AcceleratorModuleWrapper.get_module().synchronize()
                start_time = time.time()
                result = func(*args, **kwargs)
                AcceleratorModuleWrapper.get_module().synchronize()
                elapsed_time = time.time() - start_time

                # Measure the overhead from running the timer function
                measure_overhead_start_2 = time.time()
                update_elapsed_time_dict(function_name, elapsed_time)
                update_advanced_time_dict(elapsed_time)
                # Traverse back up the advanced timing dicts
                revert_current_dict_reference(saved_dict_reference)
                global TIMING_OVERHEAD_ARRAY
                overhead_time_2 = time.time() - measure_overhead_start_2
                TIMING_OVERHEAD_ARRAY = np.append(
                    TIMING_OVERHEAD_ARRAY,
                    overhead_time_1 + overhead_time_2,
                )
            else:
                # If timing is disabled, just call the function
                result = func(*args, **kwargs)
            return result

        # Ensure the wrapper function has the same type as the original
        return wrapper  # type: ignore

    return decorator


class InlineTimer:
    """
    A timer class for inline timing of code blocks.

    Parameters
    ----------
    name : str
        The name associated with the timer that will be recorded
        in the timing dictionaries.
    enabled : bool, optional
        Whether the timer is enabled, by default True.
    """

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.overhead_time = 0

    def start(self):
        """
        Starts the timer if timing is enabled.
        """
        if self.enabled and globals().get("ENABLE_TIMING", False):
            measure_overhead_start = time.time()
            saved_dict_reference = update_current_dict_reference(self.name)
            self.saved_dict_reference = saved_dict_reference
            self.overhead_time = time.time() - measure_overhead_start
            AcceleratorModuleWrapper.get_module().synchronize()
            self.start_time = time.time()

    def end(self):
        """
        Stops the timer and records the elapsed time if timing is enabled.
        """
        if self.enabled and globals().get("ENABLE_TIMING", False):
            AcceleratorModuleWrapper.get_module().synchronize()
            elapsed_time = time.time() - self.start_time
            measure_overhead_start = time.time()
            update_elapsed_time_dict(self.name, elapsed_time)
            update_advanced_time_dict(elapsed_time)
            revert_current_dict_reference(self.saved_dict_reference)
            global TIMING_OVERHEAD_ARRAY
            self.overhead_time += time.time() - measure_overhead_start
            TIMING_OVERHEAD_ARRAY = np.append(TIMING_OVERHEAD_ARRAY, self.overhead_time)


def update_elapsed_time_dict(function_name: str, elapsed_time: float):
    """
    Updates the global elapsed time dictionary with the elapsed time for a function.

    Parameters
    ----------
    function_name : str
        The name of the function being timed.
    elapsed_time : float
        The elapsed time for the function execution.
    """
    ELAPSED_TIME_DICT[function_name] = np.append(ELAPSED_TIME_DICT[function_name], elapsed_time)


def update_current_dict_reference(function_name: str) -> dict:
    """
    Updates the current reference in the advanced timing dictionary to a nested level.

    Parameters
    ----------
    function_name : str
        The name of the function being timed.

    Returns
    -------
    dict
        The previous dictionary reference.
    """
    global CURRENT_DICT_REFERENCE
    # Save the parent to traverse back to later
    saved_dict_reference = CURRENT_DICT_REFERENCE
    # Create new dict if necessary
    if function_name not in CURRENT_DICT_REFERENCE.keys():
        CURRENT_DICT_REFERENCE[function_name] = defaultdict(lambda: {})
        CURRENT_DICT_REFERENCE[function_name]["time"] = np.array([])
    # Update the pointer to the current dict
    CURRENT_DICT_REFERENCE = CURRENT_DICT_REFERENCE[function_name]
    return saved_dict_reference


def update_advanced_time_dict(elapsed_time: float):
    """
    Updates the advanced timing dictionary with the elapsed time.

    Parameters
    ----------
    elapsed_time : float
        The elapsed time for the function execution.
    """
    global CURRENT_DICT_REFERENCE
    CURRENT_DICT_REFERENCE["time"] = np.append(CURRENT_DICT_REFERENCE["time"], elapsed_time)


def revert_current_dict_reference(saved_dict_reference: dict):
    """
    Reverts the current dictionary reference in the advanced timing dictionary.

    Parameters
    ----------
    saved_dict_reference : dict
        The saved dictionary reference to revert to.
    """
    global CURRENT_DICT_REFERENCE
    CURRENT_DICT_REFERENCE = saved_dict_reference


def clear_timer_globals():
    """
    Clears the global timing dictionaries and resets the state.
    """
    global ELAPSED_TIME_DICT
    global ADVANCED_TIME_DICT
    global CURRENT_DICT_REFERENCE
    global TIMING_OVERHEAD_ARRAY
    ELAPSED_TIME_DICT = defaultdict(lambda: np.array([]))
    ADVANCED_TIME_DICT = defaultdict(lambda: {})
    CURRENT_DICT_REFERENCE = ADVANCED_TIME_DICT
    TIMING_OVERHEAD_ARRAY = np.array([])


def plot_elapsed_time_bar_plot(
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    figsize: Optional[tuple] = None,
    only_include_leafs: bool = False,
    advanced_time_dict: Optional[dict] = None,
    elapsed_time_dict: Optional[dict] = None,
    sort: bool = False,
):
    """
    Plots a simple bar chart of elapsed times for timed functions.

    Parameters
    ----------
    include : list of str, optional
        Specific functions to include in the plot, by default None.
    exclude : list of str, optional
        Specific functions to exclude from the plot, by default None.
    top_n : int, optional
        The top N functions with the largest elapsed times to include, by default None.
    only_include_leafs : bool, optional
        If True, includes only leaf functions in the plot, by default False.
    figsize : tuple, optional
        The figure size for the plot, by default None.
    label_fontsize: int, optional
        The font size for bar labels.
    advanced_time_dict : dict, optional
        By default, this will be the global `ADVANCED_TIME_DICT`. Override this
        to plot some other dict. The `ADVANCED_TIME_DICT` is only used when
        `only_include_leafs` is True.
    elapsed_time_dict : dict, optional
        By default, this will be the global `ELAPSED_TIME_DICT`. Override this
        to plot some other dict.
    """
    # If elapsed_time_dict was not passed in, use the
    # value from the global
    if elapsed_time_dict is None:
        global ELAPSED_TIME_DICT
        elapsed_time_dict = ELAPSED_TIME_DICT

    if only_include_leafs:
        if advanced_time_dict is None:
            global ADVANCED_TIME_DICT
            advanced_time_dict = ADVANCED_TIME_DICT
        elapsed_time_dict = select_leaf_functions(advanced_time_dict)

    elapsed_time_dict = return_dict_subset_copy(elapsed_time_dict, include, exclude)

    # Remove the prefix from keys and calculate sums
    sums = [np.sum(elapsed_time_dict[key]) for key in elapsed_time_dict.keys()]

    generate_time_barplot(
        times=sums,
        labels=list(elapsed_time_dict.keys()),
        colors="skyblue",
        title="Total elapsed time for each function",
        figsize=figsize,
        sort=sort,
        top_n=top_n,
    )


def plot_elapsed_time_bar_plot_advanced(
    function_name: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    max_levels: int = None,
    use_long_bar_labels: bool = False,
    figsize: Optional[tuple] = None,
    label_fontsize: Optional[int] = None,
    advanced_time_dict: Optional[dict] = None,
    exclude_below_time_fraction: float = 1e-2,
    sort: bool = False,
):
    """
    Plots a detailed breakdown of execution times for a specific function.

    Parameters
    ----------
    function_name : str
        The name of the function to analyze.
    max_levels : int, optional
        The maximum levels of nested function calls to display, by default None.
    use_long_bar_labels : bool, optional
        Whether to use long labels for the bars, by default False.
    figsize : tuple, optional
        The figure size for the plot, by default None.
    label_fontsize: int, optional
        The font size for bar labels.
    advanced_time_dict : dict, optional
        By default, this will be the global `ADVANCED_TIME_DICT`. Override this
        to plot some other dict.
    exclude_below_time_fraction : float, optional
        Excludes functions contributing less than this fraction of total time, by default 1e-2.

    Returns
    -------
    dict
        A dictionary of execution times for each function in the call stack.
    """

    if advanced_time_dict is None:
        global ADVANCED_TIME_DICT
        advanced_time_dict = ADVANCED_TIME_DICT

    def find_key_in_nested_dict(nested_dict: dict, target_key):
        for key, value in nested_dict.items():
            if key == target_key:
                return value
            if isinstance(value, dict):
                result = find_key_in_nested_dict(value, target_key)
                if result is not None:
                    return result

    # Find the dictionary containing the function_name key
    result_dict = find_key_in_nested_dict(advanced_time_dict, function_name)
    if result_dict is None:
        raise ValueError(f"Key '{function_name}' not found in the nested dictionary.")

    result_dict = return_dict_subset_copy(result_dict, include, exclude)

    # Prepare data for plotting
    short_labels = []
    full_call_stack_labels = []
    times = []
    colors = []

    total_execution_time = result_dict["time"].sum()

    def collect_data(d, prefix="", function_name="", level=0):
        if max_levels is not None and level > max_levels:
            return
        symbol = " \u2192 "
        for sub_key, sub_value in d.items():
            if isinstance(sub_value, dict):
                collect_data(sub_value, prefix + sub_key + symbol, sub_key, level + 1)
            elif sub_key == "time":
                if sub_value.sum() / total_execution_time < exclude_below_time_fraction:
                    return
                if prefix == "":
                    full_call_stack_labels.append("Total")
                    short_labels.append("Total")
                    colors.append("dodgerblue")
                else:
                    full_call_stack_labels.append(prefix.rstrip("/")[: -len(symbol)])
                    short_labels.append(function_name)
                    colors.append("skyblue")
                times.append(sub_value.sum())

    # Get total execution times for each of the called functions
    collect_data({k: v for k, v in result_dict.items() if k != "time"})

    # Add the total execution time of the function at the front of the list
    times.insert(0, total_execution_time)
    full_call_stack_labels.insert(0, "Total")
    short_labels.insert(0, "Total")
    colors.insert(0, "dodgerblue")

    if use_long_bar_labels:
        bar_labels = full_call_stack_labels
    else:
        bar_labels = short_labels

    if top_n is not None:
        top_n = top_n + 1

    generate_time_barplot(
        times=times,
        labels=bar_labels,
        colors=colors,
        title=f"Execution time breakdown for\n{function_name}",
        figsize=figsize,
        label_fontsize=label_fontsize,
        sort=sort,
        top_n=top_n,
    )

    print(text_bf(f"Execution summary of {function_name}\n"))
    print(text_bf(f"Total execution time: {total_execution_time:.3g} s\n"))
    print(text_bf("Execution times:"))
    if sort:
        idx = np.argsort(times[1:]) + 1
    else:
        idx = range(1, len(short_labels))
    for i in idx:
        print(f"{i}. {short_labels[i]}: {times[i]:.2g} s")
    print()
    print(text_bf("Function call stack info:"))
    for i in range(1, len(full_call_stack_labels)):
        print(f"{i}. {full_call_stack_labels[i]}")

    selected_elapsed_times = dict(zip(short_labels, times))
    return selected_elapsed_times


def generate_time_barplot(
    times: dict,
    labels: list,
    colors: Optional[Union[list, str]],
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
    label_fontsize: Optional[int] = None,
    sort: bool = False,
    top_n: Optional[int] = None,
):
    """
    Generates a horizontal bar plot for elapsed times.

    Parameters
    ----------
    times : dict
        Elapsed times for the functions.
    labels : list
        Labels for the bars.
    colors : list or str, optional
        Colors for the bars, by default None.
    title : str, optional
        Title for the plot, by default None.
    figsize : tuple, optional
        Figure size for the plot, by default None.
    label_fontsize : int, optional
        Font size for the bar labels.
    """
    if label_fontsize is None:
        label_fontsize = default_label_font_size
    if sort:
        sort_idx = np.argsort(times)[::-1]
        times = [times[i] for i in sort_idx]
        labels = [labels[i] for i in sort_idx]
        if isinstance(colors, list):
            colors = [colors[i] for i in sort_idx]
    if top_n is not None:
        sort_idx = np.argsort(times)[::-1][:top_n]
        keep_idx = np.sort(sort_idx)
        times = [times[i] for i in keep_idx]
        labels = [labels[i] for i in keep_idx]
        if isinstance(colors, list):
            colors = [colors[i] for i in keep_idx]

    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(times)), times, color=colors)
    plt.gca().invert_yaxis()  # Invert the y-axis to have the largest bar on top
    plt.tick_params(left=False, labelleft=False)

    # Add labels at the start of each bar
    for i, (bar, label) in enumerate(zip(bars, labels)):
        plt.text(
            0,  # Start at the left edge
            bar.get_y() + bar.get_height() / 2,  # Vertically centered
            " " + label,
            ha="left",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
        )

    # Set x-axis and title
    plt.grid(linestyle=":")
    plt.gca().set_axisbelow(True)
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Function")
    plt.title(title)
    # plt.tight_layout()
    plt.show()


def plot_elapsed_time_vs_call_number(
    elapsed_time_dict: Dict[str, List[float]],
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    linestyle: str = "-",
    top_n: Optional[int] = None,  # plot top N largest sums
    exclude_below_time_fraction: float = 1e-2,
):
    elapsed_time_dict = return_dict_subset_copy(elapsed_time_dict, include, exclude)
    if top_n is not None:
        elapsed_time_dict = return_top_n_entries(elapsed_time_dict, top_n)

    total_execution_time = np.max([v.sum() for v in elapsed_time_dict.values()])

    for k, v in elapsed_time_dict.items():
        if hasattr(v, "__len__") and len(v) > 2:  # temp fix
            if v.sum() / total_execution_time > exclude_below_time_fraction:
                plt.plot(v, linestyle, label=k)

    # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(linestyle=":")
    plt.gca().set_axisbelow(True)
    plt.ylabel("Elapsed Time (s)")
    plt.xlabel("Call number")
    plt.title("Elapsed time vs call number")
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.show()

    # Separate figure for the legend
    fig_legend = plt.figure()#figsize=(4, 2))  # Adjust size as needed
    fig_legend.legend(handles, labels, loc='center', frameon=False)
    plt.axis('off')  # Turn off the axes
    plt.show()


def return_dict_subset_copy(
    elapsed_time_dict: dict,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> dict:
    elapsed_time_dict = copy.deepcopy(elapsed_time_dict)
    if exclude is not None:
        [elapsed_time_dict.pop(k) for k in exclude]
    if include is not None:
        elapsed_time_dict = {
            k: elapsed_time_dict[k] for k in elapsed_time_dict.keys() if k in include
        }
    return elapsed_time_dict


def return_top_n_entries(elapsed_time_dict: dict, top_n: int) -> dict:
    elapsed_time_dict = copy.deepcopy(elapsed_time_dict)
    # Sort dictionary items based on the sum of their values and get top N
    sorted_items = sorted(
        elapsed_time_dict.items(),
        # key=lambda x: sum(x[1]) if hasattr(x[1], "__iter__") else float("-inf"),
        key=lambda x: sum(x[1]) if hasattr(x[1], "__iter__") else x[1],
        reverse=True,
    )[:top_n]
    keys = list(elapsed_time_dict.keys())
    keep_keys = list(dict(sorted_items).keys())
    for k in keys:
        if k not in keep_keys:
            elapsed_time_dict.pop(k)

    return elapsed_time_dict


def select_leaf_functions(advanced_time_dict: dict) -> dict[str, np.ndarray]:
    time_key = "time"

    def recursive_leaf_finder(
        nested_dict: dict, last_key: str = "top level", leaf_functions: dict = {}
    ) -> dict[str, np.ndarray]:
        # Check if function is a leaf by seeing if any of the
        # values in the dict are dicts
        is_a_leaf = np.any([isinstance(v, dict) for v in nested_dict.values()])
        if is_a_leaf:
            for k, v in nested_dict.items():
                if k != time_key:
                    recursive_leaf_finder(v, k, leaf_functions)
        else:
            if last_key in leaf_functions.keys():
                # Note: if a function is called in more than one
                # function then the array are no longer timer
                # ordered.
                leaf_functions[last_key] = np.append(
                    leaf_functions[last_key],
                    nested_dict[time_key],
                )
            else:
                leaf_functions[last_key] = nested_dict[time_key]
        return leaf_functions

    return recursive_leaf_finder(advanced_time_dict)


def text_bf(string: str) -> str:
    return f"\033[1m{string}\033[0m"
