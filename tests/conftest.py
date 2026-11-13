import pytest
import os
from ptychi.timing.io import get_timestamp_for_timing_files


def pytest_addoption(parser):
    parser.addoption(
        "--high-tol", action="store_true", help="Use high tolerance for certain tests."
    )
    parser.addoption("--all", action="store_true", help="Run all tests.")
    parser.addoption("--local-only", action="store_true", help="Just run local-only test cases.")
    parser.addoption(
        "--action",
        default="all",
        help="Select the action to run. Can be: "
             "\"generate\" - just generate outputs and store them; "
             "\"compare\" - just run comparison with reference data; "
             "\"all\" - run everything.",
    )
    parser.addoption(
        "--save-timing", action="store_true", help="Enable function timing and save results."
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "local: mark test as local-only")
    config.addinivalue_line("markers", "e2e: complete reconstruction regression")
    # Save timestamps for timing file metadata
    timestamp, date_string, time_string = get_timestamp_for_timing_files()
    os.environ["TIMESTAMP"] = timestamp
    os.environ["DATE_STRING"] = date_string
    os.environ["TIME_STRING"] = time_string


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.name.endswith("_e2e.py"):
            item.add_marker(pytest.mark.e2e)
    if config.getoption("--all"):
        return
    skip_local = pytest.mark.skip(reason="need --all option to run")
    skip_non_local = pytest.mark.skip(reason="skipped due to --local-only flag")
    for item in items:
        if "local" in item.keywords and not config.getoption("--local-only"):
            item.add_marker(skip_local)
        elif "local" not in item.keywords and config.getoption("--local-only"):
            item.add_marker(skip_non_local)


@pytest.fixture
def numerical_cpu():
    """Isolate global tensor defaults changed by task construction."""
    import random

    import numpy as np
    import torch

    import ptychi.maths as pmath
    from ptychi.utils import get_default_complex_dtype, set_default_complex_dtype

    device = torch.get_default_device()
    dtype = torch.get_default_dtype()
    complex_dtype = get_default_complex_dtype()

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    nondeterministic = pmath.get_allow_nondeterministic_algorithms()
    fft_precision = pmath.get_use_double_precision_for_fft()
    deterministic = torch.are_deterministic_algorithms_enabled()
    rng_state = torch.random.get_rng_state()
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    set_default_complex_dtype(torch.complex128)
    try:
        yield
    finally:
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        set_default_complex_dtype(complex_dtype)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        pmath.set_allow_nondeterministic_algorithms(nondeterministic)
        pmath.set_use_double_precision_for_fft(fft_precision)
        torch.use_deterministic_algorithms(deterministic)
        torch.random.set_rng_state(rng_state)
