import pytest


def pytest_addoption(parser):
    parser.addoption("--high-tol", action="store_true", help='Use high tolerance for certain tests.')
    parser.addoption("--all", action="store_true", help='Run all tests.')
    
    
def pytest_configure(config):
    config.addinivalue_line("markers", "local: mark test as local-only")

    
def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        # --all given in cli: do not skip slow tests
        return
    skip_local = pytest.mark.skip(reason="need --all option to run")
    for item in items:
        if "local" in item.keywords:
            item.add_marker(skip_local)