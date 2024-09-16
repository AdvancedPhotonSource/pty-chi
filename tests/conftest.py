local_only_tests = [
    'test_2d_ptycho_autodiff_lynx215.py'
]


def pytest_addoption(parser):
    parser.addoption("--high-tol", action="store_true", help='Use high tolerance for certain tests.')
    parser.addoption("--all", action="store_true", help='Run all tests.')
    
    
def pytest_collection_modifyitems(config, items):
    run_all = config.getoption("--all")

    # If filenames are given explicitly, always run them even if --all is not given
    for arg in config.args:
        if arg.endswith('.py'):
            run_all = True
            break

    if not run_all:
        for item in items:
            if item.fspath.basename in local_only_tests:
                items.remove(item)
    print('collected items:')
    for item in items:
        print('  ' + str(item))
    