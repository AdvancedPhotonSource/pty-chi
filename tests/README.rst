=======================
How to run test scripts
=======================

Test inputs and golden outputs are versioned with DVC. Their pointers live in
``tests/reference/data.dvc`` and ``tests/reference/gold_data.dvc`` alongside the
code. The binary files are stored on an institutional SSH server.

1. Ask the developers for the DVC remote URL and read access to the data.
2. Install DVC with SSH support::

       uv tool install 'dvc[ssh]==3.67.1'

3. From the Pty-Chi repository root, configure the supplied remote URL and
   fetch the data for the current checkout (replace the example URL)::

       export PTYCHO_CI_DATA_URL='ssh://user@storage.example.org/path/to/test-data'
       dvc remote add --local --force -d testdata "$PTYCHO_CI_DATA_URL"
       dvc pull

   The remote URL is saved in Git-ignored ``.dvc/config.local``. Repeat the
   ``dvc remote add`` command when changing the URL or setting up a new checkout.

4. Tests default to ``tests/reference`` in this checkout. If you previously set
   ``PTYCHO_CI_DATA_DIR`` to the old data repository, unset it to use this default.
   An explicit ``PTYCHO_CI_DATA_DIR`` still overrides the data location and emits
   a warning recommending migration to DVC.
5. Run the tests with the CI environment and tolerances::

       uv run --locked --with torch==2.12.1 --with torchvision pytest tests --high-tol

Run ``dvc pull`` after switching branches or pulling code changes to materialize
the data selected by the new pointers.

Updating reference data
-----------------------

Maintainers need write access to the DVC remote. From the repository root, point
the generators at this checkout, regenerate the intended golden outputs, and
record their new version::

    unset PTYCHO_CI_DATA_DIR
    uv run --locked --with torch==2.12.1 --with torchvision python tests/test_lsqml_e2e.py --generate-gold
    dvc add tests/reference/gold_data
    dvc push
    git add tests/reference/gold_data.dvc

Commit the pointer together with the corresponding code changes, then push the
Git branch. For input-data changes, also run ``dvc add tests/reference/data`` and
stage ``tests/reference/data.dvc`` before committing. Review regenerated outputs
before accepting them as the new baseline.

Optionally run ``dvc install`` in each clone to install Git hooks, including a
pre-push hook that uploads data with ``dvc push``. Updating pointers with
``dvc add`` remains part of the regeneration workflow.

GitHub Actions cache
--------------------

The workflow defaults ``PTYCHO_CI_CACHE_DIR`` to a ``.dvc-cache`` directory next
to the checkout. With the standard runner layout, this is
``_work/pty-chi/.dvc-cache``, alongside ``_work/pty-chi/pty-chi``.

To override this location, set the ``PTYCHO_CI_CACHE_DIR`` repository variable
under Settings > Secrets and variables > Actions > Variables to an absolute
path outside the checkout, writable by the runner account. The workflow
configures the cache before fetching the reference data.

The cache survives checkout cleanup, allowing later runs to restore previously
cached data versions locally and download only missing objects. Working data
still lives in ``tests/reference`` within the checkout.
