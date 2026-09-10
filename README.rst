.. image:: docs/source/img/logo.png
   :alt: Pty-chi Logo
   :align: center
   :width: 200px

.. epigraph::
    Pty-Chi is actively maintained, and we welcome feature requests, bug reports, and contributions from the community.

    If you need a new capability, or have implemented useful extensions in your own branch or derivative codebase, we strongly encourage you to consider contributing those changes back to Pty-Chi through an issue or pull request whenever practical, rather than maintaining a separate repository with overlapping functionality.

    Keeping broadly useful improvements in the shared upstream project helps reduce fragmentation, preserves a clear development history, and makes new capabilities easier for the wider community to discover, maintain, and build upon.

Welcome to the repository of Pty-chi, a PyTorch-based ptychography reconstruction library!

.. image:: https://zenodo.org/badge/858453195.svg
  :target: https://doi.org/10.5281/zenodo.15277806


============
Installation
============

We recommend using `uv <https://docs.astral.sh/uv/>`_ for environment and
package management. uv is a modern lightweight package manager for Python
featuring fast speed and deterministic builds. When creating a uv virtual
environment, the environment directory and all the packages installed in it are
kept in the current working directory.

Standard installation
---------------------
If you are adding Pty-Chi to a uv-managed project, use::

    uv add ptychi

If you are installing Pty-Chi into an existing uv virtual environment that is not
managed as a uv project, use::

    uv pip install ptychi

Pty-Chi provides uv extras to select a PyTorch build:

* ``cpu``: CPU-only PyTorch build.
* ``cuda126``: PyTorch build with CUDA 12.6.
* ``cuda132``: PyTorch build with CUDA 13.2.
* ``rocm72``: PyTorch build with ROCm 7.2 on Linux.

For example::

    uv add "ptychi[cuda132]"

or::

    uv pip install "ptychi[cpu]"

If you are not using uv, install the default build from PyPI with::

    pip install ptychi


Developer installation
----------------------

Use developer installation when you want to modify the code and test the changes,
or when you run into build issues that drive you to install the package from source.

First ``cd`` into the **root level** of your local clone of Pty-Chi, and then
create a new uv virtual environment with Python 3.11::

    uv venv --python 3.11

Then install Pty-Chi and its dependencies using::

    uv sync

You can select a PyTorch build with the same extras used for standard
installation. For example::

    uv sync --extra cpu
    uv sync --extra cuda126
    uv sync --extra cuda132
    uv sync --extra rocm72

You can now run scripts *inside the project directory* with::

    uv run <script.py>

without activating the environment. Alternatively, you can activate the environment
with::

    source .venv/bin/activate

and then run scripts with::

    python <script.py>

This allows you to run scripts located anywhere in your system.


=======================
How to run test scripts 
=======================

1. Contact the developers to be given access to the APS GitLab repository
   that holds test data. **You need to have an account on APS GitLab**.
2. After gaining access, clone the GitLab data repository to your
   hard drive. 
3. Set ``PTYCHO_CI_DATA_DIR`` to the ``ci_data`` directory of the data
   repository: ``export PTYCHO_CI_DATA_DIR="path_to_data_repo/ci_data"``.
4. Run any test scripts in ``tests`` with Python.

======================
To use non-Nvidia GPUs
======================

Pty-Chi works on GPUs from different vendors than NVidia. For example, Intel.
To run Pty-Chi with Intel GPUs, add these lines right after importing `torch`
and `ptychi`::

   torch.set_default_device("xpu")
   ptychi.device.set_torch_accelerator_module(torch.xpu)


======================
Reading documentations
======================

Pty-Chi's documentation is hosted on `Read the Docs <https://pty-chi.readthedocs.io/>`_.

You can also build the docs and view them in your browser locally.
To build the docs, install the dependencies as the first step::

    pip install -r docs/requirements.txt

Then::

   cd docs
   make html

You can then view the docs by opening ``docs/build/html/index.html`` in your browser.


=================
Developer's Guide
=================

Please refer to the developer's guide in the `Wiki page <https://github.com/AdvancedPhotonSource/pty-chi/wiki>`_
for more information on how to contribute to the project.

🛑 **For major changes such as adding a new reconstructor or those involving extensive structural changes to the codebase, 
please first discuss with the maintainers by creating an issue. Consider creating a new package inheriting Pty-Chi's reconstructors
and data structures for experimental features and algorithms. See**
`Developer’s guide: Extending Pty-Chi with external algorithm packages <https://github.com/AdvancedPhotonSource/pty-chi/wiki/Developer%27s-guide:-Extending-Pty%E2%80%90chi-with-external-algorithm-packages>`_.

=================
Citation
=================

If Pty-Chi is useful for your research, please consider citing the following paper::

    @misc{ptychi_2025_arxiv,
        title = {Pty-Chi: A PyTorch-based modern ptychographic data analysis package}, 
        author = {Ming Du and Hanna Ruth and Steven Henke and Yi Jiang and Viktor Nikitin and Ashish Tripathi and Junjing Deng and Jeffrey Klug and Peco Myint and Tao Zhou and Nicholas Schwarz and Mathew Cherukara and Alec Sandy and Stefan Vogt},
        year = {2025},
        eprint = {2510.20929},
        archivePrefix = {arXiv},
        primaryClass = {physics.optics},
        url = {https://arxiv.org/abs/2510.20929}, 
    }
