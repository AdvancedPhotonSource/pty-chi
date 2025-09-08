.. image:: docs/source/img/logo.png
   :alt: Pty-chi Logo
   :align: center
   :width: 200px


Welcome to the repository of Pty-chi, a PyTorch-based ptychography reconstruction library!

.. image:: https://zenodo.org/badge/858453195.svg
  :target: https://doi.org/10.5281/zenodo.15277806


============
Installation
============

Standard installation
---------------------
The easiest way to install the latest release is through PyPI. 

First, create a new conda environment with Python 3.11:
::

    conda create -n ptychi python=3.11

Then install Pty-Chi using::

    pip install ptychi


Developer installation
----------------------

Use developer installation when you want to modify the code and test the changes,
or when you run into build issues that drive you to install the package from source.
We recommend using Conda/pip or uv for environment and package management.

Option 1: Installation with uv
++++++++++++++++++++++++++++++

Uv is a modern lightweight package manager for Python featuring fast speed and
deterministic builds. When creating a uv virtual environment, the environment
directory and all the packages inatalled in it are kept in the current working
directory -- unlike Conda, where the environments are centrally managed. Therefore,
first ``cd`` into the **root level** of your local clone of Pty-Chi, and then create
a new uv virtual environment with Python 3.11::

    uv venv --python 3.11

Then install Pty-Chi and its dependencies using::

    uv sync

You can now run scripts *inside the project directory* with::

    uv run <script.py>

without activating the environment. Alternatively, you can activate the environment
with::

    source .venv/bin/activate

and then run scripts with::

    python <script.py>

This allows you to run scripts located anywhere in your system.

Option 2: Installation with Conda and pip
+++++++++++++++++++++++++++++++++++++++++

To install the latest code in the `main` branch, clone the repository to your workspace, and create a new conda environment
using::

    conda create -n ptychi python=3.11

Then install the package and its dependencies using::

    pip install -r requirements.txt
    pip install -e .


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

Please refer to the developer's guide for more information on how to contribute
to the project. The developer's guide is hosted on the
`Wiki <https://git.aps.anl.gov/ptycho_software/pty-chi/-/wikis/Developer's-guide/home>`_ page of Pty-Chi's 
APS GitLab repository.
To gain access to the APS GitLab repository, please contact the developers.
