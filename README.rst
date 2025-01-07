.. image:: docs/source/img/logo.png
   :alt: Pty-chi Logo
   :align: center
   :width: 200px


Welcome to the repository of Pty-chi, a PyTorch-based ptychography reconstruction library!


============
Installation
============

Clone the repository to your workspace, and create a new conda environment
using::

    conda create -n ptychi -c conda-forge -c nvidia --file requirements-dev.txt

Then install the package using::

    pip install -e .


=======================
How to run test scripts 
=======================

1. Contact the developers to be given access to the APS GitLab repository
   that holds test data. 
2. After gaining access, clone the GitLab data repository to your
   hard drive. 
3. Set ``PTYCHO_CI_DATA_DIR`` to the ``ci_data`` directory of the data
   repository: ``export PTYCHO_CI_DATA_DIR="path_to_data_repo/ci_data"``.
4. Run any test scripts in ``tests`` with Python.


=====================
How to build the docs
=====================

To build the docs, install the dependencies as the first step::

   pip install -e .[docs]

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
