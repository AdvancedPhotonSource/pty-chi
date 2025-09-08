Devices
=======

Pty-Chi supports GPU acceleration through PyTorch's native CUDA support. At this moment,
multi-GPU support is only available for the ``AutodiffPtychography`` engine. Other engines
support at most 1 GPU. 

On a computer with multiple GPUs, you can set the device to use by setting the ``CUDA_VISIBLE_DEVICES``
environment variable. For example, to use the first GPU, you can run::

    export CUDA_VISIBLE_DEVICES=0


To disable GPU acceleration, set the variable to an empty string.

Note that it is always recommended to set the variable in terminal before running the code. 
If you have to set the variable in the Python code, make sure to set it before importing PyTorch
using ``os.environ["CUDA_VISIBLE_DEVICES"] = "<GPU index>"``. Setting the variable in Python
will not take effect if it is done after PyTorch is imported.

Non-Nvidia GPUs
---------------

Pty-Chi works on GPUs from different vendors than NVidia. For example, Intel.
To run Pty-Chi with Intel GPUs, add these lines right after importing `torch`
and `ptychi`::

   torch.set_default_device("xpu")
   ptychi.device.set_torch_accelerator_module(torch.xpu)

Multi-GPU and multi-processing
------------------------------

Some reconstruction engines support multi-GPU and/or multi-processing. The biggest benefit
of using multi-GPU or multi-processing is to split the computation of update vectors across
different devices, reducing the VRAM usage on each device. Note that multi-processing does
not always make the computation faster unless the data size is very large because it incurs
communication overhead.

Multi-GPU engines
+++++++++++++++++

The automatic differentiation (Autodiff) engine supports multi-GPU through PyTorch's
``DataParallel`` wrapper. The reconstructor uses all available GPUs by default without
additional settings. To limit it to a single GPU, set ``CUDA_VISIBLE_DEVICES`` before
launching the reconstruction job::

    export CUDA_VISIBLE_DEVICES=0

Multi-processing engines
++++++++++++++++++++++++

Multi-GPU support is enabled for some analytical engine(s) (currently only LSQML)
through the multi-processing feature of PyTorch in ``torch.distributed``. To 
enable multi-processing, you must launch the reconstruction script using ``torchrun``::

    torchrun --nnodes=1 --nproc_per_node=2 reconstruction_script.py

The ``--nnodes`` and ``--nproc_per_node`` arguments specify the number of nodes and 
the number of processes per node, respectively. For single-node machines, keep it to 1.
When a job is launched in this way, Pty-Chi will sign a rank to the GPU indexed
``rank % n_gpus`` where ``n_gpus`` is the number of GPUs available, so as to max
out the number of GPUs while minimizing the number of ranks on each GPU. It is
not recommended, and in some cases not allowed to use launch more processes than
the number of GPUs.

``torchrun`` spawns all processes at the beginning, so the reconstruction script
will also be executed in all processes. If you have post-analysis or data saving
routines in that script, make sure they don't produce unexpected results when executed
in multiple processes. It is generally advised to execute such routines only on rank 0::

    import torch.distributed as dist

    # Set up and run task

    if dist.get_rank() == 0:
        # Do post-analysis or data saving

``dist.get_rank()`` is only callable after the task object is instantiated
where it initializes the process group.
