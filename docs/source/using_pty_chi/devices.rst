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
