PtychographyTask
================

``PtychographyTask`` is the main entry point for running a reconstruction.
Instantiate it with the appropriate options, call :meth:`run
<ptychi.api.task.PtychographyTask.run>`, and then inspect the parameters that
live inside the task's reconstructor. The sections below highlight two
workflow helpers that make day-to-day usage easier.


Copying data from another task
------------------------------

When exploring multiple option sets, you can seed a fresh task with the
results of a previous run instead of reloading arrays from disk. Use
:meth:`~ptychi.api.task.PtychographyTask.copy_data_from_task` to copy the
object, probe, probe positions, and/or OPR mode weights from another task
instance:

.. code-block:: python

    warm_start_task = api.PtychographyTask(new_options)
    warm_start_task.copy_data_from_task(reference_task)

Pass ``params_to_copy`` if you only want a subset of parameters. The method
automatically detaches tensors from autograd and writes them directly into the
new reconstructor's parameter group so you can immediately continue training.


Managing accelerator memory
---------------------------

For workflows that juggle several tasks on a single GPU, call
:meth:`~ptychi.api.task.PtychographyTask.set_large_tensor_device` to offload or
reload the heavy buffers:

.. code-block:: python

    task.set_large_tensor_device("cpu")  # Offload object/probe/data to host
    # ... run another task ...
    task.set_large_tensor_device()        # Bring buffers back to the default device

Moving tensors to CPU frees accelerator memory while retaining the rest of the
task state (options, timers, history). Calling the method again with 
``"cuda"`` returns the buffers to GPU (or call it with no arguments to move the
buffers to the current default device) and re-synchronizes the reconstructor's
internal forward-model caches so the next ``run`` call can resume immediately.
