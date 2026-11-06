PtychographyTask
================

``PtychographyTask`` is the main entry point for running a reconstruction.
Instantiate it with the appropriate options, call :meth:`run
<ptychi.api.task.PtychographyTask.run>`, and then inspect the parameters that
live inside the task's reconstructor. The sections below highlight two
workflow helpers that make day-to-day usage easier.


Passing Reconstruction Data
---------------------------

Options contain reconstruction settings. Large arrays are passed directly to
``PtychographyTask``:

.. code-block:: python

    task = api.PtychographyTask(
        options,
        diffraction_data=diffraction_data,
        object_data=object_guess,
        probe_data=probe_guess,
        probe_position_x_px=positions_px[:, 1],
        probe_position_y_px=positions_px[:, 0],
        opr_mode_weights_data=opr_mode_weights,
        valid_pixel_mask=valid_pixel_mask,
    )

``opr_mode_weights_data`` and ``valid_pixel_mask`` are optional. If OPR weights
are omitted for a single-OPR-mode probe, the task initializes all weights to 1.
If ``valid_pixel_mask`` is omitted or set to ``None``, all detector pixels are
treated as valid.

Pass ``detector_response_map=(multiplier, offset)`` to model a shared affine
detector response. Both components must be NumPy arrays or tensors with the
same ``(H, W)`` shape and pixel ordering as the input diffraction patterns.
With ``data_options.fft_shift=True`` (the default), supply the maps in original
detector coordinates, aligned with the centered far-field patterns, without
manually FFT-shifting them. The task applies ``torch.fft.fftshift`` to both maps
and patterns along the last two axes. With ``fft_shift=False``, the maps are
used as supplied: align them with the already-shifted patterns when the
far-field DC pixel is at the top left, or with the unshifted detector images
for near-field data. After detector cropping and blur, the forward model applies
``intensity * multiplier + offset``. The multiplier is dimensionless and the
offset uses measured-intensity units. The default, ``None``, leaves the
predicted intensity unchanged.

For compatibility, the previous option-held data fields still work during a
transition period and emit ``DeprecationWarning`` when used:
``data_options.data``, ``data_options.valid_pixel_mask``,
``object_options.initial_guess``, ``probe_options.initial_guess``,
``probe_position_options.position_x_px``,
``probe_position_options.position_y_px``, and
``opr_mode_weight_options.initial_weights``.


Copying data from another task
------------------------------

When exploring multiple option sets, you can seed a fresh task with the
results of a previous run instead of reloading arrays from disk. Use
:meth:`~ptychi.api.task.PtychographyTask.copy_data_from_task` to copy the
object, probe, probe positions, and/or OPR mode weights from another task
instance:

.. code-block:: python

    warm_start_task = api.PtychographyTask(
        new_options,
        diffraction_data=diffraction_data,
        object_data=object_guess,
        probe_data=probe_guess,
        probe_position_x_px=positions_px[:, 1],
        probe_position_y_px=positions_px[:, 0],
    )
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
