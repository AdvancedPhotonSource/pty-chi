Data structures
===============

Pty-Chi is built on PyTorch, so on the most basic level, all data structures are
PyTorch tensors. When a hardware accelerator device (like a GPU) is used, these tensors
might reside on the device. To plot them or save them to disk, transfer them to the
CPU using ``tensor.cpu()``. You may also convert them to NumPy arrays using
``tensor.cpu().numpy()``.

For user friendliness, Pty-Chi's high-level interface accepts both NumPy arrays and
PyTorch tensors for initial guesses and data for the object, probe, probe positions,
and initial OPR mode weights. If PyTorch tensors are used, they do not need to be
transferred to accelerator devices as Pty-Chi does this internally.


Tensor shapes
-------------

The table below lists the expected shapes of the tensors for different reconstruction
parameters.

.. list-table::
   :header-rows: 1
   :widths: 30 70 70

   * - Parameter
     - Shape
     - Remarks
   * - Object
     - ``(n_slices, height, width)``
     -
   * - Probe
     - ``(n_opr_modes, n_incoherent_modes, height, width)``
     -
   * - Probe positions
     - ``(n_positions, 2)``
     - Probe positions should follow row-major order, i.e., y-coordinates come first.
   * - Diffraction data
     - ``(n_positions, height, width)``
     -
   * - OPR mode weights
     - ``(n_positions, n_opr_modes)``
     -


Fetching data
-------------

The high-level Python interface of Pty-Chi manages reconstruction jobs using the
``PtychographyTask`` class. When the reconstruction job finishes or is paused,
you may get the data using :meth:`~ptychi.api.task.PtychographyTask.get_data_to_cpu`.
For example::

    # Get the object
    object = task.get_data_to_cpu('object', as_numpy=True)


Alternatively, you can also get reconstruction parameters from the task object
directly without data structure conversion. Reconstruction parameters, such as 
the object and the probe, are stored in ``task.parameter_group``. 
By calling ``task.parameter_group.object``, you get a `ReconstructionParameter`
object for the object function. You can then get the data from the parameter
object using ``.data``.
