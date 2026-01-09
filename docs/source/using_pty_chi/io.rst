Input and output
================

Numerical data
--------------

Pty-Chi expects NumPy arrays or PyTorch tensors for large numerical data
such as diffraction patterns, initial guesses of object, probe, probe positions,
and OPR mode weights. The structures of these tensors are described in
:doc:`data_structures`. Pty-Chi does not enforce any file format for thsse 
input data stored on hard drive. The way of loading data to memory is up 
to the user as long as the data passed to Pty-Chi complies with the required 
tensor shapes. However, we do recommend using Ptychodus to prepare the input
data so as to maintain a standardized data file format that facilitates data
sharing and reproducibility.


Settings
--------

To export settings in an `Options` object, one can use either the `get_dict`
method of the `Options` object or the `get_options_as_dict` method of the
`PtychographyTask` object to obtain a dictionary of the settings, then save
it to a JSON file.


.. code-block:: python

    import json

    options = api.LSQMLOptions()

    options.data_options.data = data
    # ...

    task = api.PtychographyTask(options)

    # Option 1: get settings dictionary from the Options object
    options_dict = options.get_dict()

    # Option 2: get settings dictionary from the PtychographyTask object
    options_dict = task.get_options_as_dict()

    # Save to JSON file
    with open("settings.json", "w") as f:
        json.dump(options_dict, f)


Settings can also be loaded from JSON files.


.. code-block:: python

    with open("settings.json", "r") as f:
        options_dict = json.load(f)

    options = api.LSQMLOptions()
    options.load_from_dict(options_dict)

Note that the dictionaries for import/export should only contain the settings.
The following large arrays are not included in the dictionaries exported, 
and will be disregarded when loading the settings if they are present.

- ``DataOptions.data``
- ``ObjectOptions.initial_guess``
- ``ObjectOptions.valid_pixel_mask``
- ``ProbeOptions.initial_guess``
- ``ProbePositionOptions.position_x/y_px``
- ``OPRModeWeightsOptions.initial_weights``

These data should be exported or loaded separately.


Outputs
-------

Once a reconstruction finishes, a :class:`~ptychi.api.task.PtychographyTask`
already holds every optimizable tensor you might want to examine or save.
Use the helper accessors below instead of reaching into internal buffers so
that lazy object/probe generators can finalize their states before you read
them.

Fetching reconstructed tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``task.get_data(name)`` returns a detached tensor on the current device for
any of ``"object"``, ``"probe"``, ``"probe_positions"``, or
``"opr_mode_weights"``. To immediately move the data to host memory, call
``task.get_data_to_cpu(name, as_numpy=True)``. The shortcut methods
``task.get_probe_positions_x`` and ``task.get_probe_positions_y`` provide the
individual coordinate arrays if you only need the scan map. The snippet below
copies the complex object estimate to NumPy and saves it:

.. code-block:: python

    obj = task.get_data_to_cpu("object", as_numpy=True)
    np.save("object_final.npy", obj)

Extracting the ROI content
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pty-Chi tracks which portion of the simulation grid the scan illuminated so
you can focus on that cropped region. Grab the current object parameter from
the reconstructor, fetch its bounding box, and then request the ROI tensor:

.. code-block:: python

    obj_param = task.reconstructor.parameter_group.object
    bbox = obj_param.get_probe_position_frame_roi_bounding_box()
    obj_roi = obj_param.get_roi(bbox)

``obj_roi`` now contains only the slices that lie inside the illuminated
bounding box (with bounds reported in probe-position coordinates). You can
pass that tensor to visualization utilities or convert it to NumPy in the
same way as other fetched data.
