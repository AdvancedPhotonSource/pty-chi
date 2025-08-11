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
