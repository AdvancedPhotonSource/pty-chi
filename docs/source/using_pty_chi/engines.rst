Engines
=======

Pty-Chi supports multiple algorithms, or engines, for ptychography reconstruction. 
The table below compares their merits and limitations. 

.. list-table::
   :stub-columns: 1
   :widths: 40 40 40 40 40

   * - Engine
     - **LSQML**
     - **PIE**
     - **Difference map**
     - **Autodiff**
   * - Minibatching allowed
     - Yes
     - Yes  
     - No
     - Yes
   * - GPU support
     - Multiple (multi-process)
     - Single
     - Single
     - Multiple
   * - Memory usage
     - Moderate
     - Moderate
     - High
     - Moderate
   * - Mixed-state probe support
     - Yes
     - Yes
     - Yes
     - Yes
   * - OPR support
     - Yes
     - Yes
     - No
     - Yes
   * - Multislice support
     - Yes
     - Yes
     - No
     - Yes

