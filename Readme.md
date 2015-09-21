PRME
----

Python/Cython implementation of the: "Personalized Ranking Metric Embedding for
Next New POI Recommendation" paper.

Notes
-----

This the PRME model from the paper, not the PRME-G. Should be simple enough
to adapt the code for PRME-G.

Dependencies for library
------------------------
   * Cython
   * Numpy
   * Pandas

How to install
--------------

Clone the repo

::

$ git clone https://github.com/flaviovdf/prme.git

Make sure you have cython and numpy. If not run as root (or use your distros package manager)

::

$ pip install numpy

::

$ pip install Cython

Install

::

$ python setup.py install

Run the main script or the cross_val script:

$ python main.py data_file num_latent_factors model.h5

This will read the data_file, decompose with num_latent_factors and save
the model under the filename model.h5

The model is a pandas HDFStore. Just read-it with:

::

>> import pandas as pd

>> pd.HDFStore('model.h5')

The keys of this store have the output matrices described in the paper.

Input Format
------------

The input file should have this format:

dt <tab> user <item> from <item> to

That is, a tab separated file where the first column is the amount of time the user 
spent on `from` before going to `to`. The second column is the user id, the third 
is the `from` object, whereas the fourth is the destination `to` object. I used 
this input on other repositores, thus the main  reason I kept it here. 

References
----------
.. [1] Shanshan Feng, Xutao Li, Yifeng Zeng, Gao Cong, Yeow Meng Chee, Quan Yuan
   "Personalized Ranking Metric Embedding for Next New POI Recommendation" - IJCAI 2015 
