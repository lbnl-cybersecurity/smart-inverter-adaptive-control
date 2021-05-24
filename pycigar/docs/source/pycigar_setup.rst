..    include:: <isonum.txt>
.. contents:: Table of contents

Local Installation of PyCIGAR
==================

**It is highly recommended that users install**
`Anaconda <https://www.anaconda.com/download>`_ **or**
`Miniconda <https://conda.io/miniconda.html>`_
**for Python and the setup instructions will assume that you are
doing so.**

Installing PyCIGAR
------------------------

In this section we install PyCIGAR.

If you have not done so already, download the PyCIGAR github repository.

::

    git clone https://github.com/lbnl-cybersecurity/smart-inverter-adaptive-control.git
    cd pycigar

We begin by creating a conda environment and installing PyCIGAR and its
dependencies within the environment. This can be done by running the below
script. Be sure to run the below commands from ``/path/to/pycigar``.

::

    # create a conda environment
    conda env create -f environment.yml
    source activate pycigar

If the conda install fails, you can also install the requirements using pip by calling

::

    # install PyCIGAR within the environment
    pip install -e .

The option `-e` will allow you to modify PyCIGAR module and it will directly change the module
without the need for reinstallation.

Testing your installation
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the above modules have been successfully installed, we can test the
installation by running a few examples. Before trying to run any examples, be
sure to enter your conda environment by typing:

::

    source activate pycigar

Let’s see some action:

::

    python examples/opendss/test_env.py
