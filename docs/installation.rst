============
Installation
============

SPyCi-PDB uses only Python-based APIs for which we expect it to run
native on any system Python can run, as long as third-party back-calculator
installation requirements are met.

Please note that the IDPConformerGenerator library is required for parsing
PDB structure coordinates. Details for IDPConformerGenerator
can be found `here <https://github.com/julie-forman-kay-lab/IDPConformerGenerator>`_.

Full installation instructures are highlighted below.

.. note::
    As highlighted in ``requirements.txt``, listed below are all of the Python packages required.
    The following list will be updated if more packages are added:

    | ``numpy >= 1.22.4, < 2``
    | ``numba >= 0.53.0, < 1``
    | ``pybind11 >= 2.9.2, < 3``
    | ``pandas >= 1.2.4, < 2``
    | ``scipy >= 1.8.1, < 2``
    | ``matplotlib >= 3.5.2, < 4``
    | ``natsort >= 8, < 9``
    | ``tox == 4``
    
    As mentioned above, ``IDPConformerGenerator`` libraries are required.

If you already have an ``idpconfgen`` environment
-------------------------------------------------
It is recommended to install IDPConformerGenerator first to have the
environment prepared for SPyCi-PDB. Please visit the
`IDPConformerGenerator repository <https://github.com/julie-forman-kay-lab/IDPConformerGenerator>`_
to install IDPConformerGenerator environment first.

Activate the ``idpconfgen`` environment (conda or virtualenv) and
update the environment with the dependencies seen in SPyCi-PDB::

    pip install -r existing_requirements.txt --upgrade

Install SPyCi-PDB on top of IDPConformerGenerator::

    python setup.py develop --no-deps

Standalone From Source
----------------------

Clone from the official repository::

    git clone https://github.com/julie-forman-kay-lab/SPyCi-PDB

Navigate to the new ``SPyCi-PDB`` folder::

    cd SPyCi-PDB

Run the following commands to install ``spycipdb`` dependencies if
Anaconda is used as your Python package manager::

    conda env create -f requirements.yml
    conda activate spycipdb

.. note::
    If you don't use Anaconda to manage your Python installations, you can use
    ``virtualenv`` and the ``requirements.txt`` file following the commands:

    | ``virtualenv spycienv --python=3.8``
    | ``source spycienv/bin/activate``
    | ``pip install -r requirements.txt``

    If you have difficulties installing ``spycipdb``, raise an issue in the
    main GitHub repository, and we will help you.

Install ``spycipdb`` in development mode in order for your installation to be
always up-to-date with the repository::

    python setup.py develop --no-deps

.. note::
    The above applies also if you used ``virtualenv`` instead of ``conda``.

**Remember** to activate the ``spycipdb`` environment every time you open a new
terminal window, from within the repository folder, choose yours::

    # Installation with Anaconda
    conda activate spycipdb

    # Installation with virtualenv
    source spycienv/bin/activate

To update to the latest version, navigate to the repository folder, activate the
``spycipdb`` python environment as described above, and run the commands::

    git pull

    # if you used anaconda to create the python environment, run:
    conda env update -f requirements.yml

    # if you used virtual env to create the python environment, run:
    pip install -r requirements.txt  --upgrade

    python setup.py develop --no-deps

Your installation will become up to date with the latest developments.

Please note that IDPConformerGenerator will be installed alongside SPyCi-PDB
as the library is required for processing PDB files. You can find the installation
of IDPConformerGenerator in ``src/idpconfgen/``.

Updating SPyCi-PDB
------------------
When new back-calculators are added or when dependencies change from the
GitHub, please update your environment with the latest dependencies after
performing a ``git pull``.

To update an anaconda environment::

    conda env update --file requirements.yml --prune

To update a virtualenv environment::

    pip install -r requirements.txt --upgrade

Then perform ``setup.py`` again for good measure::

    python setup.py develop --no-deps

Installing Third-party Software
---------------------------------------

Some functionalities of ``SPyCi-PDB`` require third-party software.
These are not mandatory to install unless you want to use such operations.

DEERPREdict
```````````

.. note::
    A default PRE calculator will be installed along with ``SPyCi-PDB``.

    However, if you would like to use DEERPREdict as an alternative
    predictor for PRE values, specifically, predicting intensity ratios
    instead of the default distance calculation, please see below.

If you've installed a recent version of SPyCi-PDB (past v0.5.X), DEERPREdict
should already be installed in your Python environment.

If you do not have DEERPREdict as a package, the installation of DEERPREdict
can be done by using ``pip`` as highlighted on their `website<https://deerpredict.readthedocs.io/_docs/installation.html>`_.

Specifically you can install DEERpredict on-top of the SPyCi-PDB environment.::

    pip install DEERPREdict


UCBShift
````````

.. note::
    Module is only required if you wish to perform Chemical Shift back-calculations.
    The following installation will revert Python to version 3.8.15.
    
    You should be in the parent directory where ``SPyCi-PDB`` was cloned to
    with the Python environment is deactivated.

Clone the UCBShift repository from `Zi Hao Liu's fork <https://github.com/menoliu/CSpred>`_ on
GitHub.::

    git clone https://github.com/menoliu/CSpred

Enter the ``CSpred`` folder and make a ``models`` directory, then download and
extract the latest `trained models <https://datadryad.org/stash/dataset/doi:10.6078/D1B974>`_
to ``CSpred/models`` directory.

Move back into the ``SPyCi-PDB`` directory and enter the ``thirdparty/ucbshift_reqs`` folder::

    cd ..
    cd SPyCi-PDB
    cd ./thirdparty/ucbshift_reqs

Run the following commands to install ``UCBShift`` dependencies if
Anaconda is used as your Python package manager::

    conda env update --name spycipdb --file ucbshift_requirements.yml --prune

Run the following commands to install ``UCBShift`` dependencies if
virtualenv was used to install SPyCi-PDB::

    pip install -r ucbshift_requirements.txt

Go back to the ``SPyCi-PDB`` directory and reinstall ``spycipdb`` and
``idpconfgen`` if needed.::
    
    conda activate spycipdb
    cd ../..
    python setup.py develop --no-deps
    cd ./src/idpconfgen/
    python setup.py develop --no-deps

The following is the same with virtualenv::

    source spycienv/bin/activate
    cd ../..
    python setup.py develop --no-deps
    cd ./src/idpconfgen/
    python setup.py develop --no-deps

Currently, reinstallation is required as UCBShift changes the Python version.
We will be working on a fix to streamline this process soon by using package
handlers such as ``PyPi``. Thank you for your patience.

If you're running into DSSP errors, please make sure you only have one version
of DSSP installed and the preferred DSSP version is 3.X. Please refer to
`this issue <https://github.com/julie-forman-kay-lab/IDPConformerGenerator/issues/48>`_
for troubleshooting an older version of DSSP on a newer version of Ubuntu.

ATSAS v3.1.1 - CRYSOL v3.0
``````````````````````````

.. note::
    ATSAS installation is only required for the :code:`saxs` module.

Please visit the `ATSAS website <https://www.embl-hamburg.de/biosaxs/download.html>`_
to download v3.1.1 of ATSAS. Theoretically, SPyCi-PDB will work if you already
have ATSAS v3.X installed.

Test your installation via::

    crysol -h

PALES v6.0
``````````

.. note::
    PALES installation is only required for the :code:`rdc` module.

A package of PALES v6.0 for Linux is already included in the ``thirdparty/`` folder.
Downloaded and extracted from the `Ad Bax Group <https://spin.niddk.nih.gov/bax/software/PALES/index.html>`_.

For use with x64 bit Linux Ubuntu 20.04.X LTS and 18.04.X, you must install the i386 architecture
along with required package libraries::

    sudo dpkg --add-architecture i386
    sudo apt update
    sudo apt install libc6:i386 libncurses5:i386 libstdc++6:i386 libx11-6:i386

If the last command above fails, run the following instead::

    sudo apt install multiarch-support
