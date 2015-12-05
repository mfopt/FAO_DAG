# FAO_DAG

The simplest way to use matrix-free CVXPY and matrix-free cone solvers is to clone the matrix-free CVXPY [repository here](https://github.com/SteveDiamond/cvxpy) and the matrix-free SCS [repository here](https://github.com/SteveDiamond/scs) and install the packages from source.

It is possible to use matrix-free CVXPY with a faster version of matrix-free SCS.
Simply follow these instructions:

1. Switch to the ``pogs`` branch of matrix-free CVXPY and reinstall the package.
2. Clone this repository.
3. Run ``git submodule init`` and ``git submodule update``.
4. Go to the ``include/scs`` directory and run ``git checkout FAO`` followed by ``make``.
5. Go to the ``src/python`` directory and run ``python setup.py install``.

The faster version of matrix-free SCS does not support all the linear functions
available in CVXPY.
To call the faster version of matrix-free SCS from CVXPY, set the solver option
``solver=MAT_FREE_SCS``.
You can still use the slower version of matrix-free SCS that supports all the
linear functions by setting the solver option ``solver=OLD_SCS_MAT_FREE``.

It is also possible to install the matrix-free POGS solver, which
runs on a GPU. Follow these instructions:

1. Switch to the ``pogs`` branch of matrix-free CVXPY and reinstall the package.
2. Clone this repository and run ``git checkout pogs_gpu``.
3. Run ``git submodule init`` and ``git submodule update``.
4. Go to the ``include/pogs_fork`` directory and run ``git checkout cone`` followed by ``make gpu``.
5. Go to the ``src/python`` directory and modify and run ``make.sh``.

Matrix-free POGS does not support all the linear functions available in CVXPY.
The installation above also only allows you to use matrix-free POGS within
the ``src/python`` directory.
To call matrix-free POGS from CVXPY, set the solver option
``solver=MAT_FREE_POGS``.
The solver option ``double=True/False`` specifies whether single or double
precision floating point is used in the solver.
