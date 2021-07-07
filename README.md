# Numerical approximation of Statistical Solutions of incompressible Navier-Stokes Equations
This repository consists of divergence-free H(div) numerical solvers for the incompressible Navier-Stokes Equations and Stokes equations. The code is written in C++ and it has both serial and MPI-parallelized versions of the numerical solvers.

## Dependencies
1. MFEM-4.1, you can read more [info](https://mfem.org) and [download](https://mfem.org/download).
Follow the [instructions](https://mfem.org/building/) for the parallel-build of MFEM.
This library further depends on many libraries, of which only the following are needed:
    1. Hypre, version - 2.10.0b
    1. PARMETIS, version - 4.0.3
    1. BLAS, LAPACK

1. Eigen library, [info](https://eigen.tuxfamily.org), download version - 3.3.7.

1. For writing the config files [JSON](https://github.com/nlohmann/json), download version - 3.7.0
1. The formatting library [fmt](https://fmt.dev/6.0.0).

1. Unit testing with [GoogleTest](https://github.com/google/googletest) framework.