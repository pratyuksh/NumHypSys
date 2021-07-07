#ifndef MYMFEM_PERROR_HPP
#define MYMFEM_PERROR_HPP

#include "mfem.hpp"
using namespace mfem;


//! Computes H1 seminorm of parallel grid function
//! wrt to coefficient
double globalH1SemiNorm
(MPI_Comm comm, const ParGridFunction &u,
 Coefficient &u_Ref, VectorCoefficient &uGrad_ref);

#endif /// MYMFEM_PERROR_HPP
