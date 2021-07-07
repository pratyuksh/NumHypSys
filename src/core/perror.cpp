#include "../../include/core/perror.hpp"


//! Computes H1 seminorm of parallel grid function
//! wrt to coefficient
double globalH1SemiNorm
(MPI_Comm comm, const ParGridFunction &u,
 Coefficient &u_ref, VectorCoefficient &uGrad_ref)
{
    int myrank;
    MPI_Comm_rank(comm, &myrank);
    
    ConstantCoefficient one(1.0);
    double loc_seminorm_H1 = u.ComputeH1Error
            (&u_ref, &uGrad_ref, &one, 1.0, 1);
    
    double glob_seminorm_H1;
    loc_seminorm_H1 *= loc_seminorm_H1;
    MPI_Allreduce(&loc_seminorm_H1, &glob_seminorm_H1, 1,
                  MPI_DOUBLE, MPI_SUM, comm);
    
    return sqrt(glob_seminorm_H1);
}

// End of file
