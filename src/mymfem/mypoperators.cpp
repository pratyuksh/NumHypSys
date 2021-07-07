#include "../../include/mymfem/mypoperators.hpp"
#include <assert.h>


//! Applies the mean-free pressure operator
void mymfem :: ParMeanFreePressureOp :: 
Mult (const Vector &x, Vector& y) const
{
    assert(x.Size() == m_lfone->Size());
    assert(y.Size() == m_lfone->Size());

    if (m_extOp)
    {
        m_extOp->Mult(x, y);
        double mean = 0;
        double mymean = ((*m_lfone)*y);
        MPI_Allreduce(&mymean, &mean, 1, MPI_DOUBLE, MPI_SUM,
                      m_lfone->GetComm());
        mean /= m_area;
        y.Add(-mean, *m_massInvLfone);
    }
    else
    {
        y = x;
        double mean = 0;
        double mymean = ((*m_lfone)*x);
        MPI_Allreduce(&mymean, &mean, 1, MPI_DOUBLE, MPI_SUM,
                      m_lfone->GetComm());
        mean /= m_area;
        //std::cout << m_myrank << "\t"
        //          << m_lfone->Size() << "\t"
        //          << x.Size() << "\t"
        //          << mymean << "\t"
        //          << mean << "\t"
        //          << m_area << std::endl;
        y.Add(-mean, *m_massInvLfone);
    }
}
