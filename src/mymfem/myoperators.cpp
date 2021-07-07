#include "../../include/mymfem/myoperators.hpp"
#include <assert.h>


//! Applies the mean-free pressure operator
void mymfem :: MeanFreePressureOp :: 
Mult (const Vector &x, Vector& y) const
{
    assert(x.Size() == m_lfone.Size());
    assert(y.Size() == m_lfone.Size());

    if (m_extOp)
    {
        m_extOp->Mult(x, y);
        double mean = (m_lfone*y)/m_area;
        y.Add(-mean, m_massInvLfone);
    }
    else
    {
        y = x;
        double mean = (m_lfone*x)/m_area;
        //std::cout << x.Size() << "\t"
        //          << mean  << "\t"
        //          << m_area << std::endl;
        y.Add(-mean, m_massInvLfone);
    }
}
