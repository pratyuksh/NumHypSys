#include "../../../include/uq/stats/stats.hpp"

#include <assert.h>

#include "../../../include/includes.hpp"


//! Constructor
Statistics :: Statistics (MPI_Comm& comm,
                          int nsamples, int size)
    : m_comm(comm), m_nsamples(nsamples), m_size(size)
{
    m_first.SetSize(size); m_first = 0.;
    m_second.SetSize(size); m_second = 0.;
}

//! Adds a sample to compute first moment
void Statistics :: add_to_first_moment(Vector& u)
{
    assert(u.Size() == m_size);
    m_first += u;
}

//! Adds a sample to compute second moment
void Statistics :: add_squared_to_second_moment(Vector& u)
{
    assert(u.Size() == m_size);
    for (int k=0; k<m_size; k++) {
        m_second(k) += u(k)*u(k);
    }
}

//! Evaluates mean from first moment
void Statistics :: mean (Vector& avg)
{
    MPI_Reduce(m_first.GetData(), avg.GetData(),
               m_first.Size(), MPI_DOUBLE,
               MPI_SUM, IamRoot, m_comm);

    if (avg.Size() > 0) {
        avg *= (1./m_nsamples);
    }
}

//! Evaluates variance from mean and second moment
void Statistics :: variance (const Vector& avg,
                             Vector& var)
{
    MPI_Reduce(m_second.GetData(), var.GetData(),
               m_second.Size(), MPI_DOUBLE,
               MPI_SUM, IamRoot, m_comm);

    if (var.Size() && avg.Size())
    {
        var *= (1./(m_nsamples-1));
        double coeff = m_nsamples * 1./(m_nsamples - 1);
        for (int k=0; k<m_size; k++) {
            var(k) -= avg(k)*avg(k)*coeff;
        }
    }
}

// End of file
