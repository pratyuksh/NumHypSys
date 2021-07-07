#ifndef UQ_STATS_HPP
#define UQ_STATS_HPP

#include "../../core/config.hpp"
#include "mfem.hpp"

using namespace mfem;


//! Class Statistics
//! aggregates first and second moments
//! to compute mean and variance
class Statistics
{
public:
    //! Constructor
    explicit Statistics(MPI_Comm& comm,
                        int nsamples, int size);

    //! Adds a sample to compute first moment
    void add_to_first_moment(Vector&);

    //! Adds a sample to compute second moment
    void add_squared_to_second_moment(Vector&);

    //! Evaluates mean from first moment
    void mean(Vector&);

    //! Evaluates variance from mean and second moment
    void variance(const Vector&, Vector&);

    //! Returns first moment
    Vector get_first_moment() {
        return m_first;
    }

    //! Returns second moment
    Vector get_second_moment() {
        return m_second;
    }
    
private:
    MPI_Comm m_comm;

    int m_nsamples, m_size;
    Vector m_first, m_second;
};


#endif /// UQ_STATS_HPP
