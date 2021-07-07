#ifndef UQ_SCHEDULER_HPP
#define UQ_SCHEDULER_HPP

#include "../../core/config.hpp"
#include "mfem.hpp"

using namespace mfem;


//! Class Scheduler
//! creates the schedule
//! by distributing samples to processors
class Scheduler
{
public:
    //! Constructors
    explicit Scheduler (MPI_Comm &comm,
                        const nlohmann::json& config);

    explicit Scheduler (MPI_Comm &comm,
                        const int nsamples);
    
    //! Creates the schedule
    //! by distributing samples to processors
    Array<int> operator() (void);
    
private:
    MPI_Comm m_comm;
    int m_nsamples;
};

//! Makes the schedule by distributing the scheduling info
//! from root to all the other processors
std::pair<int, Array<int>> make_schedule (MPI_Comm&, int);


#endif /// UQ_SCHEDULER_HPP
