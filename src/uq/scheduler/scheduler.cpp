#include "../../../include/uq/scheduler/scheduler.hpp"
#include "../../../include/includes.hpp"

#include <fmt/format.h>


//! Constructors
Scheduler :: Scheduler (MPI_Comm& comm,
                        const nlohmann::json& config)
    : m_comm (comm)
{ m_nsamples = config["uq_nsamples"]; }

Scheduler :: Scheduler (MPI_Comm& comm,
                        const int nsamples)
    : m_comm (comm), m_nsamples (nsamples)
{}

//! Creates the schedule
//! by distributing samples to processors
Array<int> Scheduler :: operator() (void)
{
    int nprocs;
    MPI_Comm_size(m_comm, &nprocs);

    Array<int> dist_nsamples(nprocs);
    dist_nsamples = m_nsamples/nprocs;
    for(int k=0; k<m_nsamples%nprocs; k++) {
        dist_nsamples[k]++;
    }

    return dist_nsamples;
}


//! Makes the schedule by distributing the scheduling info
//! from root to all the other processors
std::pair<int, Array<int>> make_schedule (MPI_Comm& comm,
                                          int nsamples)
{
    int nprocs, myrank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);

    int mynsamples;
    Array<int> dist_nsamples;

    if (myrank == IamRoot)
    {
        Scheduler scheduler(comm, nsamples);
        dist_nsamples = scheduler();

        // scatter scheduling info to groups
        int info = MPI_Scatter(dist_nsamples.GetData(),
                               1,
                               MPI_INT,
                               &mynsamples,
                               1,
                               MPI_INT,
                               IamRoot, comm);
        if (info != MPI_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "Error while scheduling. "
                "[{}]", info));
        }
    }
    else
    {
        // scatter scheduling info to groups
        int info = MPI_Scatter(nullptr,
                               0,
                               MPI_INT,
                               &mynsamples,
                               1,
                               MPI_INT,
                               IamRoot, comm);
        if (info != MPI_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "Error while scheduling. "
                "[{}]", info));
        }
    }
    //std::cout << "My rank: "
    //          << myrank << "\t"
    //          << mynsamples << std::endl;

    return {mynsamples, dist_nsamples};
}


// End of file
