#include "../../../include/uq/scheduler/communicator.hpp"
#include "../../../include/uq/scheduler/self_scheduler.hpp"
#include "../../../include/includes.hpp"


//! Constructors
Communicator :: Communicator (MPI_Comm& comm,
                              const nlohmann::json& config)
    : m_global_comm (comm)
{
    m_nprocsPerGroup = config["uq_comm_nprocs_per_group"];
}

Communicator :: Communicator (MPI_Comm& comm,
                              const int nprocsPerGroup)
    : m_global_comm (comm), m_nprocsPerGroup (nprocsPerGroup) {}

//! Creates the inter- and intra-group communicators
void Communicator :: operator() (void)
{
    int nprocs, myrank;
    MPI_Comm_size(m_global_comm, &nprocs);
    MPI_Comm_rank(m_global_comm, &myrank);
    assert(nprocs%m_nprocsPerGroup == 0);

    // create an intra-group communicator
    int mygroupcolor = myrank/m_nprocsPerGroup;
    MPI_Comm_split(m_global_comm, mygroupcolor, myrank,
                   &m_intra_group_comm);

    // create an inter-group communicator
    int mygrouprank;
    MPI_Comm_rank(m_intra_group_comm, &mygrouprank);
    MPI_Comm_split(m_global_comm, mygrouprank, myrank,
                   &m_inter_group_comm);
}

//! Creates the inter- and intra-group communicators
void Communicator :: selfScheduling (void)
{
    int nprocs, myrank;
    MPI_Comm_size(m_global_comm, &nprocs);
    MPI_Comm_rank(m_global_comm, &myrank);
    int nworkers = nprocs-1;
    assert(nworkers%m_nprocsPerGroup == 0);

    int mygroupcolor;
    if (myrank == MANAGER) {
        mygroupcolor = 0; // group with only the manager
    }
    else {
        mygroupcolor = 1+(myrank-1)/m_nprocsPerGroup;
    }
    // create a communicator within a group of workers
    MPI_Comm_split(m_global_comm, mygroupcolor, myrank,
                   &m_intra_group_comm);

    // create a communicator between the manager
    // and the root workers of different groups
    int mygrouprank;
    MPI_Comm_rank(m_intra_group_comm, &mygrouprank);
    MPI_Comm_split(m_global_comm, mygrouprank, myrank,
                   &m_inter_group_comm);
}

//! Makes inter- and intra-group communicators
std::pair<MPI_Comm, MPI_Comm>
make_communicators (MPI_Comm& global_comm, int nprocsPerGroup)
{
    Communicator communicator(global_comm, nprocsPerGroup);
    communicator();
    return {communicator.get_intra_group_comm(),
            communicator.get_inter_group_comm()};
}

//! Makes inter- and intra-group communicators
//! for self-scheduling
std::pair<MPI_Comm, MPI_Comm>
make_selfScheduling_communicators
(MPI_Comm& global_comm, int nworkersPerGroup)
{
    Communicator communicator(global_comm, nworkersPerGroup);
    communicator.selfScheduling();
    return {communicator.get_intra_group_comm(),
            communicator.get_inter_group_comm()};
}


// End of file
