#ifndef UQ_COMMUNICATOR_HPP
#define UQ_COMMUNICATOR_HPP

#include "../../core/config.hpp"
#include "mpi.h"


//! Class Communication
//! creates inter- and intra-group communicators
class Communicator
{
public:
    //! Constructors
    explicit Communicator (MPI_Comm &comm,
                           const nlohmann::json& config);

    explicit Communicator (MPI_Comm &comm,
                           const int nprocsPerGroup);
    
    //! Creates the inter- and intra-group communicators
    void operator() (void);

    //! Creates manager and workers communicator groups
    void selfScheduling (void);

    //! Returns intra-group communicator
    inline MPI_Comm get_intra_group_comm() {
        return m_intra_group_comm;
    }

    //! Returns inter-group communicator
    inline MPI_Comm get_inter_group_comm() {
        return m_inter_group_comm;
    }
    
private:
    MPI_Comm m_global_comm;
    MPI_Comm m_intra_group_comm, m_inter_group_comm;
    int m_nprocsPerGroup;
};

//! Makes inter- and intra-group communicators
std::pair<MPI_Comm, MPI_Comm> make_communicators
(MPI_Comm&, int);

//! Makes inter- and intra-group communicators
//! for self-scheduling
std::pair<MPI_Comm, MPI_Comm>
make_selfScheduling_communicators (MPI_Comm&, int);


#endif /// UQ_COMMUNICATOR_HPP
