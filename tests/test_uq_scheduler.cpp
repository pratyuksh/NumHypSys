#include <gtest/gtest.h>

#include <iostream>
#include "assert.h"

#include "../include/includes.hpp"
#include "../include/uq/scheduler/communicator.hpp"
#include "../include/uq/scheduler/scheduler.hpp"

using namespace std;


TEST(Scheduling, communicator)
{
    int nprocs, myrank;
    MPI_Comm global_comm = MPI_COMM_WORLD;
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    // test 1
    if (nprocs == 6) {
        int nprocsPerGroup = 3;

        MPI_Comm intra_group_comm, inter_group_comm;
        std::tie (intra_group_comm, inter_group_comm)
                = make_communicators(global_comm,
                                     nprocsPerGroup);

        int myintragrouprank, mygroup;
        MPI_Comm_rank(intra_group_comm, &myintragrouprank);
        MPI_Comm_rank(inter_group_comm, &mygroup);

        // truth values
        int true_myintragrouprank, true_mygroup;
        if (myrank == 0) {
            true_myintragrouprank= 0;
            true_mygroup = 0;
        }
        else if (myrank == 1) {
            true_myintragrouprank = 1;
            true_mygroup= 0;
        }
        else if (myrank == 2) {
            true_myintragrouprank = 2;
            true_mygroup= 0;
        }
        else if (myrank == 3) {
            true_myintragrouprank = 0;
            true_mygroup= 1;
        }
        else if (myrank == 4) {
            true_myintragrouprank = 1;
            true_mygroup= 1;
        }
        else if (myrank == 5) {
            true_myintragrouprank = 2;
            true_mygroup= 1;
        }

        ASSERT_EQ(myintragrouprank, true_myintragrouprank);
        ASSERT_EQ(mygroup, true_mygroup);
    }

    // test 2
    if (nprocs == 6) {
        int nprocsPerGroup = 2;

        MPI_Comm intra_group_comm, inter_group_comm;
        std::tie (intra_group_comm, inter_group_comm)
                = make_communicators(global_comm,
                                     nprocsPerGroup);

        int myintragrouprank, mygroup;
        MPI_Comm_rank(intra_group_comm, &myintragrouprank);
        MPI_Comm_rank(inter_group_comm, &mygroup);

        // truth values
        int true_myintragrouprank, true_mygroup;
        if (myrank == 0) {
            true_myintragrouprank= 0;
            true_mygroup = 0;
        }
        else if (myrank == 1) {
            true_myintragrouprank = 1;
            true_mygroup= 0;
        }
        else if (myrank == 2) {
            true_myintragrouprank = 0;
            true_mygroup= 1;
        }
        else if (myrank == 3) {
            true_myintragrouprank = 1;
            true_mygroup= 1;
        }
        else if (myrank == 4) {
            true_myintragrouprank = 0;
            true_mygroup= 2;
        }
        else if (myrank == 5) {
            true_myintragrouprank = 1;
            true_mygroup= 2;
        }

        ASSERT_EQ(myintragrouprank, true_myintragrouprank);
        ASSERT_EQ(mygroup, true_mygroup);
    }
}


TEST(Scheduling, scheduler)
{
    int nprocs, myrank;
    MPI_Comm global_comm = MPI_COMM_WORLD;
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    // test 1
    if (nprocs == 6) {
        int nsamples = 16;
        int mynsamples;
        Array<int> dist_nsamples;
        std::tie(mynsamples, dist_nsamples)
                = make_schedule(global_comm, nsamples);

        int true_mynsamples;
        if (myrank < 4) {
            true_mynsamples = 3;
        } else {
            true_mynsamples = 2;
        }
        ASSERT_EQ(true_mynsamples, mynsamples);

        Array<int> true_dist_nsamples;
        if (myrank == IamRoot) {
            true_dist_nsamples.SetSize(6);
            true_dist_nsamples = 3;
            true_dist_nsamples[4] = 2;
            true_dist_nsamples[5] = 2;

            for (int i=0; i<6; i++)
                ASSERT_EQ(true_dist_nsamples[i],
                          dist_nsamples[i]);
        } else {
            ASSERT_EQ(dist_nsamples.Size(), 0);
        }
    }

    // test 1
    if (nprocs == 6) {
        int nprocsPerGroup = 2;

        MPI_Comm intra_group_comm, inter_group_comm;
        std::tie (intra_group_comm, inter_group_comm)
                = make_communicators(global_comm,
                                     nprocsPerGroup);

        int myintragrouprank, mygroup;
        MPI_Comm_rank(intra_group_comm, &myintragrouprank);
        MPI_Comm_rank(inter_group_comm, &mygroup);

        int nsamples = 16;
        int mynsamples;
        Array<int> dist_nsamples;
        std::tie(mynsamples, dist_nsamples)
                = make_schedule(inter_group_comm, nsamples);

        int true_mynsamples;
        if (mygroup < 1) {
            true_mynsamples = 6;
        } else {
            true_mynsamples = 5;
        }
        ASSERT_EQ(true_mynsamples, mynsamples);

        Array<int> true_dist_nsamples;
        if (mygroup == IamRoot) {
            true_dist_nsamples.SetSize(3);
            true_dist_nsamples = 5;
            true_dist_nsamples[0] = 6;

            for (int i=0; i<3; i++)
                ASSERT_EQ(true_dist_nsamples[i],
                          dist_nsamples[i]);
        } else {
            ASSERT_EQ(dist_nsamples.Size(), 0);
        }
    }
}
