#include <gtest/gtest.h>

#include <iostream>
#include "assert.h"

#include "../include/includes.hpp"
#include "../include/uq/sampler/sampler.hpp"
#include "../include/uq/scheduler/communicator.hpp"
#include "../include/uq/scheduler/scheduler.hpp"

using namespace std;


TEST(Sampler, sampler1)
{
    int nprocs, myrank;
    MPI_Comm global_comm = MPI_COMM_WORLD;
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    if (nprocs == 6) {
        int nparams = 2;
        int nsamples = 16;

        // make schedule
        int mynsamples;
        Array<int> dist_nsamples;
        std::tie(mynsamples, dist_nsamples)
                = make_schedule(global_comm, nsamples);

        // make samples
        std::string samplerType = "uniform";
        DenseMatrix all_myomegas;
        Array<int> all_mysampleIds;
        std::tie(all_myomegas, all_mysampleIds)
                = make_samples(global_comm,
                               samplerType,
                               nparams, mynsamples,
                               dist_nsamples);

        if (myrank == IamRoot)
        {
            DenseMatrix true_all_omegas(nparams, nsamples);

            Array<int> cum_dist_nsamples(dist_nsamples);
            cum_dist_nsamples.PartialSum();
            int nsamples = cum_dist_nsamples[nprocs-1];

            Sampler<Uniform> unifSampler(nparams,
                                           nsamples);
            true_all_omegas = unifSampler.generate();

            // gather all omegas
            Array<int> recvcounts(nprocs);
            Array<int> displs(nprocs);
            displs[0] = 0;
            recvcounts[0] = dist_nsamples[0]*nparams;
            for (int i=1; i<nprocs; i++) {
                displs[i] = cum_dist_nsamples[i-1]*nparams;
                recvcounts[i] = dist_nsamples[i]*nparams;
            }

            DenseMatrix all_omegas(nparams, nsamples);
            MPI_Gatherv(all_myomegas.Data(),
                        mynsamples*nparams,
                        MPI_DOUBLE,
                        all_omegas.Data(),
                        recvcounts.GetData(),
                        displs.GetData(),
                        MPI_DOUBLE,
                        IamRoot, global_comm);

            for (int j=0; j<nsamples; j++)
                for (int i=0; i<nparams; i++)
                    ASSERT_EQ(true_all_omegas(i,j),
                              all_omegas(i,j));
        }
        else {
            MPI_Gatherv(all_myomegas.Data(),
                        mynsamples*nparams,
                        MPI_DOUBLE,
                        nullptr,
                        nullptr,
                        nullptr,
                        MPI_DOUBLE,
                        IamRoot, global_comm);
        }
    }
}

TEST(Sampler, sampler2)
{
    int nprocs, myrank;
    MPI_Comm global_comm = MPI_COMM_WORLD;
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    if (nprocs == 6) {
        int nprocsPerGroup = 2;
        int ngroups = 3;

        int nparams = 2;
        int nsamples = 8;

        // make communicators
        MPI_Comm intra_group_comm, inter_group_comm;
        std::tie (intra_group_comm, inter_group_comm)
                = make_communicators(global_comm,
                                     nprocsPerGroup);

        int myintragrouprank, mygroup;
        MPI_Comm_rank(intra_group_comm, &myintragrouprank);
        MPI_Comm_rank(inter_group_comm, &mygroup);

        // make schedule
        int mynsamples;
        Array<int> dist_nsamples;
        std::tie(mynsamples, dist_nsamples)
                = make_schedule(inter_group_comm, nsamples);

        // make samples
        std::string samplerType = "uniform";
        DenseMatrix all_myomegas;
        Array<int> all_mysampleIds;
        std::tie(all_myomegas, all_mysampleIds)
                = make_samples(intra_group_comm,
                               inter_group_comm,
                               samplerType,
                               nparams, mynsamples,
                               dist_nsamples);

        if (mygroup == IamRoot)
        {
            DenseMatrix true_all_omegas(nparams, nsamples);

            Array<int> cum_dist_nsamples(dist_nsamples);
            cum_dist_nsamples.PartialSum();
            int nsamples = cum_dist_nsamples[ngroups-1];

            // generate true all omegas on global root
            if (myintragrouprank == IamRoot) {
                Sampler<Uniform> unifSampler(nparams,
                                               nsamples);
                true_all_omegas = unifSampler.generate();
            }
            MPI_Bcast(true_all_omegas.Data(),
                      nparams*nsamples,
                      MPI_DOUBLE, IamRoot, intra_group_comm);
            MPI_Barrier(intra_group_comm);

            // gather all omegas
            Array<int> recvcounts(ngroups);
            Array<int> displs(ngroups);
            displs[0] = 0;
            recvcounts[0] = dist_nsamples[0]*nparams;
            for (int i=1; i<ngroups; i++) {
                displs[i] = cum_dist_nsamples[i-1]*nparams;
                recvcounts[i] = dist_nsamples[i]*nparams;
            }

            DenseMatrix all_omegas(nparams, nsamples);
            MPI_Gatherv(all_myomegas.Data(),
                        mynsamples*nparams,
                        MPI_DOUBLE,
                        all_omegas.Data(),
                        recvcounts.GetData(),
                        displs.GetData(),
                        MPI_DOUBLE,
                        IamRoot, inter_group_comm);
            MPI_Barrier(intra_group_comm);

            for (int j=0; j<nsamples; j++)
                for (int i=0; i<nparams; i++)
                    ASSERT_EQ(true_all_omegas(i,j),
                              all_omegas(i,j));
        }
        else {
            MPI_Gatherv(all_myomegas.Data(),
                        mynsamples*nparams,
                        MPI_DOUBLE,
                        nullptr,
                        nullptr,
                        nullptr,
                        MPI_DOUBLE,
                        IamRoot, inter_group_comm);
        }
    }
}
