#include "../../../include/uq/sampler/sampler.hpp"
//#include "../../../include/uq/sampler/sobol.hpp"
#include "../../../include/includes.hpp"

#include <fmt/format.h>


//! Template specialization
//! for Uniform distribution between [-1, +1]

//! Constructors
Sampler <Uniform> :: Sampler (const nlohmann::json& config)
    : m_unif(-1, +1),
      m_gen(Generator<std::uniform_real_distribution
            <double>>())
{
    m_nparams = config["uq_nparams"];
    m_nsamples = config["uq_nsamples"];
}

Sampler <Uniform> :: Sampler (const int nparams,
                              const int nsamples)
    : m_unif(-1, +1),
      m_gen(Generator<std::uniform_real_distribution
            <double>>()),
      m_nparams(nparams), m_nsamples(nsamples) {}

Sampler <Uniform> :: Sampler (const int seed,
                              const int nparams,
                              const int nsamples)
    : m_unif(-1, +1),
      m_gen(Generator<std::uniform_real_distribution
            <double>>(seed)),
      m_nparams(nparams), m_nsamples(nsamples) {}

//! Generates one sample
Vector Sampler <Uniform> :: generate_one_sample()
{
    Vector omegas(m_nparams);
    for(int k=0; k<m_nparams; k++) {
        omegas(k) = m_gen(m_unif);
    }

    return omegas;
}

//! Generates all samples
DenseMatrix Sampler <Uniform> :: generate()
{
    DenseMatrix all_omegas(m_nparams, m_nsamples);
    for(int i=0; i<m_nsamples; i++)
        all_omegas.SetCol(i, generate_one_sample());
    return all_omegas;
}


//! Template specialization
//! for Sobol points
/*
//! Constructors
Sampler <Sobol> :: Sampler (const nlohmann::json& config)
{
    m_nparams = config["uq_nparams"];
    m_nsamples = config["uq_nsamples"];
}

Sampler <Sobol> :: Sampler (const int nparams,
                            const int nsamples)
    : m_seed(0),
      m_nparams(nparams), m_nsamples(nsamples) {}

Sampler <Sobol> :: Sampler (const int seed,
                            const int nparams,
                            const int nsamples)
    : m_seed(seed),
      m_nparams(nparams), m_nsamples(nsamples) {}

//! Generates all samples
DenseMatrix Sampler <Sobol> :: generate()
{
    double *d = i8_sobol_generate(m_nparams, m_nsamples,
                                  m_seed);
    DenseMatrix all_omegas(d, m_nparams, m_nsamples);

    // map to (-1,+1)
    for (int j=0; j<m_nsamples; j++)
        for (int i=0; i<m_nparams; i++) {
            all_omegas(i,j) *= 2;
            all_omegas(i,j) -= 1;
        }

    return all_omegas;
}*/


//! Makes samples
//! by scattering the samples from root
//! to all the other processors
std::pair<DenseMatrix, Array<int>>
make_samples (MPI_Comm& comm,
              std::string samplerType,
              int nparams,
              int mynsamples,
              Array<int> dist_nsamples,
              int seed)
{
    int nprocs, myrank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);

    DenseMatrix all_myomegas(nparams, mynsamples);
    Array<int> all_mysampleIds(mynsamples);
    if (myrank == IamRoot)
    {
        Array<int> cum_dist_nsamples(dist_nsamples);
        cum_dist_nsamples.PartialSum();

        int nsamples = cum_dist_nsamples[nprocs-1];
        DenseMatrix all_omegas(nparams, nsamples);
        Array<int> all_sampleIds(nsamples);
        if (samplerType == "uniform")
        {
            Sampler<Uniform> unifSampler
                    (seed, nparams, nsamples);
            all_omegas = unifSampler.generate();
        }
        /*else if (samplerType == "sobol") {
            Sampler<Sobol> sobolSampler
                    (seed, nparams, nsamples);
            all_omegas = sobolSampler.generate();
        }*/

        // enumerate all samples
        for (int i=0; i<nsamples; i++) {
            all_sampleIds[i] = i;
        }

        // scatter all omegas to groups
        Array<int> sendcounts(nprocs);
        Array<int> displs(nprocs);
        displs[0] = 0;
        sendcounts[0] = dist_nsamples[0]*nparams;
        for (int i=1; i<nprocs; i++) {
            displs[i] = cum_dist_nsamples[i-1]*nparams;
            sendcounts[i] = dist_nsamples[i]*nparams;
        }
        int info = MPI_Scatterv(all_omegas.Data(),
                                sendcounts.GetData(),
                                displs.GetData(),
                                MPI_DOUBLE,
                                all_myomegas.Data(),
                                mynsamples*nparams,
                                MPI_DOUBLE,
                                IamRoot, comm);
        if (info != MPI_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "Error while scattering samples. "
                "[{}]", info));
        }

        // scatter all sampleIds to groups
        displs[0] = 0;
        sendcounts[0] = dist_nsamples[0];
        for (int i=1; i<nprocs; i++) {
            displs[i] = cum_dist_nsamples[i-1];
            sendcounts[i] = dist_nsamples[i];
        }
        info = MPI_Scatterv(all_sampleIds.GetData(),
                            sendcounts.GetData(),
                            displs.GetData(),
                            MPI_INT,
                            all_mysampleIds.GetData(),
                            mynsamples,
                            MPI_INT,
                            IamRoot, comm);
        if (info != MPI_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "Error while scattering sampleIds. "
                "[{}]", info));
        }
    }
    else
    {
        // scatter all omegas to groups
        int info = MPI_Scatterv(nullptr,
                                nullptr,
                                nullptr,
                                MPI_DOUBLE,
                                all_myomegas.Data(),
                                mynsamples*nparams,
                                MPI_DOUBLE,
                                IamRoot, comm);
        if (info != MPI_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "Error while scattering samples. "
                "[{}]", info));
        }

        // scatter all sampleIds to groups
        info = MPI_Scatterv(nullptr,
                            nullptr,
                            nullptr,
                            MPI_INT,
                            all_mysampleIds.GetData(),
                            mynsamples,
                            MPI_INT,
                            IamRoot, comm);
        if (info != MPI_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "Error while scattering sampleIds. "
                "[{}]", info));
        }
    }

    return {all_myomegas, all_mysampleIds};
}


//! Makes samples
//! by scattering the samples from root
//! to all the other processors
std::pair<DenseMatrix, Array<int>>
make_samples (MPI_Comm& intra_group_comm,
              MPI_Comm& inter_group_comm,
              std::string samplerType,
              int nparams,
              int mynsamples,
              Array<int> dist_nsamples,
              int seed)
{
    int ngroups, mygroup, mygrouprank;
    MPI_Comm_size(inter_group_comm, &ngroups);
    MPI_Comm_rank(inter_group_comm, &mygroup);
    MPI_Comm_rank(intra_group_comm, &mygrouprank);

    DenseMatrix all_myomegas(nparams, mynsamples);
    Array<int> all_mysampleIds(mynsamples);
    if (mygroup == IamRoot)
    {
        Array<int> cum_dist_nsamples(dist_nsamples);
        cum_dist_nsamples.PartialSum();

        int nsamples = cum_dist_nsamples[ngroups-1];
        DenseMatrix all_omegas(nparams, nsamples);
        Array<int> all_sampleIds(nsamples);

        // generate all omegas on global root
        if (mygrouprank == IamRoot) {
            if (samplerType == "uniform")
            {
                Sampler<Uniform> unifSampler
                        (seed, nparams, nsamples);
                all_omegas = unifSampler.generate();
            }
            /*else if (samplerType == "sobol") {
                Sampler<Sobol> sobolSampler
                        (seed, nparams, nsamples);
                all_omegas = sobolSampler.generate();
            }*/
            //std::cout << "\n\n";
            //all_omegas.Print();

            // enumerate all samples
            for (int i=0; i<nsamples; i++) {
                all_sampleIds[i] = i;
            }
        }

        // broadcast all omegas to the root group
        MPI_Bcast(all_omegas.Data(), nparams*nsamples,
                  MPI_DOUBLE, IamRoot, intra_group_comm);

        // scatter all omegas to groups
        Array<int> sendcounts(ngroups);
        Array<int> displs(ngroups);
        displs[0] = 0;
        sendcounts[0] = dist_nsamples[0]*nparams;
        for (int i=1; i<ngroups; i++) {
            displs[i] = cum_dist_nsamples[i-1]*nparams;
            sendcounts[i] = dist_nsamples[i]*nparams;
        }
        int info = MPI_Scatterv(all_omegas.Data(),
                                sendcounts.GetData(),
                                displs.GetData(),
                                MPI_DOUBLE,
                                all_myomegas.Data(),
                                mynsamples*nparams,
                                MPI_DOUBLE,
                                IamRoot, inter_group_comm);
        if (info != MPI_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "Error while scattering samples. "
                "[{}]", info));
        }

        // broadcast all sampleIds to the root group
        MPI_Bcast(all_sampleIds.GetData(), nsamples,
                  MPI_INT, IamRoot, intra_group_comm);

        // scatter all sampleIds to groups
        displs[0] = 0;
        sendcounts[0] = dist_nsamples[0];
        for (int i=1; i<ngroups; i++) {
            displs[i] = cum_dist_nsamples[i-1];
            sendcounts[i] = dist_nsamples[i];
        }
        info = MPI_Scatterv(all_sampleIds.GetData(),
                            sendcounts.GetData(),
                            displs.GetData(),
                            MPI_INT,
                            all_mysampleIds.GetData(),
                            mynsamples,
                            MPI_INT,
                            IamRoot, inter_group_comm);
        if (info != MPI_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "Error while scattering sampleIds. "
                "[{}]", info));
        }
    }
    else
    {
        // scatter all omegas to groups
        int info = MPI_Scatterv(nullptr,
                                nullptr,
                                nullptr,
                                MPI_DOUBLE,
                                all_myomegas.Data(),
                                mynsamples*nparams,
                                MPI_DOUBLE,
                                IamRoot, inter_group_comm);
        if (info != MPI_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "Error while scattering samples. "
                "[{}]", info));
        }

        // scatter all sampleIds to groups
        info = MPI_Scatterv(nullptr,
                            nullptr,
                            nullptr,
                            MPI_INT,
                            all_mysampleIds.GetData(),
                            mynsamples,
                            MPI_INT,
                            IamRoot, inter_group_comm);
        if (info != MPI_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "Error while scattering sampleIds. "
                "[{}]", info));
        }
    }

    return {all_myomegas, all_mysampleIds};
}


//! Makes sample Ids
//! by scattering the sample Ids from root
//! to all the other processors
Array<int>
make_sampleIds (MPI_Comm& comm,
                int mynsamples,
                Array<int> dist_nsamples)
{
    int nprocs, myrank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);

    Array<int> all_mysampleIds(mynsamples);
    if (myrank == IamRoot)
    {
        Array<int> cum_dist_nsamples(dist_nsamples);
        cum_dist_nsamples.PartialSum();

        int nsamples = cum_dist_nsamples[nprocs-1];
        Array<int> all_sampleIds(nsamples);

        // enumerate all samples
        for (int i=0; i<nsamples; i++) {
            all_sampleIds[i] = i;
        }

        // scatter all sampleIds to groups
        Array<int> sendcounts(nprocs);
        Array<int> displs(nprocs);
        displs[0] = 0;
        sendcounts[0] = dist_nsamples[0];
        for (int i=1; i<nprocs; i++) {
            displs[i] = cum_dist_nsamples[i-1];
            sendcounts[i] = dist_nsamples[i];
        }
        int info = MPI_Scatterv(all_sampleIds.GetData(),
                                sendcounts.GetData(),
                                displs.GetData(),
                                MPI_INT,
                                all_mysampleIds.GetData(),
                                mynsamples,
                                MPI_INT,
                                IamRoot, comm);
        if (info != MPI_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "Error while scattering sampleIds. "
                "[{}]", info));
        }
    }
    else
    {
        // scatter all sampleIds to groups
        int info = MPI_Scatterv(nullptr,
                                nullptr,
                                nullptr,
                                MPI_INT,
                                all_mysampleIds.GetData(),
                                mynsamples,
                                MPI_INT,
                                IamRoot, comm);
        if (info != MPI_SUCCESS) {
            throw std::runtime_error(fmt::format(
                "Error while scattering sampleIds. "
                "[{}]", info));
        }
    }

    return all_mysampleIds;
}

// End of file
