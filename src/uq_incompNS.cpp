#include "../include/core/config.hpp"
#include "../include/core/utilities.hpp"
#include "../include/uq/scheduler/self_scheduler.hpp"
#include "../include/uq/sampler/sampler.hpp"
#include "../include/incompNS/solver_factory.hpp"

#include <iostream>

namespace fs = std::filesystem;


class SelfScheduler
{
public:
    //! Constructor
    SelfScheduler (MPI_Comm& comm,
                   const nlohmann::json config,
                   const std::string base_mesh_dir)
        : m_comm(comm), m_config(config)
    {
        int nprocs;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &m_myrank);

        m_nworkers = nprocs-1;
        if (m_nworkers < 1) {
            std::cout << "\nAtleast 1 worker is needed "
                         "for the self-scheduler!\n\n";
            abort();
        }

        std::string sub_mesh_dir = config["mesh_dir"];
        m_mesh_dir = base_mesh_dir+sub_mesh_dir;

        m_lx = config["level_x"];
        m_Nt = config["num_time_steps"];

        m_bool_dumpOut = false;
        if (config.contains("uq_dump_output")) {
            m_bool_dumpOut = config["uq_dump_output"];
        }
        std::string base_out_dir = config["base_out_dir"];
        std::string sub_out_dir = config["sub_out_dir"];
        m_output_dir = base_out_dir+"/"+sub_out_dir+"/";
        m_ensemble_output_dir = m_output_dir
                +"lx"+std::to_string(m_lx)+"/";
        // manager creates output dir, if not there
        if (m_bool_dumpOut && m_myrank == MANAGER) {
            fs::create_directories(m_output_dir);
            fs::create_directories(m_ensemble_output_dir);
        }
        m_meshName_suffix = "_lx"+std::to_string(m_lx);
        m_solName_suffix = "_lx"+std::to_string(m_lx);
        if (m_myrank == MANAGER) {
            std::cout << "Output directory: "
                      << m_output_dir << std::endl;
            std::cout << "Ensemble output directory: "
                      << m_ensemble_output_dir << std::endl;
        }

        m_nparams = config["uq_num_params"];
        if (m_myrank == MANAGER) // manager
        {
            m_nsamples = config["uq_num_samples"];

            m_seed = 0;
            if (config.contains("uq_generator_seed")) {
                m_seed = config["uq_generator_seed"];
            }

            m_samplerType = "uniform";
            if (config.contains("uq_sampler_type")) {
                m_samplerType = config["uq_sampler_type"];
            }
        }
    }

    //! Initializes the manager and workers
    //! Manager creates the tasks
    //! Workers initialize the solvers for tasks
    void init ()
    {
        if (m_myrank == MANAGER) // manager
        {
            m_omegas.SetSize(m_nparams, m_nsamples);
            m_sampleIds.SetSize(m_nsamples);

            // generate all samples
            if (m_samplerType == "uniform")
            {
                Sampler<Uniform> unifSampler
                        (m_seed, m_nparams, m_nsamples);
                m_omegas = unifSampler.generate();
            }
            /*else if (m_samplerType == "sobol") {
                Sampler<Sobol> sobolSampler
                        (m_seed, m_nparams, m_nsamples);
                m_omegas = sobolSampler.generate();
            }*/
            //std::cout << "\n\n";
            m_omegas.Print();

            // enumerate samples
            for (int i=0; i<m_nsamples; i++) {
                m_sampleIds[i] = i;
            }

            dump_samples();
        }
        else // workers
        {
            // test case
            m_testCase = make_incompNS_test_case(m_config);

            // initialize incompNS solver
            m_solver = make_solver
                    (m_config, m_testCase, m_mesh_dir,
                     m_lx, m_Nt);
            m_solver->init();

            // discretisation and observer
            m_discr = m_solver->get_discr();
            m_observer = m_solver->get_observer();

            if (m_myrank == MANAGER+1) {
                dump_mesh();
            }
        }

        MPI_Barrier(m_comm);
    }

    void run()
    {
        if (m_myrank == MANAGER)
        {
            auto samplesDoneByWorker = manager();
            std::cout << "\nSamples done by workers: ";
            samplesDoneByWorker.Print();
            std::cout << "\n\n";
        }
        else // workers
        {
            worker();
        }
    }

    //! Manages the allocation of samples
    Array<int> manager()
    {
        MPI_Status status;
        Vector sendSample;
        Array<int> samplesDoneByWorker(m_nworkers);
        samplesDoneByWorker = 0;

        int nsamplesSent = 0;
        Array<int> curSamplesComp;

        // send first sample to each worker
        for (int worker=1; worker<=m_nworkers; worker++)
        {
            if (nsamplesSent < m_nsamples)
            {
                std::cout << "Manager sent sample "
                          << nsamplesSent << " to worker "
                          << worker << std::endl;

                // append cur list of samples
                // sent for computation
                curSamplesComp.Append(nsamplesSent);

                // send sample number
                MPI_Send(&nsamplesSent, 1, MPI_INT,
                         worker, TAG_SAMPLE_INFO, m_comm);

                // send sample
                m_omegas.GetColumnReference
                        (nsamplesSent, sendSample);
                MPI_Send(sendSample.GetData(),
                         m_nparams, MPI_DOUBLE,
                         worker, TAG_SAMPLE, m_comm);

                // increment sent samples counter
                nsamplesSent++;
            }
            else {
                MPI_Send(&nsamplesSent, 1, MPI_INT,
                         worker, TAG_DONE, m_comm);
            }
        }

        // send/receive to/from workers
        Array<int> samplesDone;
        while (samplesDone.Size() < m_nsamples)
        {
            // receive finished sample info from worker
            int sampleId;
            MPI_Recv(&sampleId, 1, MPI_INT,
                     MPI_ANY_SOURCE, TAG_SAMPLE_DONE,
                     m_comm, &status);
            samplesDone.Append(sampleId);

            // update cur list of samples
            // sent for computation
            curSamplesComp.DeleteFirst(sampleId);
            dump_computeSampleIds(curSamplesComp);

            // dump finished sample ids
            dump_finishedSampleIds(samplesDone);

            // increment number of samples done
            // by this worker
            int worker = status.MPI_SOURCE;
            samplesDoneByWorker[worker-1]++;

            std::cout << "Manager received sample "
                      << sampleId << " from worker "
                      << worker << std::endl;

            if (nsamplesSent < m_nsamples)
            {
                std::cout << "Manager sent sample "
                          << nsamplesSent << " to worker "
                          << worker << std::endl;

                // append cur list of samples
                // sent for computation
                curSamplesComp.Append(nsamplesSent);
                dump_computeSampleIds(curSamplesComp);

                // send sample number
                MPI_Send(&nsamplesSent, 1, MPI_INT,
                         worker, TAG_SAMPLE_INFO, m_comm);

                // send sample
                m_omegas.GetColumnReference
                        (nsamplesSent, sendSample);
                MPI_Send(sendSample.GetData(),
                         m_nparams, MPI_DOUBLE,
                         worker, TAG_SAMPLE, m_comm);

                // increment sent samples counter
                nsamplesSent++;
            }
            else {
                MPI_Send(&nsamplesSent, 1, MPI_INT,
                         worker, TAG_DONE, m_comm);
            }
        }

        return samplesDoneByWorker;
    }

    //! Computes the alloated sample
    void worker()
    {
        MPI_Status status;
        int sampleId;
        Vector omega(m_nparams);

        while (true)
        {
            // receive sample info
            MPI_Recv(&sampleId, 1, MPI_INT,
                     MANAGER, MPI_ANY_TAG,
                     m_comm, &status);

            if (status.MPI_TAG == TAG_SAMPLE_INFO)
            {
                MPI_Recv(omega.GetData(),
                         m_nparams, MPI_DOUBLE,
                         MANAGER, TAG_SAMPLE,
                         m_comm, &status);
                std::cout << "Worker " << m_myrank
                          << " received sample "
                          << sampleId << std::endl;

                // compute solution
                m_testCase->set_perturbations(omega);
                std::unique_ptr<BlockVector> U;
                m_solver->run(U);

                // inform manager when sample done
                MPI_Send(&sampleId, 1, MPI_INT,
                         MANAGER, TAG_SAMPLE_DONE, m_comm);

                // dump solution
                dump_velocity(sampleId, U->GetBlock(0));
            }
            else if (status.MPI_TAG == TAG_DONE)
            { // no more samples, finished
                std::cout << "Worker " << m_myrank
                          << " finished!" << std::endl;
                break;
            }
            else {
                std::cout << "\nError: unknown tag "
                          << status.MPI_TAG
                          << " received by process "
                          << m_myrank << "\n\n";
                break;
            }
        }
    }

    //! Releases allocated memory
    void finalize()
    {
        if (m_myrank == MANAGER) {
            // do nothing
        }
        else {
            if (m_solver) { delete m_solver; }
        }
    }

    //! Writes samples to file
    void dump_samples()
    {
        auto json = nlohmann::json{};
        json["generator_seed"] = m_seed;
        json["num_params"] = m_nparams;
        json["num_samples"] = m_nsamples;
        for (int j=0; j<m_nsamples; j++) {
            for (int i=0; i<m_nparams; i++) {
                json["omegas"]
                        [static_cast<unsigned int>
                        (i + j*m_nparams)]
                            = m_omegas(i, j);
            }
        }

        std::string samples_file
                = m_output_dir+"samples_lx"
                +std::to_string(m_lx)+".json";
        std::cout << "\nSamples file: "
                  << samples_file << "\n\n";
        auto file = std::ofstream(samples_file);
        // always check that you can write to the file.
        assert(file.good());

        file << json.dump(2);
        file.close();
    }

    //! Writes current sample ids sent for computing to file
    void dump_computeSampleIds(Array<int> computeSampleIds)
    {
        if (m_bool_dumpOut)
        {
            auto json = nlohmann::json{};
            json["num_compute_samples"]
                    = computeSampleIds.Size();
            for (int i=0; i<computeSampleIds.Size(); i++) {
                json["compute_samples"]
                        [static_cast<unsigned int>
                        (i)] = computeSampleIds[i];
            }
            std::string computeSampleIds_file
                    = m_output_dir+"computeSampleIds_lx"
                    +std::to_string(m_lx)+".json";
            std::cout << "\nCompute sample ids file: "
                      << computeSampleIds_file << "\n\n";
            auto file = std::ofstream(computeSampleIds_file);
            // always check that you can write to the file.
            assert(file.good());

            file << json.dump(2);
            file.close();
        }
    }

    //! Writes finished sample ids to file
    void dump_finishedSampleIds(Array<int> finishedSampleIds)
    {
        if (m_bool_dumpOut)
        {
            auto json = nlohmann::json{};
            json["num_finished_samples"]
                    = finishedSampleIds.Size();
            for (int i=0; i<finishedSampleIds.Size(); i++) {
                json["finished_samples"]
                        [static_cast<unsigned int>
                        (i)] = finishedSampleIds[i];
            }
            std::string finishedSampleIds_file
                    = m_output_dir+"finishedSampleIds_lx"
                    +std::to_string(m_lx)+".json";
            std::cout << "\nFinished sample ids file: "
                      << finishedSampleIds_file << "\n\n";
            auto file = std::ofstream(finishedSampleIds_file);
            // always check that you can write to the file.
            assert(file.good());

            file << json.dump(2);
            file.close();
        }
    }

    //! Writes mesh to file
    void dump_mesh() const
    {
        if (m_bool_dumpOut)
        {
            std::string mesh_file = m_output_dir
                    +"mesh"+m_meshName_suffix;
            std::cout << "\nMesh file: "
                      << mesh_file << "\n\n";
            std::ofstream mesh_ofs(mesh_file.c_str());
            mesh_ofs.precision(m_precision);
            m_discr->get_mesh()->Print(mesh_ofs);
            mesh_ofs.close();
        }
    }

    //! Writes velocity solution to files
    void dump_velocity(int sampleId,
                       Vector& V) const
    {
        if (m_bool_dumpOut)
        {
            auto fes = m_discr->get_fespaces()[0];
            auto v = std::make_shared<GridFunction>();
            v->MakeRef(fes, V);

            std::string sol_name
                    = m_ensemble_output_dir+"velocity_s"
                    +std::to_string(sampleId);
            std::cout << "Velocity solution file: "
                      << sol_name << std::endl;

            std::ofstream sol_ofs(sol_name.c_str());
            if (!sol_ofs.is_open()) {
                std::cout << "Error while opening "
                             "sol file: "
                          << sol_name << std::endl;
            }
            else {
                sol_ofs.precision(m_precision);
                v->Save(sol_ofs);
                sol_ofs.close();
            }
        }
    }

private:
    // communicator
    MPI_Comm m_comm;
    int m_myrank;
    int m_nworkers;

    // config info
    nlohmann::json m_config;
    std::string m_samplerType;
    std::string m_mesh_dir;
    int m_lx, m_Nt;

    bool m_bool_dumpOut;
    std::string m_meshName_suffix;
    std::string m_solName_suffix;
    std::string m_output_dir;
    std::string m_ensemble_output_dir;
    int m_precision = 15;

    // manager data
    int m_nparams, m_nsamples, m_seed;
    DenseMatrix m_omegas;
    Array<int> m_sampleIds;

    // worker data
    std::shared_ptr<IncompNSTestCases> m_testCase;
    std::shared_ptr<IncompNSFEM> m_discr;
    std::shared_ptr<IncompNSObserver> m_observer;
    IncompNSSolver * m_solver = nullptr;
};


//! Runs Monte Carlo simulations
//! for Incompressible Navier-Stokes
//! using Self-scheduling algorithm
void run_uq_incompNS (const nlohmann::json config,
                      const std::string base_mesh_dir)
{
    MPI_Comm global_comm(MPI_COMM_WORLD);
    SelfScheduler selfScheduler
            (global_comm, config, base_mesh_dir);
    selfScheduler.init();
    selfScheduler.run();
    selfScheduler.finalize();
}


int main(int argc, char *argv[])
{   
    // Initialize MPI.
    MPI_Init(&argc, &argv);

    // Read config json
    auto config = get_global_config(argc, argv);
    const std::string host = config["host"];
    const std::string run = config["run"];
    std::string base_mesh_dir("../meshes/");
    
    if (run == "simulation") {
        run_uq_incompNS(std::move(config),
                        std::move(base_mesh_dir));
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}


// End of file
