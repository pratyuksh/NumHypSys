#include <gtest/gtest.h>

#include <iostream>
#include "assert.h"

#include "../include/includes.hpp"
#include "../include/mymfem/cell_avgs.hpp"
#include "../include/mymfem/pcell_avgs.hpp"
#include "../include/uq/stats/stats.hpp"
#include "../include/uq/scheduler/communicator.hpp"
#include "../include/uq/scheduler/scheduler.hpp"
#include "../include/uq/sampler/sampler.hpp"
#include "../include/incompNS/solver.hpp"
#include "../include/incompNS/psolver.hpp"

using namespace std;


class TestUqModel1
{
public:
    TestUqModel1(const nlohmann::json& config,
                 std::string base_mesh_dir)
    {
        std::string sub_mesh_dir = config["mesh_dir"];
        const std::string mesh_dir
                = base_mesh_dir+sub_mesh_dir;

        const int lx = config["level_x"];
        const int Nt = config["num_time_steps"];

        m_testCase = make_incompNS_test_case(config);

        m_incompNS_solver = new IncompNSBackwardEulerSolver
                (config, m_testCase, mesh_dir, lx, Nt);
        m_mesh = m_incompNS_solver->get_mesh();
        m_discr = m_incompNS_solver->get_discr();
        m_observer = m_incompNS_solver->get_observer();

        m_discr->set(m_mesh);
        m_block_offsets = m_discr->get_block_offsets();
        m_observer->init(m_discr);

        // cell averages
        m_cellAvgs = new CellAverages
                (m_discr->get_fespaces()[0]->GetMesh());
        m_sfes0 = m_cellAvgs->get_sfes();
    }

    ~ TestUqModel1 () {
        delete m_incompNS_solver;
        delete m_cellAvgs;
    }

    void operator() (const Vector& omegas,
                     Vector& vx, Vector& vy)
    {
        m_testCase->set_perturbations(omegas);

        std::unique_ptr<BlockVector> U
                = std::make_unique<BlockVector>
                  (m_block_offsets);

        Array<FiniteElementSpace*> fespaces
                = m_discr->get_fespaces();

        std::shared_ptr <GridFunction> v
                = std::make_shared<GridFunction>();
        std::shared_ptr <GridFunction> p
                = std::make_shared<GridFunction>();

        v->MakeRef(fespaces[0], U->GetBlock(0));
        p->MakeRef(fespaces[1], U->GetBlock(1));

        // initial conditions
        IncompNSInitialVelocityCoeff v0_coeff(m_testCase);
        IncompNSInitialPressureCoeff p0_coeff(m_testCase);
        v0_coeff.SetTime(0);
        p0_coeff.SetTime(0);
        v->ProjectCoefficient(v0_coeff);
        p->ProjectCoefficient(p0_coeff);
        //m_observer->visualize_velocities(v);

        // compute cell averages
        vx.SetSize(m_sfes0->GetTrueVSize());
        vy.SetSize(m_sfes0->GetTrueVSize());
        std::tie (vx, vy) = m_cellAvgs->eval(v.get());
    }

    void visualize_velocity(Vector& V)
    {
        std::shared_ptr <GridFunction> v
                = std::make_shared<GridFunction>();

        v->MakeRef(m_sfes0, V);
        (*m_observer)(v);
    }

private:
    std::shared_ptr<IncompNSTestCases> m_testCase;
    std::shared_ptr<Mesh> m_mesh;
    std::shared_ptr<IncompNSFEM> m_discr;
    std::shared_ptr<IncompNSObserver> m_observer;

    IncompNSSolver * m_incompNS_solver = nullptr;
    Array<int> m_block_offsets;

    CellAverages * m_cellAvgs = nullptr;
    FiniteElementSpace *m_sfes0 = nullptr;
};

TEST(Statistics, stats1)
{
    int nprocs, myrank;
    MPI_Comm global_comm = MPI_COMM_WORLD;
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    // uq test model
    std::string filename
            = "../config_files/unit_tests/"
              "test1_stats_incompNS_uqTestModel.json";
    auto config = get_global_config(filename);
    std::string base_mesh_dir("../meshes/");
    TestUqModel1 testUqModel(config, base_mesh_dir);

    const int nparams = config["uq_num_params"];
    const int nsamples = config["uq_num_samples"];
    if (nprocs == 6)
    {
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

        Statistics *stats = nullptr;
        Vector vx_mean, vx_variance;
        for (int k=0; k<mynsamples; k++)
        {
            Vector omegas(nparams);
            all_myomegas.GetColumn(k, omegas);

            Vector vx, vy;
            testUqModel(omegas, vx, vy);

            if (k == 0) {
                stats = new Statistics(global_comm,
                                       nsamples,
                                       vx.Size());
                if (myrank == IamRoot) {
                    vx_mean.SetSize(vx.Size());
                    vx_variance.SetSize(vx.Size());
                }
            }
            stats->add_to_first_moment(vx);
            stats->add_squared_to_second_moment(vx);
        }
        std::cout << "Processor " << myrank
                  << " finished running "
                  << mynsamples << " samples." << std::endl;

        // compute mean and variance
        stats->mean(vx_mean);
        stats->variance(vx_mean, vx_variance);

        // visualize
        if (myrank == IamRoot) {
            //testUqModel.visualize_velocity(vx_mean);
            //testUqModel.visualize_velocity(vx_variance);
            double TOL = 1e-2;
            for (int k=0; k<vx_variance.Size(); k++) {
                ASSERT_LE(std::fabs(vx_variance(k) - 0.5),
                          TOL);
            }
        }
    }
}


class TestUqModel2
{
public:
    TestUqModel2(MPI_Comm& comm,
                 const nlohmann::json& config,
                 std::string base_mesh_dir)
        : m_comm (comm)
    {
        MPI_Comm_rank(m_comm, &m_myrank);
        MPI_Comm_size(m_comm, &m_nprocs);

        std::string sub_mesh_dir = config["mesh_dir"];
        const std::string mesh_dir
                = base_mesh_dir+sub_mesh_dir;

        const int lx = config["level_x"];
        const int Nt = config["num_time_steps"];

        m_testCase = make_incompNS_test_case(config);

        IncompNSBackwardEulerParSolver incompNS_solver
                (comm, config, m_testCase,
                 mesh_dir, lx, Nt);
        m_pmesh = incompNS_solver.get_mesh();
        m_discr = incompNS_solver.get_discr();
        m_observer = incompNS_solver.get_observer();

        m_discr->set(m_pmesh);
        m_block_offsets = m_discr->get_block_offsets();
        m_block_trueOffsets
                = m_discr->get_block_trueOffsets();

        // cell averages
        m_cellAvgs = new ParCellAverages
                (m_discr->get_fespaces()[0]->GetParMesh());
        m_sfes0 = m_cellAvgs->get_sfes();
    }

    ~ TestUqModel2 () {
        if (m_cellAvgs) { delete m_cellAvgs; }
    }

    void operator() (const Vector& omegas,
                     Vector& vx, Vector& vy)
    {
        m_testCase->set_perturbations(omegas);

        Array<ParFiniteElementSpace*> fespaces
                = m_discr->get_fespaces();

        std::unique_ptr <BlockVector> U_buf
                = std::make_unique<BlockVector>
                (m_block_offsets);
        std::shared_ptr <ParGridFunction> v
                = std::make_shared<ParGridFunction>
                (fespaces[0],
                U_buf->GetData());
        std::shared_ptr <ParGridFunction> p
                = std::make_shared<ParGridFunction>
                (fespaces[1],
                U_buf->GetData() + m_block_offsets[1]);

        // initial conditions
        IncompNSInitialVelocityCoeff v0_coeff(m_testCase);
        IncompNSInitialPressureCoeff p0_coeff(m_testCase);
        v0_coeff.SetTime(0);
        p0_coeff.SetTime(0);
        v->ProjectCoefficient(v0_coeff);
        p->ProjectCoefficient(p0_coeff);
        //(*m_observer)(v);

        // compute cell averages
        vx.SetSize(m_sfes0->GetTrueVSize());
        vy.SetSize(m_sfes0->GetTrueVSize());
        std::tie (vx, vy) = m_cellAvgs->eval(v.get());
    }

    void visualize_velocity(Vector& V)
    {
        std::shared_ptr <ParGridFunction> v
                = std::make_shared<ParGridFunction>();

        v->MakeRef(m_sfes0, V);
        (*m_observer)(v);
    }

private:
    MPI_Comm m_comm;
    int m_nprocs, m_myrank;

    std::shared_ptr<IncompNSTestCases> m_testCase;
    std::shared_ptr<ParMesh> m_pmesh;
    std::shared_ptr<IncompNSParFEM> m_discr;
    std::shared_ptr<IncompNSParObserver> m_observer;

    Array<int> m_block_offsets;
    Array<int> m_block_trueOffsets;

    ParCellAverages * m_cellAvgs = nullptr;
    ParFiniteElementSpace *m_sfes0 = nullptr;
};

TEST(Statistics, stats2)
{
    int nprocs, myrank;
    MPI_Comm global_comm = MPI_COMM_WORLD;
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    std::string filename
            = "../config_files/unit_tests/"
              "test2_stats_incompNS_uqTestModel.json";
    auto config = get_global_config(filename);
    std::string base_mesh_dir("../meshes/");

    const int nparams = config["uq_num_params"];
    const int nsamples = config["uq_num_samples"];
    if (nprocs == 6)
    {
        int nprocsPerGroup = 2;

        // make communicators
        MPI_Comm intra_group_comm, inter_group_comm;
        std::tie (intra_group_comm, inter_group_comm)
                = make_communicators(global_comm,
                                     nprocsPerGroup);

        int mygroup, mygrouprank;
        MPI_Comm_rank(inter_group_comm, &mygroup);
        MPI_Comm_rank(intra_group_comm, &mygrouprank);

        // uq test model
        TestUqModel2 testUqModel(intra_group_comm,
                                 config, base_mesh_dir);

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

        Statistics *stats = nullptr;
        Vector vx_mean, vx_variance;
        for (int k=0; k<mynsamples; k++)
        {
            Vector omegas(nparams);
            all_myomegas.GetColumn(k, omegas);

            Vector vx, vy;
            testUqModel(omegas, vx, vy);

            if (k == 0) {
                stats = new Statistics(inter_group_comm,
                                       nsamples,
                                       vx.Size());
                if (mygroup == IamRoot) {
                    vx_mean.SetSize(vx.Size());
                    vx_variance.SetSize(vx.Size());
                }
            }
            stats->add_to_first_moment(vx);
            stats->add_squared_to_second_moment(vx);
        }
        std::cout << "Group " << mygroup
                  << " , processor " << mygrouprank
                  << " finished running "
                  << mynsamples << " samples." << std::endl;

        // compute mean and variance
        stats->mean(vx_mean);
        stats->variance(vx_mean, vx_variance);

        // visualize
        if (mygroup == IamRoot) {
            //testUqModel.visualize_velocity(vx_mean);
            //testUqModel.visualize_velocity(vx_variance);
            double TOL = 1e-2;
            for (int k=0; k<vx_variance.Size(); k++) {
                ASSERT_LE(std::fabs(vx_variance(k) - 0.5),
                          TOL);
            }
        }
    }
}


// End of file
