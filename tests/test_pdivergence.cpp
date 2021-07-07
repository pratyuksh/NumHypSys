#include <gtest/gtest.h>

#include "mfem.hpp"
using namespace mfem;

#include <iostream>
#include <fstream>
#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../include/core/config.hpp"
#include "../include/stokes/assembly.hpp"
#include "../include/incompNS/test_cases_factory.hpp"
#include "../include/incompNS/coefficients.hpp"
#include "../include/incompNS/pobserver.hpp"
#include "../include/incompNS/putilities.hpp"
#include "../include/uq/scheduler/communicator.hpp"
#include "../include/uq/scheduler/scheduler.hpp"
#include "../include/uq/sampler/sampler.hpp"


TEST (IncompNSUtils, parDivgFreeVelQuadMesh)
{
    int nprocs, myrank;
    MPI_Comm global_comm = MPI_COMM_WORLD;
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    // config
    std::string filename
            = "../config_files/unit_tests/"
              "pincompNS_svs.json";
    auto config = get_global_config(filename);

    // mesh file
    std::string base_mesh_dir("../meshes/");
    std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;
    const std::string mesh_file
            = mesh_dir+"quad_mesh_l0.mesh";

    const int nparams = 10;
    const int nsamples = 3;
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

        // test case
        std::shared_ptr<IncompNSTestCases> testCase
            = make_incompNS_test_case(config);

        // mesh
        const int lx = config["level_x"];
        Mesh *mesh = new Mesh(mesh_file.c_str());
        for (int k=0; k<lx; k++) {
            mesh->UniformRefinement();
        }
        std::shared_ptr<ParMesh> pmesh
                = std::make_shared<ParMesh>
                (intra_group_comm, *mesh);
        delete mesh;

        // FE spaces
        int deg = config["deg_x"];
        int ndim = pmesh->Dimension();
        FiniteElementCollection *hdiv_coll
                = new RT_FECollection(deg, ndim);
        ParFiniteElementSpace *R_space
                = new ParFiniteElementSpace(pmesh.get(),
                                            hdiv_coll);

        FiniteElementCollection *l2_coll
                = new L2_FECollection(deg, ndim);
        ParFiniteElementSpace *W_space
                = new ParFiniteElementSpace(pmesh.get(),
                                            l2_coll);

        // Divergence operator
        // \int_{\Omega} div(u_h) q_h d_{\Omega}
        ParMixedBilinearForm *div_form
                = new ParMixedBilinearForm(R_space,
                                           W_space);
        ConstantCoefficient one(-1.0);
        div_form->AddDomainIntegrator
                (new VectorFEDivergenceIntegrator(one));
        div_form->Assemble();
        div_form->Finalize();
        HypreParMatrix *div = div_form->ParallelAssemble();
        delete div_form;

        // observer
        std::shared_ptr<IncompNSParObserver> observer
                = std::make_shared<IncompNSParObserver>
                (intra_group_comm, config, lx);

        // velocity
        std::shared_ptr <ParGridFunction> v
                = std::make_shared<ParGridFunction>(R_space);
        Vector omegas;
        all_myomegas.GetColumn(0, omegas);
        IncompNSInitialVelocityCoeff v0_coeff(testCase);
        testCase->set_perturbations(omegas);
        v0_coeff.SetTime(0);
        v->ProjectCoefficient(v0_coeff);
        //(*observer) (v);

        double div_old = measure_divergence(div, v.get());
        std::cout << "My group: " << mygroup
                  << "\tMy group rank: " << mygrouprank
                  << "\tWeak divergence before cleaning: "
                  << div_old << std::endl;

        // make divergence free
        ParDivergenceFreeVelocity divFreeVel (config, pmesh);
        divFreeVel (v.get());

        double div_new = measure_divergence(div, v.get());
        std::cout << "My group: " << mygroup
                  << "\tMy group rank: " << mygrouprank
                  << "\tWeak divergence after cleaning: "
                  << div_new << std::endl;

        double TOL=1E-5;
        ASSERT_LE(div_new, TOL);
    }
}

TEST (IncompNSUtils, parDivgFreeVelTriMesh)
{
    int nprocs, myrank;
    MPI_Comm global_comm = MPI_COMM_WORLD;
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    // config
    std::string filename
            = "../config_files/unit_tests/"
              "pincompNS_svs.json";
    auto config = get_global_config(filename);

    // mesh file
    std::string base_mesh_dir("../meshes/");
    std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;
    const std::string mesh_file
            = mesh_dir+"tri_mesh_l0.mesh";

    const int nparams = 10;
    const int nsamples = 3;
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

        // test case
        std::shared_ptr<IncompNSTestCases> testCase
            = make_incompNS_test_case(config);

        // mesh
        const int lx = config["level_x"];
        Mesh *mesh = new Mesh(mesh_file.c_str());
        for (int k=0; k<lx; k++) {
            mesh->UniformRefinement();
        }
        std::shared_ptr<ParMesh> pmesh
                = std::make_shared<ParMesh>
                (intra_group_comm, *mesh);
        delete mesh;

        // FE spaces
        int deg = config["deg_x"];
        int ndim = pmesh->Dimension();
        FiniteElementCollection *hdiv_coll
                = new RT_FECollection(deg, ndim);
        ParFiniteElementSpace *R_space
                = new ParFiniteElementSpace(pmesh.get(),
                                            hdiv_coll);

        FiniteElementCollection *l2_coll
                = new L2_FECollection(deg, ndim);
        ParFiniteElementSpace *W_space
                = new ParFiniteElementSpace(pmesh.get(),
                                            l2_coll);

        // Divergence operator
        // \int_{\Omega} div(u_h) q_h d_{\Omega}
        ParMixedBilinearForm *div_form
                = new ParMixedBilinearForm(R_space,
                                           W_space);
        ConstantCoefficient one(-1.0);
        div_form->AddDomainIntegrator
                (new VectorFEDivergenceIntegrator(one));
        div_form->Assemble();
        div_form->Finalize();
        HypreParMatrix *div = div_form->ParallelAssemble();
        delete div_form;

        // observer
        std::shared_ptr<IncompNSParObserver> observer
                = std::make_shared<IncompNSParObserver>
                (intra_group_comm, config, lx);

        // velocity
        std::shared_ptr <ParGridFunction> v
                = std::make_shared<ParGridFunction>(R_space);
        Vector omegas;
        all_myomegas.GetColumn(0, omegas);
        IncompNSInitialVelocityCoeff v0_coeff(testCase);
        testCase->set_perturbations(omegas);
        v0_coeff.SetTime(0);
        v->ProjectCoefficient(v0_coeff);
        //(*observer) (v);

        double div_old = measure_divergence(div, v.get());
        std::cout << "My group: " << mygroup
                  << "\tMy group rank: " << mygrouprank
                  << "\tWeak divergence before cleaning: "
                  << div_old << std::endl;

        // make divergence free
        ParDivergenceFreeVelocity divFreeVel (config, pmesh);
        divFreeVel (v.get());

        double div_new = measure_divergence(div, v.get());
        std::cout << "My group: " << mygroup
                  << "\tMy group rank: " << mygrouprank
                  << "\tWeak divergence after cleaning: "
                  << div_new << std::endl;

        double TOL=1E-4;
        ASSERT_LE(div_new, TOL);
    }
}
