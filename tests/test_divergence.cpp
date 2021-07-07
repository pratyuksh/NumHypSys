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
#include "../include/incompNS/observer.hpp"
#include "../include/incompNS/utilities.hpp"
#include "../include/uq/sampler/sampler.hpp"


TEST (IncompNSUtils, measureDivergenceQuadMesh)
{
    // config
    std::string filename
            = "../config_files/unit_tests/"
              "incompNS_svs.json";
    auto config = get_global_config(filename);

    // mesh file
    std::string base_mesh_dir("../meshes/");
    std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;
    const std::string mesh_file
            = mesh_dir+"quad_mesh_l0.mesh";
    const int lx = config["level_x"];

    // test case
    std::shared_ptr<IncompNSTestCases> testCase
        = make_incompNS_test_case(config);

    // mesh
    std::shared_ptr<Mesh> mesh
            = std::make_shared<Mesh>(mesh_file.c_str());
    for (int k=0; k<lx; k++) {
        mesh->UniformRefinement();
    }

    // FE spaces
    int deg = config["deg_x"];
    int ndim = mesh->Dimension();
    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(deg, ndim);
    FiniteElementSpace *R_space
            = new FiniteElementSpace(mesh.get(), hdiv_coll);

    FiniteElementCollection *l2_coll
            = new L2_FECollection(deg, ndim);
    FiniteElementSpace *W_space
            = new FiniteElementSpace(mesh.get(), l2_coll);

    // Divergence operator
    // \int_{\Omega} div(u_h) q_h d_{\Omega}
    MixedBilinearForm *div_form
            = new MixedBilinearForm(R_space,
                                    W_space);
    ConstantCoefficient one(-1.0);
    div_form->AddDomainIntegrator
            (new VectorFEDivergenceIntegrator(one));
    div_form->Assemble();
    div_form->Finalize();
    SparseMatrix *div = div_form->LoseMat();
    delete div_form;

    // velocity
    std::shared_ptr <GridFunction> v
            = std::make_shared<GridFunction>(R_space);
    Sampler<Uniform> unifSampler(10, 1);
    Vector omegas = unifSampler.generate_one_sample();
    IncompNSInitialVelocityCoeff v0_coeff(testCase);
    v0_coeff.SetTime(0);
    v->ProjectCoefficient(v0_coeff);

    double divVal = measure_divergence(div, v.get());

    double TOL=1E-8;
    ASSERT_LE(divVal, TOL);
}

TEST (IncompNSUtils, measureDivergenceTriMesh)
{
    // config
    std::string filename
            = "../config_files/unit_tests/"
              "incompNS_svs.json";
    auto config = get_global_config(filename);

    // mesh file
    std::string base_mesh_dir("../meshes/");
    std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;
    const std::string mesh_file
            = mesh_dir+"tri_mesh_l0.mesh";
    const int lx = config["level_x"];

    // test case
    std::shared_ptr<IncompNSTestCases> testCase
        = make_incompNS_test_case(config);

    // mesh
    std::shared_ptr<Mesh> mesh
            = std::make_shared<Mesh>(mesh_file.c_str());
    for (int k=0; k<lx; k++) {
        mesh->UniformRefinement();
    }

    // FE spaces
    int deg = config["deg_x"];
    int ndim = mesh->Dimension();
    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(deg, ndim);
    FiniteElementSpace *R_space
            = new FiniteElementSpace(mesh.get(), hdiv_coll);

    FiniteElementCollection *l2_coll
            = new L2_FECollection(deg, ndim);
    FiniteElementSpace *W_space
            = new FiniteElementSpace(mesh.get(), l2_coll);

    // Divergence operator
    // \int_{\Omega} div(u_h) q_h d_{\Omega}
    MixedBilinearForm *div_form
            = new MixedBilinearForm(R_space,
                                    W_space);
    ConstantCoefficient one(-1.0);
    div_form->AddDomainIntegrator
            (new VectorFEDivergenceIntegrator(one));
    div_form->Assemble();
    div_form->Finalize();
    SparseMatrix *div = div_form->LoseMat();
    delete div_form;

    // velocity
    std::shared_ptr <GridFunction> v
            = std::make_shared<GridFunction>(R_space);
    Sampler<Uniform> unifSampler(10, 1);
    Vector omegas = unifSampler.generate_one_sample();
    IncompNSInitialVelocityCoeff v0_coeff(testCase);
    v0_coeff.SetTime(0);
    v->ProjectCoefficient(v0_coeff);

    double divVal = measure_divergence(div, v.get());

    double TOL=2E-5;
    ASSERT_LE(divVal, TOL);
}


TEST (IncompNSUtils, divgFreeVelQuadMesh)
{
    // config
    std::string filename
            = "../config_files/unit_tests/"
              "incompNS_svs.json";
    auto config = get_global_config(filename);

    // mesh file
    std::string base_mesh_dir("../meshes/");
    std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;
    const std::string mesh_file
            = mesh_dir+"quad_mesh_l0.mesh";

    // test case
    std::shared_ptr<IncompNSTestCases> testCase
        = make_incompNS_test_case(config);

    // mesh
    const int lx = config["level_x"];
    std::shared_ptr<Mesh> mesh
            = std::make_shared<Mesh>(mesh_file.c_str());
    for (int k=0; k<lx; k++) {
        mesh->UniformRefinement();
    }

    // FE spaces
    int deg = config["deg_x"];
    int ndim = mesh->Dimension();
    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(deg, ndim);
    FiniteElementSpace *R_space
            = new FiniteElementSpace(mesh.get(), hdiv_coll);

    FiniteElementCollection *l2_coll
            = new L2_FECollection(deg, ndim);
    FiniteElementSpace *W_space
            = new FiniteElementSpace(mesh.get(), l2_coll);

    // Divergence operator
    // \int_{\Omega} div(u_h) q_h d_{\Omega}
    MixedBilinearForm *div_form
            = new MixedBilinearForm(R_space,
                                    W_space);
    ConstantCoefficient one(-1.0);
    div_form->AddDomainIntegrator
            (new VectorFEDivergenceIntegrator(one));
    div_form->Assemble();
    div_form->Finalize();
    SparseMatrix *div = div_form->LoseMat();
    delete div_form;

    // velocity
    std::shared_ptr <GridFunction> v
            = std::make_shared<GridFunction>(R_space);
    Sampler<Uniform> unifSampler(10, 1);
    Vector omegas = unifSampler.generate_one_sample();
    IncompNSInitialVelocityCoeff v0_coeff(testCase);
    testCase->set_perturbations(omegas);
    v0_coeff.SetTime(0);
    v->ProjectCoefficient(v0_coeff);

    // oberver
    std::shared_ptr<IncompNSObserver> observer
            = std::make_shared<IncompNSObserver>
            (config, lx);
    //(*observer)(v);

    double div_old = measure_divergence(div, v.get());
    std::cout << "Weak divergence before cleaning: "
              << div_old << std::endl;

    // make divergence free
    DivergenceFreeVelocity divFreeVel (config, mesh);
    divFreeVel (v.get());
    //(*observer)(v);

    double div_new = measure_divergence(div, v.get());
    std::cout << "Weak divergence after cleaning: "
              << div_new << std::endl;

    double TOL=1E-5;
    ASSERT_LE(div_new, TOL);
}


TEST (IncompNSUtils, divgFreeVelTriMesh)
{
    // config
    std::string filename
            = "../config_files/unit_tests/"
              "incompNS_svs.json";
    auto config = get_global_config(filename);

    // mesh file
    std::string base_mesh_dir("../meshes/");
    std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;
    const std::string mesh_file
            = mesh_dir+"tri_mesh_l0.mesh";

    // test case
    std::shared_ptr<IncompNSTestCases> testCase
        = make_incompNS_test_case(config);

    // mesh
    const int lx = config["level_x"];
    std::shared_ptr<Mesh> mesh
            = std::make_shared<Mesh>(mesh_file.c_str());
    for (int k=0; k<lx; k++) {
        mesh->UniformRefinement();
    }

    // FE spaces
    int deg = config["deg_x"];
    int ndim = mesh->Dimension();
    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(deg, ndim);
    FiniteElementSpace *R_space
            = new FiniteElementSpace(mesh.get(), hdiv_coll);

    FiniteElementCollection *l2_coll
            = new L2_FECollection(deg, ndim);
    FiniteElementSpace *W_space
            = new FiniteElementSpace(mesh.get(), l2_coll);

    // Divergence operator
    // \int_{\Omega} div(u_h) q_h d_{\Omega}
    MixedBilinearForm *div_form
            = new MixedBilinearForm(R_space,
                                    W_space);
    ConstantCoefficient one(-1.0);
    div_form->AddDomainIntegrator
            (new VectorFEDivergenceIntegrator(one));
    div_form->Assemble();
    div_form->Finalize();
    SparseMatrix *div = div_form->LoseMat();
    delete div_form;

    // velocity
    std::shared_ptr <GridFunction> v
            = std::make_shared<GridFunction>(R_space);
    Sampler<Uniform> unifSampler(10, 1);
    Vector omegas = unifSampler.generate_one_sample();
    IncompNSInitialVelocityCoeff v0_coeff(testCase);
    testCase->set_perturbations(omegas);
    v0_coeff.SetTime(0);
    v->ProjectCoefficient(v0_coeff);

    // oberver
    std::shared_ptr<IncompNSObserver> observer
            = std::make_shared<IncompNSObserver>
            (config, lx);
    //(*observer)(v);

    double div_old = measure_divergence(div, v.get());
    std::cout << "Weak divergence before cleaning: "
              << div_old << std::endl;

    // make divergence free
    DivergenceFreeVelocity divFreeVel (config, mesh);
    divFreeVel (v.get());
    //(*observer)(v);

    double div_new = measure_divergence(div, v.get());
    std::cout << "Weak divergence after cleaning: "
              << div_new << std::endl;

    double TOL=1E-4;
    ASSERT_LE(div_new, TOL);
}


// End of file
