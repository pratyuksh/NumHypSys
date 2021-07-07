#include <gtest/gtest.h>

#include <iostream>
#include "assert.h"

#include "../include/incompNS/solver.hpp"


TEST(Cfl, computeCfl)
{
    std::string filename
            = "../config_files/unit_tests/"
              "incompNS_tgv_test1.json";
    auto config = get_global_config(filename);
    std::string base_mesh_dir("../meshes/");

    const int lx = config["level_x"];
    assert(lx == 6);

    const std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;

    const int Nt = config["num_time_steps"];

    auto testCase = make_incompNS_test_case(config);
    IncompNSBackwardEulerSolver incompNS_solver
            (config, testCase, mesh_dir, lx, Nt);

    auto mesh = incompNS_solver.get_mesh();
    auto discr = incompNS_solver.get_discr();
    auto obs = incompNS_solver.get_observer();
    discr->set(mesh);

    auto fes = discr->get_fespaces();
    auto v = std::make_shared<GridFunction>(fes[0]);
    IncompNSInitialVelocityCoeff v0_coeff(testCase);
    v0_coeff.SetTime(0);
    v->ProjectCoefficient(v0_coeff);

    // test CFL condition
    int deg = config["deg_x"];
    double cfl_number = config["cfl_number"];
    double hMin = incompNS_solver.get_hMin();
    double true_cfl_dt = cfl_number*hMin/(deg*deg);
    std::cout << "True CFL time step: "
              << true_cfl_dt << std::endl;

    double cfl_dt = incompNS_solver.compute_time_step(v);
    std::cout << "CFL time step: "
              << cfl_dt << std::endl;

    ASSERT_LE(cfl_dt, true_cfl_dt);
}



// End of file
