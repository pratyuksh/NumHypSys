#include "../include/includes.hpp"
#include "../include/core/config.hpp"
#include "../include/stokes/solver.hpp"

#include <iostream>
#include "../include/core/utilities.hpp"


void run_stokes (const nlohmann::json& config,
                 std::string base_mesh_dir)
{
    const int lx = config["level_x"];
    std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;
    
    StokesSolver stokes_solver(config, mesh_dir, lx);
    
    auto [h_max, errL2] = stokes_solver();
    std::cout << "\n\nL2 error: " << errL2.transpose()
            << std::endl;
}


void run_stokes_convergence (const nlohmann::json& config,
                              std::string base_mesh_dir)
{
    const int Lx0 = config["min_level_x"];
    const int Lx = config["max_level_x"];
    std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;
    
    int num_levels = Lx-Lx0+1;
    Eigen::VectorXd h_max(num_levels);
    Eigen::MatrixXd errL2(2,num_levels);
    errL2.setZero();
    
    StokesSolver *stokes_solver = nullptr;
    
    for (int k=0; k<num_levels; k++)
    {   
        int lx = Lx0+k;
        stokes_solver
                = new StokesSolver(config, mesh_dir, lx);
        auto [h_max_, errL2_] = (*stokes_solver)();
        h_max(k) = h_max_;  errL2.col(k) = errL2_;
        Eigen::VectorXd errL2_k = errL2.col(k);
        std::cout << "Level: " << lx
                  << ", L2 error: " << errL2_k.transpose()
                  << std::endl;
        delete stokes_solver;
    }
    std::cout << "\n\nL2 error:\n" << errL2 << std::endl;

    // write convergence results to json file
    write_convg_json_file("stokes", config, h_max, errL2);
}

int main(int argc, char *argv[])
{   
    // Read config json
    auto config = get_global_config(argc, argv);
    const std::string host = config["host"];
    const std::string run = config["run"];
    std::string base_mesh_dir("../meshes/");
    
    if (run == "simulation") {
        run_stokes(config, base_mesh_dir);
    }
    else if (run == "xConvergence_test") {
        run_stokes_convergence(config, base_mesh_dir);
    }

    return 0;
}


// End of file
