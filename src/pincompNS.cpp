#include "../include/includes.hpp"
#include "../include/core/config.hpp"
#include "../include/core/utilities.hpp"
#include "../include/incompNS/psolver_factory.hpp"

#include <iostream>
#include <chrono>
using namespace std::chrono;


void run_incompNS (const nlohmann::json config,
                   const std::string base_mesh_dir)
{
    int myrank;
    MPI_Comm comm(MPI_COMM_WORLD);
    MPI_Comm_rank(comm, &myrank);

    const int lx = config["level_x"];
    std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;
    
    const int Nt = config["num_time_steps"];
    std::string time_integrator = "backward_euler";
    if (config.contains("time_integrator")) {
        time_integrator = config["time_integrator"];
    }

    auto testCase = make_incompNS_test_case(config);
    auto solver = make_solver
            (comm, config, testCase, mesh_dir, lx, Nt);

    auto [h_max, errL2] = (*solver)();
    delete solver;

    if (myrank == 0) {
        std::cout << "\n\nL2 error: "
                  << errL2.transpose() << std::endl;
    }
}


void run_incompNS_convergence
(const nlohmann::json config,
 const std::string base_mesh_dir)
{
    int myrank;
    MPI_Comm comm(MPI_COMM_WORLD);
    MPI_Comm_rank(comm, &myrank);

    const int Nt = config["num_time_steps"];

    const int Lx0 = config["min_level_x"];
    const int Lx = config["max_level_x"];
    std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;
    
    int num_levels = Lx-Lx0+1;
    Eigen::VectorXd h_max(num_levels);
    Eigen::MatrixXd errL2(2,num_levels);
    errL2.setZero();
    
    auto testCase = make_incompNS_test_case(config);
    IncompNSParSolver *solver = nullptr;
    
    for (int k=0; k<num_levels; k++)
    {   
        int lx = Lx0+k;
        solver = make_solver
                (comm, config, testCase, mesh_dir, lx, Nt);
        auto [h_max_, errL2_] = (*solver)();
        h_max(k) = h_max_;  errL2.col(k) = errL2_;
        if (myrank == 0) {
            Eigen::VectorXd errL2_k = errL2.col(k);
            std::cout << "Level: " << lx
                      << ", L2 error: "
                      << errL2_k.transpose()
                      << std::endl;
        }
        delete solver;
    }
    
    if (myrank == 0) {
        std::cout << "\n\nL2 error:\n"
                  << errL2 << std::endl;

        // write convergence results to json file
        write_convg_json_file("incompNS_mpi", config,
                              h_max, errL2);
    }
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
        run_incompNS(std::move(config),
                     std::move(base_mesh_dir));
    }
    else if (run == "xConvergence_test") {
        run_incompNS_convergence(std::move(config),
                                 std::move(base_mesh_dir));
    }
    
    // Finalize MPI
    MPI_Finalize();

    return 0;
}


// End of file
