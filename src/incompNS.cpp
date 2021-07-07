#include "../include/includes.hpp"
#include "../include/core/config.hpp"
#include "../include/core/utilities.hpp"
#include "../include/mymfem/utilities.hpp"
#include "../include/incompNS/solver_factory.hpp"

#include <iostream>


void run_incompNS (const nlohmann::json config,
                   const std::string base_mesh_dir)
{
    const int lx = config["level_x"];
    const std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;

    const int Nt = config["num_time_steps"];

    auto testCase = make_incompNS_test_case(config);

    auto solver = make_solver
            (config, testCase, mesh_dir, lx, Nt);
    
    auto [h_max, errL2] = (*solver)();
    delete solver;
    std::cout << "\n\nL2 error: " << errL2.transpose()
              << std::endl;
}


void run_incompNS_convergence
(const nlohmann::json config,
 const std::string base_mesh_dir)
{
    const int Nt = config["num_time_steps"];

    const int Lx0 = config["min_level_x"];
    const int Lx = config["max_level_x"];
    const std::string sub_mesh_dir = config["mesh_dir"];
    const std::string mesh_dir = base_mesh_dir+sub_mesh_dir;
    
    int num_levels = Lx-Lx0+1;
    Eigen::VectorXd h_max(num_levels);
    Eigen::MatrixXd errL2(2,num_levels);
    errL2.setZero();
    
    auto testCase = make_incompNS_test_case(config);
    IncompNSSolver *solver = nullptr;
    
    for (int k=0; k<num_levels; k++)
    {
        int lx = Lx0+k;
        solver = make_solver
                (config, testCase, mesh_dir, lx, Nt);
        auto [h_max_, errL2_] = (*solver)();
        h_max(k) = h_max_;  errL2.col(k) = errL2_;
        Eigen::VectorXd errL2_k = errL2.col(k);
        std::cout << "Level: " << lx
                  << ", L2 error: " << errL2_k.transpose()
                  << std::endl;
        delete solver;
    }
    std::cout << "\n\nL2 error:\n" << errL2 << std::endl;

    // write convergence results to json file
    write_convg_json_file("incompNS", config, h_max, errL2);
}


//! Generating a boundary refined mesh
//! with refinement towards boundary walls
void prepare_cf_mesh(const std::string base_mesh_dir) {

    const std::string mesh_file
            = base_mesh_dir+"channel/tri_mesh_l0.mesh";

    double threshold_hMax = 0.2;
    double threshold_hMin = 0.002;

    double h_min, h_max, kappa_min, kappa_max;
    auto mesh = std::make_shared<Mesh>(mesh_file.c_str());
    mesh->GetCharacteristics(h_min, h_max,
                             kappa_min, kappa_max);

    // Uniform refinement to size 0.1
    while(h_max > threshold_hMax) {
        mesh->UniformRefinement();
        mesh->GetCharacteristics(h_min, h_max,
                                 kappa_min, kappa_max);
    }

    // define mesh size function
    auto meshSizeFn = [threshold_hMax, threshold_hMin]
            (double y) {
        double a = (threshold_hMax - threshold_hMin)/0.2;
        double b = threshold_hMax - a*(0.21);
        return a*y +b;
    };

    // refine elements wrt bottom and top walls
    Array<int> marked_elements;
    for (double y=0.21; y>0; y-=0.01)
    {
        double threshold = meshSizeFn(y);
        std::cout << "\nDistance: " << y << std::endl;
        std::cout << "Size threshold: "
                  << threshold << std::endl;

        Vector center(2);
        for(int i=0; i<mesh->GetNE(); i++)
        {
            double elSize = mesh->GetElementSize(i);

            get_element_center(mesh, i, center);
            double yDist_bottom = center(1);
            double yDist_top = 0.5-center(1);

            if (yDist_bottom <= y) {
                if (elSize > threshold) {
                    marked_elements.Append(i);
                }
            }
            if (yDist_top <= y) {
                if (elSize > threshold) {
                    marked_elements.Append(i);
                }
            }
        }
        std::cout << "Number of marked elements: "
                  << marked_elements.Size() << std::endl;

        while(h_min > threshold) {
            mesh->GeneralRefinement(marked_elements);
            mesh->GetCharacteristics(h_min, h_max,
                                     kappa_min, kappa_max);
        }

        marked_elements.DeleteAll();
    }
    std::cout << "\n\nRefined mesh is conforming: "
              << mesh->Conforming() << std::endl;
    std::cout << "h_min: " << h_min << ",\t"
              << "h_max: " << h_max << std::endl;

    const std::string out_mesh_file
            = base_mesh_dir
            +"channel/refined/tri_mesh_l0.mesh";
    std::streambuf * buf;
    std::ofstream of;
    of.open(out_mesh_file.c_str());
    buf = of.rdbuf();
    std::ostream out(buf);

    mesh->Print(out);
}


int main(int argc, char *argv[])
{   
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

    //prepare_cf_mesh(base_mesh_dir);

    return 0;
}


// End of file
