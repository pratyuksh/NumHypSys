#include "../../include/core/utilities.hpp"

namespace fs = std::filesystem;


//! Writes convergence results to json file
void write_convg_json_file (std::string system_type,
                            const nlohmann::json& config,
                            Eigen::VectorXd &data_x,
                            Eigen::MatrixXd &data_y)
{
    std::string base_out_dir = config["output_dir"];
    std::string out_dir = base_out_dir+"/"+system_type;
    fs::create_directories(out_dir);

    int deg = config["deg_x"];
    std::string base_name = config["problem_type"];
    std::string mesh_elem_type = "quad";
    if (config.contains("mesh_elem_type")) {
        mesh_elem_type = config["mesh_elem_type"];
    }

    // set output file names depending on system types
    std::string outfile;
    if (system_type == "darcy"
            || system_type == "darcy_mpi"
            || system_type == "stokes"
            || system_type == "stokes_mpi")
    {
        outfile = out_dir+"/convergence_"
                +mesh_elem_type+"Mesh_"+
                base_name+"_deg"+std::to_string(deg)+".json";
    }
    else if (system_type == "incompNS"
             || system_type == "incompNS_mpi")
    {
        std::string flux_type = "upwind";
        if (config.contains("numerical_flux")) {
            flux_type = config["numerical_flux"];
        }

        outfile = out_dir+"/convergence_"
                +mesh_elem_type+"Mesh_"+
                flux_type+"_"+base_name+"_deg"
                +std::to_string(deg)+".json";
    }

    // set data names
    std::string data_x_name;
    std::vector<std::string> data_y_names;
    if (system_type == "darcy"
            || system_type == "darcy_mpi"
            || system_type == "stokes"
            || system_type == "stokes_mpi"
            || system_type == "incompNS"
            || system_type == "incompNS_mpi")
    {
        data_x_name = "h_max";
        data_y_names.push_back("velocity");
        data_y_names.push_back("pressure");
    }

    auto json = nlohmann::json{};
    for (unsigned int i=0; i<data_x.size(); i++) {
        json[data_x_name][i] = data_x(i);
        for (unsigned int j=0; j<data_y.rows(); j++) {
            json[data_y_names[j]][i] = data_y(j,i);
        }
    }

    auto file = std::ofstream(outfile);
    assert(file.good());
    // always check that you can write to the file.

    file << json.dump(2);
}


// End of file
