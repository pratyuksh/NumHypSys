#include "../include/includes.hpp"
#include "../include/core/config.hpp"
#include "../include/core/utilities.hpp"
#include "../include/mymfem/utilities.hpp"

#include "../include/uq/stats/structure_base.hpp"
#include "../include/uq/scheduler/scheduler.hpp"
#include "../include/uq/sampler/sampler.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;
using namespace std::chrono;


void dump_structure_to_json
(std::string out_dir,
 std::string sname,
 unsigned int lx,
 Eigen::VectorXd &offsets,
 Eigen::MatrixXd &sval)
{
    assert(offsets.size() == sval.rows());

    auto json = nlohmann::json{};
    for (unsigned int i=0; i<offsets.size(); i++) {
        json["offsets"][i] = offsets(i);
    }
    for (unsigned int j=0; j<sval.cols(); j++) {
        std::string colName = "p"+std::to_string(j+1);
        for (unsigned int i=0; i<sval.rows(); i++) {
            json[colName][i] = sval(i,j);
        }
    }

    fs::create_directories(out_dir);
    std::string outfile = out_dir+sname+"_lx"
            +std::to_string(lx)+".json";
    auto file = std::ofstream(outfile);
    assert(file.good());
    std::cout << outfile << std::endl;

    file << json.dump(2);
    file.close();
}

// prepares boundary data for computing structure functions
void prepare_bdry_data
(const nlohmann::json& config, Eigen::MatrixXd& U,
 unsigned int Nx, unsigned int Ny, unsigned int max_stencil)
{
    std::string problem_type = config["problem_type"];

    if (problem_type == "uq_svs") //smooth vortex-sheet
    {
        // bottom and top boundary stencils
        for (unsigned int j=0; j<max_stencil; j++)
        {
            for (unsigned int i=max_stencil;
                 i<Nx+max_stencil; i++)
            {
                // bottom
                int jb_old = int(j);
                int jb_new = int(j+Ny);
                U(2*i, jb_old) = U(2*i,jb_new);
                U(2*i+1, jb_old) = U(2*i+1,jb_new);

                // top
                int jt_old = int(j+Ny+max_stencil);
                int jt_new = int(j+max_stencil);
                U(2*i, jt_old) = U(2*i,jt_new);
                U(2*i+1, jt_old) = U(2*i+1,jt_new);
            }
        }

        // left and right boundary stencils
        for (unsigned int j=max_stencil;
             j<Ny+max_stencil; j++)
        {
            for (unsigned int i=0; i<max_stencil; i++)
            {
                // left
                int il_old = int(i);
                int il_new = int(i+Nx);
                U(2*il_old, j) = U(2*il_new,j);
                U(2*il_old+1, j) = U(2*il_new+1,j);

                // right
                int ir_old = int(i+Nx+max_stencil);
                int ir_new = int(i+max_stencil);
                U(2*ir_old, j) = U(2*ir_new,j);
                U(2*ir_old+1, j) = U(2*ir_new+1,j);
            }
        }

        for (unsigned int j=0; j<max_stencil; j++)
        {
            for (unsigned int i=0; i<max_stencil; i++)
            {
                int il_old = int(i);
                int il_new = int(i+Nx);
                int jb_old = int(j);
                int jb_new = int(j+Ny);

                int ir_old = int(i+Nx+max_stencil);
                int ir_new = int(i+max_stencil);
                int jt_old = int(j+Ny+max_stencil);
                int jt_new = int(j+max_stencil);

                // bottom left
                U(2*il_old, jb_old) = U(2*il_new,jb_new);
                U(2*il_old+1, jb_old) = U(2*il_new+1,jb_new);

                // bottom right
                U(2*ir_old, jb_old) = U(2*ir_new,jb_new);
                U(2*ir_old+1, jb_old) = U(2*ir_new+1,jb_new);

                // top left
                U(2*il_old, jt_old) = U(2*il_new,jt_new);
                U(2*il_old+1, jt_old) = U(2*il_new+1,jt_new);

                // top right
                U(2*ir_old, jt_old) = U(2*ir_new,jt_new);
                U(2*ir_old+1, jt_old) = U(2*ir_new+1,jt_new);
            }
        }
    }
    else if (problem_type == "uq_ldc") //lid-driven cavity
    {
        // top boundary stencil
        // velocity (1,0)
        for (unsigned int j=0; j<max_stencil; j++)
        {
            for (unsigned int i=0;
                 i<Nx+2*max_stencil; i++)
            {
                int jt_old = int(j+Ny+max_stencil);
                U(2*i, jt_old) = 1;
            }
        }
        std::cout << "Barrier\n\n";
    }
}

Eigen::MatrixXd prepare_sample_data
(const nlohmann::json& config,
 std::vector<double>& vx,
 std::vector<double>& vy,
 unsigned int Nx, unsigned int Ny, unsigned int max_stencil)
{
    std::string problem_type = config["problem_type"];
    Eigen::MatrixXd U;

    bool bool_with_bdry = true;
    if (config.contains("with_boundary")) {
        bool_with_bdry = config["with_boundary"];
    }

    if (bool_with_bdry)
    {
        U.resize(2*(Nx+2*max_stencil), Ny+2*max_stencil);
        U.setZero();

        // interior
        for (unsigned int j=0; j<Ny; j++)
        {
            unsigned int jj = j+max_stencil;
            for (unsigned int i=0; i<Nx; i++)
            {
                unsigned int ii = i+max_stencil;
                U(2*ii, jj) = vx[i+j*Nx];
                U(2*ii+1, jj) = vy[i+j*Nx];
            }
        }
        prepare_bdry_data(config, U,
                          Nx, Ny, max_stencil);
    }
    else // Nx - 2*max_stencil, Ny - 2*max_stencil
    {
        U.resize(2*Nx, Ny);

        // interior
        for (unsigned int j=0; j<Ny; j++)
            for (unsigned int i=0; i<Nx; i++)
            {
                U(2*i, j) = vx[i+j*Nx];
                U(2*i+1, j) = vy[i+j*Nx];
            }
    }

    return std::move(U);
}

void pp_uq_structure_mean (const nlohmann::json config)
{
    unsigned int lx = config["level_x"];
    unsigned int N = config["num_points_1d"];
    double max_offset = config["max_offset"];

    std::string base_dir = config["base_dir"];
    std::string sub_dir = config["sub_dir"];
    std::string in_dir = base_dir+"/"+sub_dir+"/stats/";

    const std::string filename
            = in_dir+"velocity_mean_lx"
            +std::to_string(lx)+".json";
    std::cout << filename << std::endl;
    std::ifstream file(filename);
    assert(file.good());
    nlohmann::json json_file;
    file >> json_file;

    auto vx = json_file["vx"].get<std::vector<double>>();
    auto vy = json_file["vy"].get<std::vector<double>>();

    int p_max = 3;
    unsigned int max_stencil
            = static_cast<unsigned int>(max_offset*N);
    StructureBase structBase{};

    Eigen::VectorXi stencil
            = Eigen::VectorXi::LinSpaced(max_stencil,
                                         1,
                                         int(max_stencil));
    Eigen::VectorXd offsets (max_stencil);
    for (unsigned int i=0; i<max_stencil; i++)
        offsets(i) = stencil(i)*(1./N);
    std::cout << offsets.transpose() << std::endl;

    auto U = prepare_sample_data(config, vx, vy,
                                 N, N, max_stencil);
    Eigen::MatrixXd smean(stencil.size(), p_max);
    for (int p=1; p<=p_max; p++) {
        auto tic = high_resolution_clock::now();
        smean.col(p-1) = structBase.eval
                (U, stencil, p)/(N*N);
        smean.col(p-1) = (smean.col(p-1)).array().pow(1./p);
        auto toc = high_resolution_clock::now();
        auto duration = (duration_cast<seconds>
                         (toc - tic)).count();
        std::cout << "Elapsed time for p=" << p << ",\t"
                  << duration << std::endl;
    }
    vx.clear();
    vy.clear();
    std::cout << smean << std::endl;

    std::string out_dir = in_dir;
    bool bool_with_bdry = true;
    if (config.contains("with_boundary")) {
        bool_with_bdry = config["with_boundary"];
    }
    if (bool_with_bdry) {
      out_dir += "structure/";
    } else {
      out_dir += "structure_noBdry/";
    }

    bool dump_output = config["dump_output"];
    if (dump_output) {
        dump_structure_to_json (out_dir, "structure_mean",
                                lx, offsets, smean);
    }
}

Eigen::MatrixXd prepare_fluctuations_data
(const nlohmann::json& config,
 std::vector<double>& vx,
 std::vector<double>& vy,
 std::vector<double>& vx_mean,
 std::vector<double>& vy_mean,
 unsigned int Nx, unsigned int Ny, unsigned int max_stencil)
{
    std::string problem_type = config["problem_type"];
    Eigen::MatrixXd U;

    bool bool_with_bdry = true;
    if (config.contains("with_boundary")) {
        bool_with_bdry = config["with_boundary"];
    }

    if (bool_with_bdry)
    {
        U.resize(2*(Nx+2*max_stencil), Ny+2*max_stencil);
        U.setZero();

        // interior
        for (unsigned int j=0; j<Ny; j++)
        {
            unsigned int jj = j+max_stencil;
            for (unsigned int i=0; i<Nx; i++)
            {
                unsigned int ii = i+max_stencil;
                U(2*ii, jj) = vx[i+j*Nx] - vx_mean[i+j*Nx];
                U(2*ii+1, jj) = vy[i+j*Nx] - vy_mean[i+j*Nx];
            }
        }
        prepare_bdry_data(config, U,
                          Nx, Ny, max_stencil);
    }
    else // Nx - 2*max_stencil, Ny - 2*max_stencil
    {
        U.resize(2*Nx, Ny);

        // interior
        for (unsigned int j=0; j<Ny; j++)
            for (unsigned int i=0; i<Nx; i++)
            {
                U(2*i, j) = vx[i+j*Nx] - vx_mean[i+j*Nx];
                U(2*i+1, j) = vy[i+j*Nx] - vy_mean[i+j*Nx];
            }
    }

    return std::move(U);
}

void pp_uq_structure_fluctuations
(const nlohmann::json config)
{
    int nprocs, myrank;
    MPI_Comm global_comm(MPI_COMM_WORLD);
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    unsigned int lx = config["level_x"];
    unsigned int N = config["num_points_1d"];

    std::string base_dir = config["base_dir"];
    std::string sub_dir = config["sub_dir"];
    std::string in_dir = base_dir+"/"+sub_dir+
            "/pp_lx"+std::to_string(lx)+"/";

    // Read mean field data
    const std::string filename
            = base_dir+"/"+sub_dir+
            "/stats/velocity_mean_lx"
            +std::to_string(lx)+".json";
    std::cout << filename << std::endl;
    std::ifstream file(filename);
    assert(file.good());
    nlohmann::json json_file;
    file >> json_file;

    auto vx_mean
            = json_file["vx"].get<std::vector<double>>();
    auto vy_mean
            = json_file["vy"].get<std::vector<double>>();

    // Process samples
    const int nsamples = config["uq_num_samples"];

    // make schedule
    int mynsamples;
    Array<int> dist_nsamples;
    std::tie(mynsamples, dist_nsamples)
            = make_schedule(global_comm, nsamples);

    // make sample Ids
    auto all_mysampleIds
            = make_sampleIds(global_comm,
                             mynsamples,
                             dist_nsamples);

    // structure function parameters
    int p_max = 3;
    double max_offset = config["max_offset"];
    unsigned int max_stencil
            = static_cast<unsigned int>(max_offset*N);
    Eigen::VectorXi stencil
            = Eigen::VectorXi::LinSpaced(max_stencil,
                                         1,
                                         int(max_stencil));
    Eigen::VectorXd offsets (max_stencil);
    for (unsigned int i=0; i<max_stencil; i++)
        offsets(i) = stencil(i)*(1./N);
    std::cout << offsets.transpose() << std::endl;

    StructureBase structBase{};
    Eigen::MatrixXd mySfluct(stencil.size(), p_max);
    mySfluct.setZero();
    for (int k=0; k<mynsamples; k++)
    {
        int sId = all_mysampleIds[k];

        // read sample data
        const std::string filename
                = in_dir+"velocity_s"
                +std::to_string(sId)+".json";
        std::cout << filename << std::endl;
        std::ifstream file(filename);
        assert(file.good());
        nlohmann::json json_file;
        file >> json_file;
        file.close();

        auto vx = json_file["vx"].get<std::vector<double>>();
        auto vy = json_file["vy"].get<std::vector<double>>();

        auto U = prepare_fluctuations_data
                (config, vx, vy, vx_mean, vy_mean,
                 N, N, max_stencil);
        Eigen::MatrixXd sampleSfluct(stencil.size(), p_max);
        for (int p=1; p<=p_max; p++) {
            sampleSfluct.col(p-1) = structBase.eval
                    (U, stencil, p)/(N*N);
        }
        mySfluct+= sampleSfluct*(1./nsamples);

        vx.clear();
        vy.clear();
    }
    vx_mean.clear();
    vy_mean.clear();

    // reduce sfluct data on root
    {
        Eigen::MatrixXd sfluct;
        if (myrank == IamRoot) {
            sfluct.resize(stencil.size(), p_max);
        }

        MPI_Reduce(mySfluct.data(), sfluct.data(),
                   int(stencil.size())*p_max,
                   MPI_DOUBLE, MPI_SUM,
                   IamRoot, global_comm);

        // dump output
        if (myrank == IamRoot)
        {
            for (int p=1; p<=p_max; p++)
                sfluct.col(p-1)
                        = (sfluct.col(p-1)).array().pow(1./p);

            std::cout << "\n" << sfluct << std::endl;
            std::string out_dir = base_dir+"/"+sub_dir+"/";
            bool bool_with_bdry = true;
            if (config.contains("with_boundary")) {
                bool_with_bdry = config["with_boundary"];
            }
            if (bool_with_bdry) {
              out_dir += "stats/structure/";
            } else {
              out_dir += "stats/structure_noBdry/";
            }

            bool dump_output = config["dump_output"];
            if (dump_output) {
                dump_structure_to_json
                        (out_dir, "structure_fluctuations",
                         lx, offsets, sfluct);
            }
        }
    }
}

void pp_uq_structure_cube (const nlohmann::json config)
{
    int nprocs, myrank;
    MPI_Comm global_comm(MPI_COMM_WORLD);
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    unsigned int lx = config["level_x"];
    unsigned int N = config["num_points_1d"];
    double max_offset = config["max_offset"];

    std::string base_dir = config["base_dir"];
    std::string sub_dir = config["sub_dir"];
    std::string in_dir = base_dir+"/"+sub_dir+
            "/pp_lx"+std::to_string(lx)+"/";

    const int nsamples = config["uq_num_samples"];

    // make schedule
    int mynsamples;
    Array<int> dist_nsamples;
    std::tie(mynsamples, dist_nsamples)
            = make_schedule(global_comm, nsamples);

    // make sample Ids
    auto all_mysampleIds
            = make_sampleIds(global_comm,
                             mynsamples,
                             dist_nsamples);

    // structure function parameters
    int p_max = 3;
    unsigned int max_stencil
            = static_cast<unsigned int>(max_offset*N);
    Eigen::VectorXi stencil
            = Eigen::VectorXi::LinSpaced(max_stencil,
                                         1,
                                         int(max_stencil));
    Eigen::VectorXd offsets (max_stencil);
    for (unsigned int i=0; i<max_stencil; i++)
        offsets(i) = stencil(i)*(1./N);
    std::cout << offsets.transpose() << std::endl;

    StructureBase structBase{};
    Eigen::MatrixXd myScube(stencil.size(), p_max);
    myScube.setZero();
    for (int k=0; k<mynsamples; k++)
    {
        int sId = all_mysampleIds[k];

        // read sample data
        const std::string filename
                = in_dir+"velocity_s"
                +std::to_string(sId)+".json";
        std::cout << filename << std::endl;
        std::ifstream file(filename);
        assert(file.good());
        nlohmann::json json_file;
        file >> json_file;
        file.close();

        auto vx = json_file["vx"].get<std::vector<double>>();
        auto vy = json_file["vy"].get<std::vector<double>>();

        auto U = prepare_sample_data
                (config, vx, vy, N, N, max_stencil);

        Eigen::MatrixXd sampleScube(stencil.size(), p_max);
        for (int p=1; p<=p_max; p++) {
            sampleScube.col(p-1) = structBase.eval
                    (U, stencil, p)/(N*N);
        }
        myScube += sampleScube*(1./nsamples);

        vx.clear();
        vy.clear();
    }
    //std::cout << "\n" << myScube << std::endl;

    // reduce scube data on root
    {
        Eigen::MatrixXd scube;
        if (myrank == IamRoot) {
            scube.resize(stencil.size(), p_max);
        }

        MPI_Reduce(myScube.data(), scube.data(),
                   int(stencil.size())*p_max,
                   MPI_DOUBLE, MPI_SUM,
                   IamRoot, global_comm);

        // dump output
        if (myrank == IamRoot)
        {
            for (int p=1; p<=p_max; p++)
                scube.col(p-1)
                        = (scube.col(p-1)).array().pow(1./p);

            std::cout << "\n" << scube << std::endl;
            std::string out_dir = base_dir+"/"+sub_dir+"/";
            bool bool_with_bdry = true;
            if (config.contains("with_boundary")) {
                bool_with_bdry = config["with_boundary"];
            }
            if (bool_with_bdry) {
              out_dir += "stats/structure/";
            } else {
              out_dir += "stats/structure_noBdry/";
            }

            dump_structure_to_json (out_dir,
                                    "structure_cube",
                                    lx, offsets, scube);
        }
    }
}


int main(int argc, char *argv[])
{   
    // Initialize MPI.
    MPI_Init(&argc, &argv);

    // Read config json
    auto config = get_global_config(argc, argv);
    const std::string run = config["run"];
    
    if (run == "pp_structure_mean") {
        pp_uq_structure_mean(std::move(config));
    }
    else if (run == "pp_structure_fluctuations") {
        pp_uq_structure_fluctuations(std::move(config));
    }
    else if (run == "pp_structure_cube") {
        pp_uq_structure_cube(std::move(config));
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}


// End of file
