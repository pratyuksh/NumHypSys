#include "../include/includes.hpp"
#include "../include/core/config.hpp"
#include "../include/core/utilities.hpp"
#include "../include/mymfem/utilities.hpp"
#include "../include/mymfem/cell_avgs.hpp"

#include "../include/uq/stats/cells_hash_table.hpp"
#include "../include/uq/stats/structure_base.hpp"
#include "../include/uq/stats/stats.hpp"
#include "../include/uq/scheduler/scheduler.hpp"
#include "../include/uq/sampler/sampler.hpp"

#include <iostream>
#include <chrono>

namespace fs = std::filesystem;
using namespace std::chrono;


//! Writes element-wise constant L2 grid function to file
void dump_ewConsL2GridFn(FiniteElementSpace *sfes,
                         Vector& V,
                         std::string fileName)
{
    auto v = new GridFunction();
    v->MakeRef(sfes, V);

    std::ofstream fileOfs(fileName.c_str());
    if (!fileOfs.is_open()) {
        std::cout << "Error while opening "
                     "file: "
                  << fileName << std::endl;
    }
    else {
        int precision = 8;
        fileOfs.precision(precision);
        v->Save(fileOfs);
    }
    delete v;
}

//! Computes mean and variance of velocity
void compute_meanVar (const nlohmann::json config)
{
    int nprocs, myrank;
    MPI_Comm global_comm(MPI_COMM_WORLD);
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    unsigned int lx = config["level_x"];
    const int nsamples = config["uq_num_samples"];

    std::string base_in_dir = config["input_dir"];
    std::string sub_in_dir = config["sub_in_dir"];
    std::string in_dir = base_in_dir+"/"+sub_in_dir+"/";

    // mesh
    const std::string mesh_file
            = in_dir+"mesh_lx"+std::to_string(lx);
    std::shared_ptr<Mesh> mesh
            = std::make_shared<Mesh>(mesh_file.c_str());

    // print mesh info
    double hMin, hMax, kappaMin, kappaMax;
    mesh->GetCharacteristics(hMin, hMax, kappaMin, kappaMax);
    std::cout << "  Minimum cell size: "
              << hMin << std::endl;
    std::cout << "  Maximum cell size: "
              << hMax << std::endl;

    // cell averages
    auto cellAvgs
            = std::make_unique<CellAverages>(mesh.get());
    auto sfes0 = cellAvgs->get_sfes();

    // make schedule
    int mynsamples;
    Array<int> dist_nsamples;
    std::tie(mynsamples, dist_nsamples)
            = make_schedule(global_comm, nsamples);

    // make sample Ids
    Array<int> all_mysampleIds;
    all_mysampleIds
            = make_sampleIds(global_comm,
                             mynsamples, dist_nsamples);


    // change input directory for samples
    in_dir += "lx"+std::to_string(lx)+"/";

    // auxiliary variables
    Vector v, vx, vy;
    v.SetSize(sfes0->GetTrueVSize());
    vx.SetSize(sfes0->GetTrueVSize());
    vy.SetSize(sfes0->GetTrueVSize());

    Vector v_mean, vx_mean, vy_mean;
    v_mean.SetSize(sfes0->GetTrueVSize());
    vx_mean.SetSize(sfes0->GetTrueVSize());
    vy_mean.SetSize(sfes0->GetTrueVSize());

    Vector v_variance, vx_variance, vy_variance;
    v_variance.SetSize(sfes0->GetTrueVSize());
    vx_variance.SetSize(sfes0->GetTrueVSize());
    vy_variance.SetSize(sfes0->GetTrueVSize());

    // init stats computation
    auto stats_vx = std::make_unique<Statistics>
            (global_comm, nsamples, vx.Size());
    auto stats_vy = std::make_unique<Statistics>
            (global_comm, nsamples, vy.Size());
    auto stats_v = std::make_unique<Statistics>
            (global_comm, nsamples, v.Size());

    for (int k=0; k<mynsamples; k++)
    {
        int sId = all_mysampleIds[k];

        // read sample data
        const std::string velocity_file
                = in_dir+"velocity_s"+std::to_string(sId);
        std::cout << velocity_file << std::endl;
        std::ifstream velocity_ifs(velocity_file.c_str());
        auto velocity = new GridFunction (mesh.get(),
                                          velocity_ifs);

        // cell-averaged velocity components
        std::tie(vx, vy) = cellAvgs->eval(velocity);
        delete velocity;

        // cell-averaged velocity magnitude
        for (int i=0; i<v.Size(); i++) {
            v(i) = std::sqrt(vx(i)*vx(i)
                             + vy(i)*vy(i));
        }

        // update stats
        stats_vx->add_to_first_moment(vx);
        stats_vx->add_squared_to_second_moment(vx);
        stats_vy->add_to_first_moment(vy);
        stats_vy->add_squared_to_second_moment(vy);
        stats_v->add_to_first_moment(v);
        stats_v->add_squared_to_second_moment(v);

        // test
        /*{
            std::string base_out_dir = config["input_dir"];
            std::string sub_out_dir = config["sub_in_dir"];
            std::string out_dir = base_out_dir
                    +"/"+sub_out_dir+"/pp_lx"+std::to_string(lx)+"/";
            {
                fs::create_directories(out_dir);
                std::string fileName;

                fileName = out_dir+"vx_s"+std::to_string(sId);
                std::cout << fileName << std::endl;
                dump_ewConsL2GridFn(sfes0, vx, fileName);

                fileName = out_dir+"vy_s"+std::to_string(sId);
                dump_ewConsL2GridFn(sfes0, vy, fileName);
            }
        }*/
    }

    // compute mean and variance
    stats_vx->mean(vx_mean);
    stats_vx->variance(vx_mean, vx_variance);
    stats_vy->mean(vy_mean);
    stats_vy->variance(vy_mean, vy_variance);
    stats_v->mean(v_mean);
    stats_v->variance(v_mean, v_variance);

    // dump stats to files
    if (myrank == IamRoot)
    {
        std::string base_out_dir = config["output_dir"];
        std::string sub_out_dir = config["sub_out_dir"];
        std::string out_dir = base_out_dir
                +"/"+sub_out_dir+"/";
        bool bool_dump_output = false;
        if (config.contains("dump_output")) {
            bool_dump_output = config["dump_output"];
        }
        if (bool_dump_output)
        {
            fs::create_directories(out_dir);
            std::string fileName;

            // Mean
            fileName = out_dir+"vxMean_lx"
                    +std::to_string(lx);
            std::cout << fileName << std::endl;
            dump_ewConsL2GridFn(sfes0, vx_mean,
                                fileName);

            fileName = out_dir+"vyMean_lx"
                    +std::to_string(lx);
            dump_ewConsL2GridFn(sfes0, vy_mean,
                                fileName);

            fileName = out_dir+"vMean_lx"
                    +std::to_string(lx);
            dump_ewConsL2GridFn(sfes0, v_mean,
                                fileName);

            // Variance
            fileName = out_dir+"vxVariance_lx"
                    +std::to_string(lx);
            std::cout << fileName << std::endl;
            dump_ewConsL2GridFn(sfes0, vx_variance,
                                fileName);

            fileName = out_dir+"vyVariance_lx"
                    +std::to_string(lx);
            dump_ewConsL2GridFn(sfes0, vy_variance,
                                fileName);

            fileName = out_dir+"vVariance_lx"
                    +std::to_string(lx);
            dump_ewConsL2GridFn(sfes0, v_variance,
                                fileName);
        }
    }
}


//! Writes structure function computations to json file
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

//! Computes structure function
//! of the mean of velocity
void compute_structure_mean (const nlohmann::json config)
{
    unsigned int lx = config["level_x"];

    std::string base_dir = config["base_dir"];
    std::string sub_dir = config["sub_dir"];
    std::string in_dir = base_dir+"/"+sub_dir+"/";

    // parameters
    int Nx = config["Nx"];
    double xl = config["xl"];
    double xr = config["xr"];

    int Ny = config["Ny"];
    double yl = config["yl"];
    double yr = config["yr"];

    double min_offset = config["min_offset"];
    double max_offset = config["max_offset"];
    unsigned int M = config["num_offset_levels"];

    // set offsets
    int p_max = 3;
    Eigen::VectorXd offsets (M+1);
    offsets(0) = 0;
    for (unsigned int i=0; i<M; i++)
        offsets(i+1) = min_offset
                + i*((max_offset - min_offset)/(M-1));
    std::cout << "\n" << offsets.transpose() << std::endl;

    // mesh
    const std::string mesh_file
            = in_dir+"mesh_lx"+std::to_string(lx);
    std::shared_ptr<Mesh> mesh
            = std::make_shared<Mesh>(mesh_file.c_str());

    // print mesh info
    double hMin, hMax, kappaMin, kappaMax;
    mesh->GetCharacteristics(hMin, hMax, kappaMin, kappaMax);
    std::cout << "  Minimum cell size: "
              << hMin << std::endl;
    std::cout << "  Maximum cell size: "
              << hMax << std::endl;

    // vx mean
    const std::string vxMean_file
            = in_dir+"stats/vxMean_lx"+std::to_string(lx);
    std::ifstream vxMean_ifs(vxMean_file.c_str());
    auto vxMean = new GridFunction (mesh.get(), vxMean_ifs);

    // vy mean
    const std::string vyMean_file
            = in_dir+"stats/vyMean_lx"+std::to_string(lx);
    std::ifstream vyMean_ifs(vyMean_file.c_str());
    auto vyMean = new GridFunction (mesh.get(), vyMean_ifs);

    // make hash table
    auto cellsHashTable
            = make_hashTable(mesh, Nx, xl, xr, Ny, yl, yr);

    // set velocities in hash table
    update_hashTable(cellsHashTable, mesh, vxMean, vyMean);

    // compute structure
    StructureBase structBase{};
    Eigen::MatrixXd smean(M, p_max);
    for (int p=1; p<=p_max; p++) {
        auto tic = high_resolution_clock::now();
        smean.col(p-1) = structBase.eval
                (cellsHashTable, offsets, p);
        smean.col(p-1) = (smean.col(p-1)).array().pow(1./p);
        auto toc = high_resolution_clock::now();
        auto duration = (duration_cast<seconds>
                         (toc - tic)).count();
        std::cout << "Elapsed time for p=" << p << ",\t"
                  << duration << std::endl;
    }
    delete vxMean;
    delete vyMean;
    std::cout << smean << std::endl;

    bool dump_output = config["dump_output"];
    std::string out_dir = in_dir;
    out_dir += "stats/structure_noBdry/";

    if (dump_output) {
        Eigen::VectorXd offsets_ = offsets.tail(M);
        dump_structure_to_json (out_dir, "structure_mean",
                                lx, offsets_, smean);
    }
}

//! Computes structure function
//! of the fluctutations of velocity
void compute_structure_fluctuations
(const nlohmann::json config)
{
    int nprocs, myrank;
    MPI_Comm global_comm(MPI_COMM_WORLD);
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    unsigned int lx = config["level_x"];

    std::string base_dir = config["base_dir"];
    std::string sub_dir = config["sub_dir"];
    std::string in_dir = base_dir+"/"+sub_dir+"/";

    // parameters
    int Nx = config["Nx"];
    double xl = config["xl"];
    double xr = config["xr"];

    int Ny = config["Ny"];
    double yl = config["yl"];
    double yr = config["yr"];

    double min_offset = config["min_offset"];
    double max_offset = config["max_offset"];
    unsigned int M = config["num_offset_levels"];

    // set offsets
    int p_max = 3;
    Eigen::VectorXd offsets (M+1);
    offsets(0) = 0;
    for (unsigned int i=0; i<M; i++)
        offsets(i+1) = min_offset
                + i*((max_offset - min_offset)/(M-1));
    std::cout << "\n" << offsets.transpose() << std::endl;

    // mesh
    const std::string mesh_file
            = in_dir+"mesh_lx"+std::to_string(lx);
    std::shared_ptr<Mesh> mesh
            = std::make_shared<Mesh>(mesh_file.c_str());

    // print mesh info
    double hMin, hMax, kappaMin, kappaMax;
    mesh->GetCharacteristics(hMin, hMax, kappaMin, kappaMax);
    std::cout << "  Minimum cell size: "
              << hMin << std::endl;
    std::cout << "  Maximum cell size: "
              << hMax << std::endl;

    // vx mean
    const std::string vxMean_file
            = in_dir+"stats/vxMean_lx"+std::to_string(lx);
    std::ifstream vxMean_ifs(vxMean_file.c_str());
    auto vxMean = new GridFunction (mesh.get(), vxMean_ifs);

    // vy mean
    const std::string vyMean_file
            = in_dir+"stats/vyMean_lx"+std::to_string(lx);
    std::ifstream vyMean_ifs(vyMean_file.c_str());
    auto vyMean = new GridFunction (mesh.get(), vyMean_ifs);

    // cell averages
    auto cellAvgs
            = std::make_unique<CellAverages>(mesh.get());
    auto sfes0 = cellAvgs->get_sfes();

    // process samples
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

    // make hash table
    auto cellsHashTable
            = make_hashTable(mesh, Nx, xl, xr, Ny, yl, yr);

    // change input directory for samples
    in_dir += "lx"+std::to_string(lx)+"/";

    // compute structure
    StructureBase structBase{};
    Eigen::MatrixXd mySfluct(M, p_max);
    mySfluct.setZero();
    Vector Vx, Vy;
    GridFunction vx, vy;
    for (int k=0; k<mynsamples; k++)
    {
        int sId = all_mysampleIds[k];

        // read sample data
        const std::string velocity_file
                = in_dir+"velocity_s"+std::to_string(sId);
        std::cout << velocity_file << std::endl;
        std::ifstream velocity_ifs(velocity_file.c_str());
        auto velocity = new GridFunction (mesh.get(),
                                          velocity_ifs);

        // cell-averaged velocity components
        std::tie(Vx, Vy) = cellAvgs->eval(velocity);
        delete velocity;

        // set velocity fluctuations in hash table
        vx.MakeRef(sfes0, Vx);
        vy.MakeRef(sfes0, Vy);
        update_hashTable(cellsHashTable, mesh,
                         &vx, vxMean, &vy, vyMean);

        Eigen::MatrixXd sampleSfluct(M, p_max);
        for (int p=1; p<=p_max; p++) {
            sampleSfluct.col(p-1) = structBase.eval
                    (cellsHashTable, offsets, p);
        }
        mySfluct += sampleSfluct*(1./nsamples);
    }
    delete vxMean;
    delete vyMean;

    // reduce sfluct data on root
    {
        Eigen::MatrixXd sfluct;
        if (myrank == IamRoot) {
            sfluct.resize(M, p_max);
        }

        MPI_Reduce(mySfluct.data(), sfluct.data(),
                   int(M)*p_max,
                   MPI_DOUBLE, MPI_SUM,
                   IamRoot, global_comm);

        // dump output
        if (myrank == IamRoot)
        {
            for (int p=1; p<=p_max; p++)
                sfluct.col(p-1)
                        = (sfluct.col(p-1)).array().pow(1./p);

            std::cout << "\n" << sfluct << std::endl;

            bool dump_output = config["dump_output"];
            std::string out_dir = base_dir+"/"+sub_dir+"/";
            out_dir += "stats/structure_noBdry/";

            if (dump_output) {
                Eigen::VectorXd offsets_ = offsets.tail(M);
                dump_structure_to_json
                        (out_dir, "structure_fluctuations",
                         lx, offsets_, sfluct);
            }
        }
    }
}

//! Computes structure function cube of velocity
void compute_structure_cube
(const nlohmann::json config)
{
    int nprocs, myrank;
    MPI_Comm global_comm(MPI_COMM_WORLD);
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    unsigned int lx = config["level_x"];

    std::string base_dir = config["base_dir"];
    std::string sub_dir = config["sub_dir"];
    std::string in_dir = base_dir+"/"+sub_dir+"/";

    // parameters
    int Nx = config["Nx"];
    double xl = config["xl"];
    double xr = config["xr"];

    int Ny = config["Ny"];
    double yl = config["yl"];
    double yr = config["yr"];

    double min_offset = config["min_offset"];
    double max_offset = config["max_offset"];
    unsigned int M = config["num_offset_levels"];

    // set offsets
    int p_max = 3;
    Eigen::VectorXd offsets (M+1);
    offsets(0) = 0;
    for (unsigned int i=0; i<M; i++)
        offsets(i+1) = min_offset
                + i*((max_offset - min_offset)/(M-1));
    std::cout << "\n" << offsets.transpose() << std::endl;

    // mesh
    const std::string mesh_file
            = in_dir+"mesh_lx"+std::to_string(lx);
    std::shared_ptr<Mesh> mesh
            = std::make_shared<Mesh>(mesh_file.c_str());

    // print mesh info
    double hMin, hMax, kappaMin, kappaMax;
    mesh->GetCharacteristics(hMin, hMax, kappaMin, kappaMax);
    std::cout << "  Minimum cell size: "
              << hMin << std::endl;
    std::cout << "  Maximum cell size: "
              << hMax << std::endl;

    // cell averages
    auto cellAvgs
            = std::make_unique<CellAverages>(mesh.get());
    auto sfes0 = cellAvgs->get_sfes();

    // process samples
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

    // make hash table
    auto cellsHashTable
            = make_hashTable(mesh, Nx, xl, xr, Ny, yl, yr);

    // change input directory for samples
    in_dir += "lx"+std::to_string(lx)+"/";

    // compute structure
    StructureBase structBase{};
    Eigen::MatrixXd myScube(M, p_max);
    myScube.setZero();
    Vector Vx, Vy;
    GridFunction vx, vy;
    for (int k=0; k<mynsamples; k++)
    {
        int sId = all_mysampleIds[k];

        // read sample data
        const std::string velocity_file
                = in_dir+"velocity_s"+std::to_string(sId);
        std::cout << velocity_file << std::endl;
        std::ifstream velocity_ifs(velocity_file.c_str());
        auto velocity = new GridFunction (mesh.get(),
                                          velocity_ifs);

        // cell-averaged velocity components
        std::tie(Vx, Vy) = cellAvgs->eval(velocity);
        delete velocity;

        // set velocity in hash table
        vx.MakeRef(sfes0, Vx);
        vy.MakeRef(sfes0, Vy);
        update_hashTable(cellsHashTable, mesh, &vx, &vy);

        Eigen::MatrixXd sampleScube(M, p_max);
        for (int p=1; p<=p_max; p++) {
            sampleScube.col(p-1) = structBase.eval
                    (cellsHashTable, offsets, p);
        }
        myScube += sampleScube*(1./nsamples);
    }

    // reduce scube data on root
    {
        Eigen::MatrixXd scube;
        if (myrank == IamRoot) {
            scube.resize(M, p_max);
        }

        MPI_Reduce(myScube.data(), scube.data(),
                   int(M)*p_max,
                   MPI_DOUBLE, MPI_SUM,
                   IamRoot, global_comm);

        // dump output
        if (myrank == IamRoot)
        {
            for (int p=1; p<=p_max; p++)
                scube.col(p-1)
                        = (scube.col(p-1)).array().pow(1./p);

            std::cout << "\n" << scube << std::endl;

            bool dump_output = config["dump_output"];
            std::string out_dir = base_dir+"/"+sub_dir+"/";
            out_dir += "stats/structure_noBdry/";

            if (dump_output) {
                Eigen::VectorXd offsets_ = offsets.tail(M);
                dump_structure_to_json
                        (out_dir, "structure_cube",
                         lx, offsets_, scube);
            }
        }
    }
}

//! Computes structure function
//! of the mean of velocity
//! in parallel
void compute_structure_mean_test
(const nlohmann::json config)
{
    int nprocs, myrank;
    MPI_Comm global_comm(MPI_COMM_WORLD);
    MPI_Comm_size(global_comm, &nprocs);
    MPI_Comm_rank(global_comm, &myrank);

    unsigned int lx = config["level_x"];

    std::string base_dir = config["base_dir"];
    std::string sub_dir = config["sub_dir"];
    std::string in_dir = base_dir+"/"+sub_dir+"/";

    // parameters
    int Nx = config["Nx"];
    double xl = config["xl"];
    double xr = config["xr"];

    int Ny = config["Ny"];
    double yl = config["yl"];
    double yr = config["yr"];

    double min_offset = config["min_offset"];
    double max_offset = config["max_offset"];
    unsigned int M = config["num_offset_levels"];

    // set offsets
    int p_max = 3;
    Eigen::VectorXd offsets (M+1);
    offsets(0) = 0;
    for (unsigned int i=0; i<M; i++)
        offsets(i+1) = min_offset
                + i*((max_offset - min_offset)/(M-1));
    std::cout << "\n" << offsets.transpose() << std::endl;

    // mesh
    const std::string mesh_file
            = in_dir+"pmesh_lx"+std::to_string(lx);
    std::shared_ptr<Mesh> mesh
            = std::make_shared<Mesh>(mesh_file.c_str());

    // print mesh info
    double hMin, hMax, kappaMin, kappaMax;
    mesh->GetCharacteristics(hMin, hMax, kappaMin, kappaMax);
    std::cout << "  Minimum cell size: "
              << hMin << std::endl;
    std::cout << "  Maximum cell size: "
              << hMax << std::endl;

    // vx mean
    const std::string vxMean_file
            = in_dir+"vxMean_lx"+std::to_string(lx);
    std::ifstream vxMean_ifs(vxMean_file.c_str());
    auto vxMean = new GridFunction (mesh.get(), vxMean_ifs);

    // vy mean
    const std::string vyMean_file
            = in_dir+"vyMean_lx"+std::to_string(lx);
    std::ifstream vyMean_ifs(vyMean_file.c_str());
    auto vyMean = new GridFunction (mesh.get(), vyMean_ifs);

    // make parallel hash table
    auto pcellsHashTable
            = make_hashTable(global_comm,
                             mesh, Nx, xl, xr, Ny, yl, yr);

    // set velocities in hash table
    update_hashTable(pcellsHashTable, mesh, vxMean, vyMean);

    // release memory
    mesh.reset();
    delete vxMean;
    delete vyMean;

    // compute structure
    StructureBase structBase{};
    Eigen::MatrixXd smean(M, p_max);
    for (int p=1; p<=p_max; p++) {
        double start = MPI_Wtime();
        smean.col(p-1) = structBase.eval
                (pcellsHashTable, offsets, p);
        smean.col(p-1) = (smean.col(p-1)).array().pow(1./p);
        double end = MPI_Wtime();
        std::cout << "Elapsed time: " << end-start
                  << std::endl;
    }

    if (myrank == IamRoot)
    {
        std::cout << smean << std::endl;
    }
}


int main(int argc, char *argv[])
{   
    // Initialize MPI.
    MPI_Init(&argc, &argv);

    // Read config json
    auto config = get_global_config(argc, argv);
    const std::string run = config["run"];
    
    if (run == "meanVar") {
        compute_meanVar(std::move(config));
    }
    else if (run == "structure_mean") {
        compute_structure_mean(std::move(config));
    }
    else if (run == "structure_fluctuations") {
        compute_structure_fluctuations(std::move(config));
    }
    else if (run == "structure_cube") {
        compute_structure_cube(std::move(config));
    }
    else if (run == "structure_mean_test") {
        compute_structure_mean_test(std::move(config));
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}


// End of file
