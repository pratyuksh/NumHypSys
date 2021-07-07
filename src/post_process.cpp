#include "../include/includes.hpp"
#include "../include/core/config.hpp"
#include "../include/core/utilities.hpp"
#include "../include/core/error.hpp"
#include "../include/mymfem/utilities.hpp"
#include "../include/incompNS/observer.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;


//! Writes output
void dump_data_to_json_file
(const nlohmann::json& config,
 Eigen::MatrixXd &data_x,
 Eigen::MatrixXd &data_y,
 std::vector<std::string> &data_x_names,
 std::vector<std::string> &data_y_names)
{
    std::string model = config["model"];
    std::string base_out_dir = config["output_dir"];
    std::string out_filename = config["out_filename"];

    std::string out_dir = base_out_dir+"/"+model;
    fs::create_directories(out_dir);

    std::string outfile = out_dir+"/"+out_filename+".json";

    std::cout << data_x << std::endl;
    std::cout << data_y << std::endl;

    auto json = nlohmann::json{};
    for (unsigned int j=0; j<data_x.cols(); j++) {
        for (unsigned int i=0; i<data_x.rows(); i++) {
            json[data_x_names[j]][i] = data_x(i,j);
            json[data_y_names[j]][i] = data_y(i,j);
        }
    }

    auto file = std::ofstream(outfile);
    assert(file.good());
    // always check that you can write to the file.

    file << json.dump(2);
    file.close();
}

//! Measure the velocity components
//! for the lid-driven cavity in a unit square
//! at the horizontl and vertical lines
//! running through the center of the physical domain
void pp_pincompNS_ldc (const nlohmann::json config)
{
    std::string sub_dir = config["sub_dir"];
    std::string input_dir = "../output/data/"+sub_dir+"/";

    int lx = config["level_x"];
    
    const std::string mesh_file
            = input_dir+"pmesh_lx"+std::to_string(lx);
    Mesh mesh(mesh_file.c_str());
    
    const std::string observable = config["observable"];
    int time_stamp = config["time_stamp"];

    const std::string sol_file = input_dir+observable
            +"_lx"+std::to_string(lx)
            +"_tId_"+std::to_string(time_stamp);

    std::ifstream sol_ifs(sol_file.c_str());
    GridFunction velocity(&mesh, sol_ifs);
    
    int N = config["number_of_measurement_points"];
    double h = 1./(N-1);
    
    /// measurement points along the vertical line
    /// through the domain center
    DenseMatrix measurePts1(mesh.Dimension(), N);
    Array <int> elIds1(N);
    Array <IntegrationPoint> ips1(N);
    for (int k=0; k<N; k++)
    {
        Vector point;
        measurePts1.GetColumnReference(k, point);
        point(0) = 0.5;
        point(1) = k*h;
    }

    /// measurement points along the horizontal line
    /// through the domain center
    DenseMatrix measurePts2(mesh.Dimension(), N);
    Array <int> elIds2(N);
    Array <IntegrationPoint> ips2(N);
    for (int k=0; k<N; k++)
    {
        Vector point;
        measurePts2.GetColumnReference(k, point);
        point(0) = k*h;
        point(1) = 0.5;
    }

    /// point locator
    bool has_shared_vertices = true;
    PointLocator point_locator(&mesh, has_shared_vertices);

    /// find elements corresponding to measurement points
    auto start = std::chrono::high_resolution_clock::now();
    int init_elId = 0;
    for (int k=0; k<N; k++) {
        Vector point;
        measurePts1.GetColumn(k, point);
        std::tie (elIds1[k],ips1[k])
                = point_locator(point, init_elId);
        init_elId = elIds1[k];
    }
    for (int k=0; k<N; k++) {
        Vector point;
        measurePts2.GetColumn(k, point);
        std::tie (elIds2[k],ips2[k])
                = point_locator(point, init_elId);
        init_elId = elIds2[k];
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    /// measure
    Eigen::MatrixXd dataOut(N,2);
    Vector v1(mesh.Dimension());
    Vector v2(mesh.Dimension());
    for (int k=0; k < N; k++)
    {
        velocity.GetVectorValue(elIds1[k], ips1[k], v1);
        dataOut(k, 0) = v1[0];

        velocity.GetVectorValue(elIds2[k], ips2[k], v2);
        dataOut(k, 1) = v2[1];

        std::cout << k << "\t"
                  << v1[0] << "\t" << v2[1] << std::endl;
    }

    auto duration = std::chrono::duration_cast
            <std::chrono::microseconds>(end - start);
    std::cout << "Search time for measurement points: "
              << duration.count() << std::endl;


    Eigen::MatrixXd dataPts(N,2);
    for (int k=0; k<N; k++) {
        dataPts(k,0) = measurePts1(1,k);
        dataPts(k,1) = measurePts2(0,k);
    }
    std::vector<std::string> dataPtsNames;
    std::vector<std::string> dataOutNames;
    dataPtsNames.push_back("y");
    dataPtsNames.push_back("x");

    dataOutNames.push_back("vx");
    dataOutNames.push_back("vy");

    dump_data_to_json_file(config, dataPts, dataOut,
                           dataPtsNames, dataOutNames);
}


//! Writes output
void dump_data_to_json_file
(const nlohmann::json& config,
 Eigen::VectorXi &data_x,
 Eigen::MatrixXd &data_y,
 std::string &data_x_names,
 std::vector<std::string> &data_y_names)
{
    std::string base_dir = config["base_dir"];
    std::string sub_out_dir = config["sub_out_dir"];
    std::string out_dir = base_dir+"/"+sub_out_dir+"/";
    std::string out_filename = config["out_filename"];
    fs::create_directories(out_dir);
    std::string outfile = out_dir+out_filename+".json";

    //std::cout << outfile << std::endl;
    //std::cout << data_x.transpose() << std::endl;
    //std::cout << data_y << std::endl;

    auto json = nlohmann::json{};
    // data_x
    for (unsigned int i=0; i<data_x.size(); i++)
        json[data_x_names][i] = data_x(i);
    // data_y
    for (unsigned int j=0; j<data_y.cols(); j++) {
        for (unsigned int i=0; i<data_y.rows(); i++) {
            json[data_y_names[j]][i] = data_y(i,j);
        }
    }

    auto file = std::ofstream(outfile);
    assert(file.good());
    // always check that you can write to the file.

    file << json.dump(2);
    file.close();
}

//! Computes the convergence behaviour
//! of mean and variance with mesh refinement
void pp_uq_convg (const nlohmann::json config)
{
    std::string base_dir = config["base_dir"];

    std::string sub_mesh_dir = config["sub_mesh_dir"];
    std::string mesh_dir = base_dir+"/"+sub_mesh_dir+"/";

    std::string sub_in_dir = config["sub_in_dir"];
    std::string in_dir = base_dir+"/"+sub_in_dir+"/";

    int nlevels = config["num_levels_x"];
    int lx_min = config["min_level_x"];

    Eigen::VectorXi meshSize(nlevels);
    Eigen::MatrixXd errorL2(nlevels, 2);
    for (int i=0; i<nlevels; i++)
    {
        int lx1 = lx_min + i;
        int lx2 = lx1+1;

        const std::string mesh_file1
                = mesh_dir+"mesh_lx"+std::to_string(lx1);
        const std::string mesh_file2
                = mesh_dir+"mesh_lx"+std::to_string(lx2);
        Mesh mesh1(mesh_file1.c_str());
        Mesh mesh2(mesh_file2.c_str());

        // mean
        /*const std::string vMean_file1 = in_dir+"vMean"
                +"_lx"+std::to_string(lx1);
        const std::string vMean_file2 = in_dir+"vMean"
                +"_lx"+std::to_string(lx2);
        std::ifstream vMean_ifs1(vMean_file1.c_str());
        std::ifstream vMean_ifs2(vMean_file2.c_str());
        GridFunction vMean1(&mesh1, vMean_ifs1);
        GridFunction vMean2(&mesh2, vMean_ifs2);

        // variance
        const std::string vVar_file1 = in_dir+"vVariance"
                +"_lx"+std::to_string(lx1);
        const std::string vVar_file2 = in_dir+"vVariance"
                +"_lx"+std::to_string(lx2);
        std::ifstream vVar_ifs1(vVar_file1.c_str());
        std::ifstream vVar_ifs2(vVar_file2.c_str());
        GridFunction vVar1(&mesh1, vVar_ifs1);
        GridFunction vVar2(&mesh2, vVar_ifs2);

        //IncompNSObserver observer{};
        //observer(&vMean1); observer(&vVar1);
        //observer(&vMean2); observer(&vVar2);

        ComputeCauchyL2Error computeCauchyL2Error{};
        errorL2(i, 0)
                = computeCauchyL2Error(vMean1, vMean2, true);
        errorL2(i, 1)
                = computeCauchyL2Error(vVar1, vVar2, true);*/

        // mean
        const std::string vxMean_file1 = in_dir+"vxMean"
                +"_lx"+std::to_string(lx1);
        const std::string vxMean_file2 = in_dir+"vxMean"
                +"_lx"+std::to_string(lx2);
        const std::string vyMean_file1 = in_dir+"vyMean"
                +"_lx"+std::to_string(lx1);
        const std::string vyMean_file2 = in_dir+"vyMean"
                +"_lx"+std::to_string(lx2);
        std::ifstream vxMean_ifs1(vxMean_file1.c_str());
        std::ifstream vxMean_ifs2(vxMean_file2.c_str());
        std::ifstream vyMean_ifs1(vyMean_file1.c_str());
        std::ifstream vyMean_ifs2(vyMean_file2.c_str());
        GridFunction vxMean1(&mesh1, vxMean_ifs1);
        GridFunction vxMean2(&mesh2, vxMean_ifs2);
        GridFunction vyMean1(&mesh1, vyMean_ifs1);
        GridFunction vyMean2(&mesh2, vyMean_ifs2);

        // variance
        const std::string vxVar_file1 = in_dir+"vxVariance"
                +"_lx"+std::to_string(lx1);
        const std::string vxVar_file2 = in_dir+"vxVariance"
                +"_lx"+std::to_string(lx2);
        const std::string vyVar_file1 = in_dir+"vyVariance"
                +"_lx"+std::to_string(lx1);
        const std::string vyVar_file2 = in_dir+"vyVariance"
                +"_lx"+std::to_string(lx2);
        std::ifstream vxVar_ifs1(vxVar_file1.c_str());
        std::ifstream vxVar_ifs2(vxVar_file2.c_str());
        std::ifstream vyVar_ifs1(vyVar_file1.c_str());
        std::ifstream vyVar_ifs2(vyVar_file2.c_str());
        GridFunction vxVar1(&mesh1, vxVar_ifs1);
        GridFunction vxVar2(&mesh2, vxVar_ifs2);
        GridFunction vyVar1(&mesh1, vyVar_ifs1);
        GridFunction vyVar2(&mesh2, vyVar_ifs2);

        //IncompNSObserver observer{};
        //observer(&vMean1); observer(&vVar1);
        //observer(&vMean2); observer(&vVar2);

        ComputeCauchyL2Error computeCauchyL2Error{};
        errorL2(i, 0)
                = computeCauchyL2Error(vxMean1, vyMean1,
                                       vxMean2, vyMean2, true);
        errorL2(i, 1)
                = computeCauchyL2Error(vxVar1, vyVar1,
                                       vxVar2, vyVar2, true);

        if (config["problem_type"] == "uq_svs") {
            meshSize(i) = static_cast<int>(4*pow(2, lx2));
        }
        else if (config["problem_type"] == "uq_ldc") {
            meshSize(i) = static_cast<int>(pow(2, lx2));
        }
        else if (config["problem_type"] == "uq_cf") {
            meshSize(i) = static_cast<int>(32*pow(2, lx2));
        }

        std::cout << "\nFor mean, Cauchy L2-error: "
                  << errorL2(i, 0) << std::endl;
        std::cout << "For variance, Cauchy L2-error: "
                  << errorL2(i, 1) << std::endl;
    }
    std::cout << "\n\nFor mean, Cauchy L2-error:\n"
              << errorL2.col(0).transpose() << std::endl;
    std::cout << "\nFor variance, Cauchy L2-error:\n"
              << errorL2.col(1).transpose() << std::endl;

    std::string dataPtsNames("Nx");
    std::vector<std::string> dataOutNames;
    dataOutNames.push_back("mean");
    dataOutNames.push_back("variance");

    bool dump_output = config["dump_output"];
    if (dump_output) {
        dump_data_to_json_file(config, meshSize, errorL2,
                               dataPtsNames, dataOutNames);
    }
}

//! Computes the convergence behaviour
//! of mean and variance with respect to the number of
//! samples on a fixed mesh
void pp_uq_convg_wrt_samples (const nlohmann::json config)
{
    std::string base_dir = config["base_dir"];
    std::string sub_in_dir = config["sub_in_dir"];

    auto list_nsamples = config["list_nsamples"].
            get<std::vector<int>>();
    auto K = list_nsamples.size();

    int lx = config["level_x"];

    Eigen::VectorXi nsamples(K-1);
    Eigen::MatrixXd errorL2(K-1, 2);
    for (unsigned int i=0; i<K-1; i++)
    {
        nsamples(i) = list_nsamples[i];

        // mesh files
        std::string mesh_dir1 = base_dir+"/"
                +sub_in_dir+"/samples"
                +std::to_string(list_nsamples[i])+"/";
        std::string mesh_dir2 = base_dir+"/"
                +sub_in_dir+"/samples"
                +std::to_string(list_nsamples[i+1])+"/";

        const std::string mesh_file1
                = mesh_dir1+"mesh_lx"+std::to_string(lx);
        Mesh mesh1(mesh_file1.c_str());
        const std::string mesh_file2
                = mesh_dir2+"mesh_lx"+std::to_string(lx);
        Mesh mesh2(mesh_file2.c_str());

        std::string input_dir1 = base_dir+"/"
                +sub_in_dir+"/samples"
                +std::to_string(list_nsamples[i])+"/stats/";
        std::string input_dir2 = base_dir+"/"
                +sub_in_dir+"/samples"
                +std::to_string(list_nsamples[i+1])+"/stats/";

        // solution files

        // mean
        /*const std::string vMean_file1 = input_dir1
                +"vMean_lx"+std::to_string(lx);
        const std::string vMean_file2 = input_dir2
                +"vMean_lx"+std::to_string(lx);
        //std::cout << "\n" << vMean_file1 << std::endl;
        //std::cout << vMean_file2 << std::endl;
        std::ifstream vMean_ifs1(vMean_file1.c_str());
        std::ifstream vMean_ifs2(vMean_file2.c_str());
        GridFunction vMean1(&mesh1, vMean_ifs1);
        GridFunction vMean2(&mesh2, vMean_ifs2);

        // variance
        const std::string vVar_file1 = input_dir1
                +"vVariance_lx"+std::to_string(lx);
        const std::string vVar_file2 = input_dir2
                +"vVariance_lx"+std::to_string(lx);
        //std::cout << "\n" << vVar_file1 << std::endl;
        //std::cout << vVar_file2 << std::endl;
        std::ifstream vVar_ifs1(vVar_file1.c_str());
        std::ifstream vVar_ifs2(vVar_file2.c_str());
        GridFunction vVar1(&mesh1, vVar_ifs1);
        GridFunction vVar2(&mesh2, vVar_ifs2);

        //IncompNSObserver observer{};
        //observer.set_bool_visualize(true);
        //observer(&vMean1);// observer(&vVar1);
        //observer(&vMean2);// observer(&vVar2);

        ConstantCoefficient zero(0);

        ComputeCauchyL2Error computeCauchyL2Error{};
        errorL2(i, 0) = computeCauchyL2Error.evalOnSameMesh
                (vMean1, vMean2);

        errorL2(i, 1) = computeCauchyL2Error.evalOnSameMesh
                (vVar1, vVar2);*/

        // mean
        const std::string vxMean_file1 = input_dir1+"vxMean"
                +"_lx"+std::to_string(lx);
        const std::string vxMean_file2 = input_dir2+"vxMean"
                +"_lx"+std::to_string(lx);
        const std::string vyMean_file1 = input_dir1+"vyMean"
                +"_lx"+std::to_string(lx);
        const std::string vyMean_file2 = input_dir2+"vyMean"
                +"_lx"+std::to_string(lx);
        std::ifstream vxMean_ifs1(vxMean_file1.c_str());
        std::ifstream vxMean_ifs2(vxMean_file2.c_str());
        std::ifstream vyMean_ifs1(vyMean_file1.c_str());
        std::ifstream vyMean_ifs2(vyMean_file2.c_str());
        GridFunction vxMean1(&mesh1, vxMean_ifs1);
        GridFunction vxMean2(&mesh2, vxMean_ifs2);
        GridFunction vyMean1(&mesh1, vyMean_ifs1);
        GridFunction vyMean2(&mesh2, vyMean_ifs2);

        // variance
        const std::string vxVar_file1 = input_dir1+"vxVariance"
                +"_lx"+std::to_string(lx);
        const std::string vxVar_file2 = input_dir2+"vxVariance"
                +"_lx"+std::to_string(lx);
        const std::string vyVar_file1 = input_dir1+"vyVariance"
                +"_lx"+std::to_string(lx);
        const std::string vyVar_file2 = input_dir2+"vyVariance"
                +"_lx"+std::to_string(lx);
        std::ifstream vxVar_ifs1(vxVar_file1.c_str());
        std::ifstream vxVar_ifs2(vxVar_file2.c_str());
        std::ifstream vyVar_ifs1(vyVar_file1.c_str());
        std::ifstream vyVar_ifs2(vyVar_file2.c_str());
        GridFunction vxVar1(&mesh1, vxVar_ifs1);
        GridFunction vxVar2(&mesh2, vxVar_ifs2);
        GridFunction vyVar1(&mesh1, vyVar_ifs1);
        GridFunction vyVar2(&mesh2, vyVar_ifs2);

        //IncompNSObserver observer{};
        //observer(&vMean1); observer(&vVar1);
        //observer(&vMean2); observer(&vVar2);

        ComputeCauchyL2Error computeCauchyL2Error{};
        errorL2(i, 0)
                = computeCauchyL2Error.evalOnSameMesh
                (vxMean1, vyMean1, vxMean2, vyMean2);
        errorL2(i, 1)
                = computeCauchyL2Error.evalOnSameMesh
                (vxVar1, vyVar1, vxVar2, vyVar2);

        std::cout << "\nFor mean, Cauchy L2-error: "
                  << errorL2(i, 0) << std::endl;
        std::cout << "For variance, Cauchy L2-error: "
                  << errorL2(i, 1) << std::endl;
    }
    std::cout << "\n\nFor mean, Cauchy L2-error:\n"
              << errorL2.col(0).transpose() << std::endl;
    std::cout << "\nFor variance, Cauchy L2-error:\n"
              << errorL2.col(1).transpose() << std::endl;

    std::string dataPtsNames("nsamples");
    std::vector<std::string> dataOutNames;
    dataOutNames.push_back("mean");
    dataOutNames.push_back("variance");

    bool dump_output = config["dump_output"];
    if (dump_output) {
        dump_data_to_json_file(config, nsamples, errorL2,
                               dataPtsNames, dataOutNames);
    }
}

//! Computes the L2 norm
//! of mean and variance with mesh refinement
void pp_uq_meanVarNorm (const nlohmann::json config)
{
    std::string base_dir = config["base_dir"];

    std::string sub_mesh_dir = config["sub_mesh_dir"];
    std::string mesh_dir = base_dir+"/"+sub_mesh_dir+"/";

    std::string sub_in_dir = config["sub_in_dir"];
    std::string in_dir = base_dir+"/"+sub_in_dir+"/";

    int nlevels = config["num_levels_x"];
    int lx_min = config["min_level_x"];

    Eigen::VectorXi meshSize(nlevels);
    Eigen::MatrixXd normL2(nlevels, 2);
    for (int i=0; i<nlevels; i++)
    {
        int lx = lx_min + i;

        const std::string mesh_file
                = mesh_dir+"mesh_lx"+std::to_string(lx);
        Mesh mesh(mesh_file.c_str());


        // mean
        const std::string vxMean_file = in_dir+"vxMean"
                +"_lx"+std::to_string(lx);
        const std::string vyMean_file = in_dir+"vyMean"
                +"_lx"+std::to_string(lx);
        std::ifstream vxMean_ifs(vxMean_file.c_str());
        std::ifstream vyMean_ifs(vyMean_file.c_str());
        GridFunction vxMean(&mesh, vxMean_ifs);
        GridFunction vyMean(&mesh, vyMean_ifs);

        // variance
        const std::string vxVar_file = in_dir+"vxVariance"
                +"_lx"+std::to_string(lx);
        const std::string vyVar_file = in_dir+"vyVariance"
                +"_lx"+std::to_string(lx);
        std::ifstream vxVar_ifs(vxVar_file.c_str());
        std::ifstream vyVar_ifs(vyVar_file.c_str());
        GridFunction vxVar(&mesh, vxVar_ifs);
        GridFunction vyVar(&mesh, vyVar_ifs);

        ConstantCoefficient zero(0);
        double vxMeanL2 = vxMean.ComputeL2Error(zero);
        double vyMeanL2 = vyMean.ComputeL2Error(zero);
        normL2(i, 0) = std::sqrt(vxMeanL2*vxMeanL2 + vyMeanL2*vyMeanL2);

        double vxVarL2 = vxVar.ComputeL2Error(zero);
        double vyVarL2 = vyVar.ComputeL2Error(zero);
        normL2(i, 1) = std::sqrt(vxVarL2*vxVarL2 + vyVarL2*vyVarL2);

        if (config["problem_type"] == "uq_svs") {
            meshSize(i) = static_cast<int>(4*pow(2, lx));
        }
        else if (config["problem_type"] == "uq_ldc") {
            meshSize(i) = static_cast<int>(pow(2, lx));
        }
        else if (config["problem_type"] == "uq_cf") {
            meshSize(i) = static_cast<int>(32*pow(2, lx));
        }

        std::cout << "\nMean L2-norm: "
                  << normL2(i, 0) << std::endl;
        std::cout << "Variance L2-norm: "
                  << normL2(i, 1) << std::endl;
    }
    std::cout << "\n\nMean L2-norm:\n"
              << normL2.col(0).transpose() << std::endl;
    std::cout << "\nVariance L2-norm:\n"
              << normL2.col(1).transpose() << std::endl;

    std::string dataPtsNames("Nx");
    std::vector<std::string> dataOutNames;
    dataOutNames.push_back("mean");
    dataOutNames.push_back("variance");

    bool dump_output = config["dump_output"];
    if (dump_output) {
        dump_data_to_json_file(config, meshSize, normL2,
                               dataPtsNames, dataOutNames);
    }
}

//! Writes the velocity components
//! as ordered (on a uniform grid) to file
void dump_ordered_velocities
(std::string out_dir, int sampleId,
 DenseMatrix &vx, DenseMatrix &vy)
{
    fs::create_directories(out_dir);
    std::string outfile = out_dir+"/velocity_s"
            +std::to_string(sampleId)+".json";

    auto json = nlohmann::json{};
    json["sample_id"] = sampleId;

    // vx
    for (int j=0; j<vx.NumCols(); j++) {
        for (int i=0; i<vx.NumRows(); i++) {
            json["vx"][i + j*vx.NumRows()]
                    = vx(i,j);
        }
    }
    // vy
    for (int j=0; j<vy.NumCols(); j++) {
        for (int i=0; i<vy.NumRows(); i++) {
            json["vy"][i + j*vy.NumRows()]
                    = vy(i,j);
        }
    }

    // always check that you can write to the file.
    auto file = std::ofstream(outfile);
    assert(file.good());
    file << json.dump(2);
    file.close();
}

//! Writes the mean of velocity components
//! as ordered (on a uniform grid) to file
void dump_ordered_velocity_mean
(std::string out_dir, int lx,
 DenseMatrix &vxM, DenseMatrix &vyM)
{
    fs::create_directories(out_dir);
    std::string outfile = out_dir+"/velocity_mean_lx"
            +std::to_string(lx)+".json";

    auto json = nlohmann::json{};
    json["lx"] = lx;

    // vx mean
    for (int j=0; j<vxM.NumCols(); j++) {
        for (int i=0; i<vxM.NumRows(); i++) {
            json["vx"][i + j*vxM.NumRows()]
                    = vxM(i,j);
        }
    }
    // vy mean
    for (int j=0; j<vyM.NumCols(); j++) {
        for (int i=0; i<vyM.NumRows(); i++) {
            json["vy"][i + j*vyM.NumRows()]
                    = vyM(i,j);
        }
    }

    // always check that you can write to the file.
    auto file = std::ofstream(outfile);
    assert(file.good());
    file << json.dump(2);
    file.close();
}

//! Writes the variance of velocity components
//! as ordered (on a uniform grid) to file
void dump_ordered_velocity_variance
(std::string out_dir, int lx,
 DenseMatrix &vxV, DenseMatrix &vyV)
{
    fs::create_directories(out_dir);
    std::string outfile = out_dir+"/velocity_variance_lx"
            +std::to_string(lx)+".json";

    auto json = nlohmann::json{};
    json["lx"] = lx;

    // vx variance
    for (int j=0; j<vxV.NumCols(); j++) {
        for (int i=0; i<vxV.NumRows(); i++) {
            json["vx"][i + j*vxV.NumRows()]
                    = vxV(i,j);
        }
    }
    // vy variance
    for (int j=0; j<vyV.NumCols(); j++) {
        for (int i=0; i<vyV.NumRows(); i++) {
            json["vy"][i + j*vyV.NumRows()]
                    = vyV(i,j);
        }
    }

    // always check that you can write to the file.
    auto file = std::ofstream(outfile);
    assert(file.good());
    file << json.dump(2);
    file.close();
}

//! Orders the velocity components of all samples
//! and their mean and variance on a uniform grid
//! for a fixed mesh
void pp_uq_ensemble (const nlohmann::json config)
{
    int lx = config["level_x"];

    double xl = config["xl"];
    double xr = config["xr"];
    int Nx = config["num_points_x"];

    double yl = config["yl"];
    double yr = config["yr"];
    int Ny = config["num_points_y"];

    int nsamples = config["uq_num_samples"];

    std::string base_dir = config["base_dir"];
    std::string sub_in_dir = config["sub_in_dir"];
    std::string in_dir = base_dir+"/"+sub_in_dir+"/";

    const std::string mesh_file
            = in_dir+"mesh_lx"+std::to_string(lx);
    Mesh mesh(mesh_file.c_str());

    bool has_shared_vertices = true;
    PointLocator point_locator(&mesh, has_shared_vertices);

    double hx = (xr-xl)/Nx;
    double hy = (yr-yl)/Ny;

    Vector x(2);
    int init_elId = 0;
    Array<int> elIds(Nx*Ny);
    Array<IntegrationPoint> ips(Nx*Ny);
    for (int j=0; j<Ny; j++)
    {
        x(1) = yl + (j + 0.5)*hy; //y-coord
        for (int i=0; i<Nx; i++)
        {
            x(0) = xl + (i + 0.5)*hx; //x-coord

            int ii = i + j*Nx;
            std::tie (elIds[ii], ips[ii])
                    = point_locator(x, init_elId);
            init_elId = elIds[ii];
        }
    }

    // set input directory
    in_dir += "lx"+std::to_string(lx)+"/";

    // set output directory
    std::string sub_out_dir = config["sub_out_dir"];
    std::string out_dir = base_dir+"/"+sub_out_dir+"/";
    out_dir += "pp_lx"+std::to_string(lx)+"/";

    // evaluate ordered velocity ensembles
    for (int n=0; n<nsamples; n++)
    {
        // velocity
        const std::string velocity_file
                = in_dir+"velocity_s"
                +std::to_string(n);
        std::ifstream velocity_ifs(velocity_file.c_str());
        GridFunction velocity(&mesh, velocity_ifs);

        DenseMatrix vxo(Nx, Ny);
        DenseMatrix vyo(Nx, Ny);

        Vector v;
        for (int j=0; j<Ny; j++)
            for (int i=0; i<Nx; i++)
            {
                int ii = i + j*Nx;
                velocity.GetVectorValue
                        (elIds[ii], ips[ii], v);

                vxo(i,j) = v(0); // x-component
                vyo(i,j) = v(1); // y-component
            }

        dump_ordered_velocities(out_dir, n, vxo, vyo);
    }

    // evaluate ordered velocity mean and variance
    /*{
        std::string in_dir = base_dir+"/"
                +sub_in_dir+"/stats/";
        std::string out_dir = base_dir+"/"
                +sub_in_dir+"/stats/";

        // vx
        const std::string vxM_file = in_dir+"vxMean_lx"
                +std::to_string(lx);
        const std::string vxV_file = in_dir+"vxVariance_lx"
                +std::to_string(lx);
        std::ifstream vxM_ifs(vxM_file.c_str());
        std::ifstream vxV_ifs(vxV_file.c_str());
        GridFunction vxM(&mesh, vxM_ifs);
        GridFunction vxV(&mesh, vxV_ifs);
        DenseMatrix vxMo(Nx, Ny);
        DenseMatrix vxVo(Nx, Ny);

        // vy
        const std::string vyM_file = in_dir+"vyMean_lx"
                +std::to_string(lx);
        const std::string vyV_file = in_dir+"vyVariance_lx"
                +std::to_string(lx);
        std::ifstream vyM_ifs(vyM_file.c_str());
        std::ifstream vyV_ifs(vyV_file.c_str());
        GridFunction vyM(&mesh, vyM_ifs);
        GridFunction vyV(&mesh, vyV_ifs);
        DenseMatrix vyMo(Nx, Ny);
        DenseMatrix vyVo(Nx, Ny);

        for (int j=0; j<Ny; j++)
            for (int i=0; i<Nx; i++)
            {
                int ii = i + j*Nx;
                vxMo(i,j) = vxM.GetValue(elIds[ii], ips[ii]);
                vxVo(i,j) = vxV.GetValue(elIds[ii], ips[ii]);
                vyMo(i,j) = vyM.GetValue(elIds[ii], ips[ii]);
                vyVo(i,j) = vyV.GetValue(elIds[ii], ips[ii]);
            }

        dump_ordered_velocity_mean(out_dir, lx, vxMo, vyMo);
        dump_ordered_velocity_variance(out_dir, lx,
                                       vxVo, vyVo);
    }*/
}

//! Measures the velocity components
//! at specified coordinates for all the samples
//! for a fixed mesh
void pp_uq_measure_at_points (const nlohmann::json config)
{
    int lx = config["level_x"];
    int nsamples = config["uq_num_samples"];

    std::string base_dir = config["base_dir"];
    std::string sub_in_dir = config["sub_in_dir"];
    std::string in_dir = base_dir+"/"+sub_in_dir+"/";

    const std::string mesh_file
            = in_dir+"mesh_lx"+std::to_string(lx);
    Mesh mesh(mesh_file.c_str());

    std::string problem_type = config["problem_type"];

    bool has_shared_vertices = true;
    PointLocator point_locator(&mesh, has_shared_vertices);

    DenseMatrix measurePts;
    Array <int> elIds;
    Array <IntegrationPoint> ips;

    int n=0;
    // set measurement points
    if (problem_type == "uq_svs")
    {
        n = 3;
        measurePts.SetSize
                (mesh.Dimension(), n);

        // point1 (0.50, 0.25)
        measurePts(0,0) = 0.50;
        measurePts(1,0) = 0.25;

        // point1 (0.50, 0.50)
        measurePts(0,1) = 0.50;
        measurePts(1,1) = 0.50;

        // point1 (0.50, 0.75)
        measurePts(0,2) = 0.50;
        measurePts(1,2) = 0.75;
    }
    else if (problem_type == "uq_ldc")
    {
        n = 5;
        measurePts.SetSize
                (mesh.Dimension(), n);

        // point1 (0.50, 0.55)
        measurePts(0,0) = 0.50;
        measurePts(1,0) = 0.50;

        // point2 (0.25, 0.25)
        measurePts(0,1) = 0.25;
        measurePts(1,1) = 0.25;

        // point3 (0.25, 0.75)
        measurePts(0,2) = 0.25;
        measurePts(1,2) = 0.75;

        // point4 (0.75, 0.25)
        measurePts(0,3) = 0.75;
        measurePts(1,3) = 0.25;

        // point5 (0.75, 0.75)
        measurePts(0,4) = 0.75;
        measurePts(1,4) = 0.75;
    }
    else if (problem_type == "uq_cf")
    {
        n = 4;
        measurePts.SetSize
                (mesh.Dimension(), n);

        // point3 (1, 0.1)
        measurePts(0,0) = 1;
        measurePts(1,0) = 0.1;

        // point4 (1, 0.2)
        measurePts(0,1) = 1;
        measurePts(1,1) = 0.2;

        // point5 (1, 0.3)
        measurePts(0,2) = 1;
        measurePts(1,2) = 0.3;

        // point6 (1, 0.4)
        measurePts(0,3) = 1;
        measurePts(1,3) = 0.4;
    }
    // find elements corresponding
    // to measurement points
    elIds.SetSize(n);
    ips.SetSize(n);
    int init_elId = 0;
    for (int k=0; k<n; k++) {
        Vector point;
        measurePts.GetColumn(k, point);
        std::tie (elIds[k], ips[k])
                = point_locator(point, init_elId);
        init_elId = elIds[k];
    }

    // set input directory
    in_dir += "lx"+std::to_string(lx)+"/";

    // evaluate at points
    int s = measurePts.NumCols();
    DenseMatrix allData(2*s, nsamples);
    for (int n=0; n<nsamples; n++)
    {
        // velocity
        const std::string velocity_file
                = in_dir+"velocity_s"
                +std::to_string(n);
        std::ifstream velocity_ifs(velocity_file.c_str());
        GridFunction velocity(&mesh, velocity_ifs);

        Vector data;
        allData.GetColumnReference(n, data);

        Vector v;
        for (int k=0; k<s; k++)
        {
            velocity.GetVectorValue(elIds[k], ips[k], v);

            data(k) = v(0);   // x-component
            data(k+s) = v(1); // y-component
        }
    }

    // dump output
    std::string sub_out_dir = config["sub_out_dir"];
    std::string out_dir = base_dir+"/"+sub_out_dir+"/";
    std::string solName_suffix = "_lx"+std::to_string(lx);
    {
        std::vector<std::string> dataPtsNames;
        dataPtsNames.push_back("vx");
        dataPtsNames.push_back("vy");

        auto json = nlohmann::json{};
        json["nsamples"] = nsamples;
        for (int j=0; j<nsamples; j++) {
            for (int i=0; i<s; i++) {
                json[dataPtsNames[0]]
                        [static_cast<unsigned int>
                        (j + i*nsamples)]
                        = allData(i, j);
                json[dataPtsNames[1]]
                        [static_cast<unsigned int>
                        (j + i*nsamples)]
                        = allData(i+s, j);
            }
        }

        std::string outfile
                = out_dir+"measuredData"
                +solName_suffix+".json";
        auto file = std::ofstream(outfile);
        // always check that you can write to the file.
        assert(file.good());

        file << json.dump(2);
        file.close();
    }
}


//! Writes the velocity components and pressure
//! to .txt file
void dump_dataSet
(std::string out_dir,
 int sampleId, Vector& omega,
 double t, Vector& x, Vector& y,
 DenseMatrix &vx, DenseMatrix &vy, DenseMatrix &p,
 bool append)
{
    fs::create_directories(out_dir);
    std::string outfile = out_dir+"/uqIncompNS_cf_s"
            +std::to_string(sampleId)+".txt";

    std::ofstream file;
    if (append) {
        file.open(outfile, std::ios_base::app);
    } else {
        file.open(outfile);
    }
    assert(file.good());

    for (int j=0; j<vx.NumCols(); j++) {
        for (int i=0; i<vx.NumRows(); i++)
        {
            file << t << "  " << x(i) << "  " << y(j) << "  ";
            for (int k = 0; k < omega.Size(); k++)
                file << omega(k) << "  ";
            file << vx(i,j) << "  " << vy(i,j) << "  "
                 << p(i,j) << std::endl;
        }
    }

    file.close();
}

//! Measures the velocity components and pressure
//! at specified coordinates for all the samples
//! for a fixed mesh
void pp_uq_cf_prepare_dataSet (const nlohmann::json config)
{
    double xl = config["xl"];
    double xr = config["xr"];
    int Nx = config["num_points_x"];

    double yl = config["yl"];
    double yr = config["yr"];
    int Ny = config["num_points_y"];

    int lx = config["level_x"];
    int nsamples = config["uq_num_samples"];

    double endTime = config["end_time"];
    double Nt = config["num_time_steps"];
    double ht = endTime/Nt;

    int maxTimeId = config["uq_max_timeId"];
    int jumpTimeId = config["uq_jump_timeId"];

    std::string base_dir = config["base_dir"];
    std::string sub_in_dir = config["sub_in_dir"];
    std::string in_dir = base_dir+"/"+sub_in_dir+"/";

    // mesh
    const std::string mesh_file
            = in_dir+"mesh_lx"+std::to_string(lx);
    std::cout << mesh_file << std::endl;
    Mesh mesh(mesh_file.c_str());

    // samples
    const std::string samples_file
            = in_dir+"samples_lx"+std::to_string(lx)+".json";
    auto samplesConfig = get_global_config(samples_file);
    int nparams = samplesConfig["num_params"];
    int allNsamples = samplesConfig["num_samples"];
    assert(nsamples <= allNsamples);
    std::vector<double> omegas_ = samplesConfig["omegas"];
    DenseMatrix omegas(omegas_.data(), nparams, nsamples);

    /*// y-coords
    double hy = (yr-yl)/(Ny-1);
    Vector y(Ny);
    double ym = (yl + yr)/2;
    for (int j=0; j<Ny; j++)
    {
        double y_ = yl + j*hy;
        double a, b, c;
        if (y_ < ym) {
            a = 1./(ym-yl);
            b = -2*a*yl;
            c = yl*(1 + a*yl);
        } else {
            a = 1./(ym-yr);
            b = -2*a*yr;
            c = yr*(1 + a*yr);
        }
        y(j) = a*y_*y_ + b*y_ + c;
    }*/

    // y-coords
    Ny = 161;
    Vector y(Ny);
    for (int j=0; j<50; j++)
    {
        double h = 0.1/50;
        y(j) = j*h;
        y(Ny-j-1) = 0.5 - j*h;
    }
    for (int j=0; j<=60; j++)
    {
        y(j+50) = 0.1 + j*(0.3/60);
    }
    y.Print();

    // x-coords
    double hx = (xr-xl)/(Nx-1);
    Vector x(Nx);
    for (int i=0; i<Nx; i++) {
        x(i) = xl + i*hx;
    }

    // point locator
    bool has_shared_vertices = true;
    PointLocator point_locator(&mesh, has_shared_vertices);

    // find points
    int init_elId = 0;
    Vector coords(2);
    Array<int> elIds(Nx*Ny);
    Array<IntegrationPoint> ips(Nx*Ny);
    for (int j=0; j<Ny; j++) {
        coords(1) = y(j);
        for (int i=0; i<Nx; i++) {
            coords(0) = x(i);
            int ii = i + j*Nx;
            std::tie (elIds[ii], ips[ii])
                    = point_locator(coords, init_elId);
            init_elId = elIds[ii];
        }
    }

    // set input directory
    in_dir += "lx"+std::to_string(lx)+"/";

    // set output directory
    std::string sub_out_dir = config["sub_out_dir"];
    std::string out_dir = base_dir+"/"+sub_out_dir+"/";
    out_dir += "dataSet_lx"+std::to_string(lx)+"/";

    // evaluate velocity and pressure at given grid points
    for (int s=0; s<nsamples; s++)
    {
        for (int tId=0; tId<=maxTimeId; tId+=jumpTimeId)
        {
            // velocity
            const std::string velocity_file
                    = in_dir+"velocity_s"
                    +std::to_string(s)
                    +"_tId"+std::to_string(tId);
            std::cout << velocity_file << std::endl;
            std::ifstream velocity_ifs(velocity_file.c_str());
            GridFunction velocity(&mesh, velocity_ifs);

            // pressure
            const std::string pressure_file
                    = in_dir+"pressure_s"
                    +std::to_string(s)
                    +"_tId"+std::to_string(tId);
            std::ifstream pressure_ifs(pressure_file.c_str());
            std::cout << pressure_file << std::endl;
            GridFunction pressure(&mesh, pressure_ifs);

            DenseMatrix vx(Nx, Ny);
            DenseMatrix vy(Nx, Ny);
            DenseMatrix p(Nx, Ny);

            Vector v;
            double t = ht*tId;
            for (int j=0; j<Ny; j++)
                for (int i=0; i<Nx; i++)
                {
                    int ii = i + j*Nx;
                    velocity.GetVectorValue
                            (elIds[ii], ips[ii], v);

                    vx(i,j) = v(0); // vx
                    vy(i,j) = v(1); // vy
                    p(i,j) = pressure.GetValue(elIds[ii], ips[ii]);
                }

            Vector omega;
            omegas.GetColumn(s, omega);
            if (tId == 0) {
                dump_dataSet(out_dir, s, omega,
                             t, x, y, vx, vy, p, false);
            } else {
                dump_dataSet(out_dir, s, omega,
                             t, x, y, vx, vy, p, true);
            }
        }
    }
}


//! Calls the post-processing routines
//! for incompressible Navrier-Stokes
int main(int argc, char *argv[])
{   
    // Read config json
    auto config = get_global_config(argc, argv);
    const std::string run = config["run"];
    
    if (run == "post_process")
    {
        const std::string model = config["model"];
        const std::string problem_type
                = config["problem_type"];

        if (model == "incompNS") {
            if (problem_type == "ldc") {
                pp_pincompNS_ldc(std::move(config));
            }
        }
    }
    else if (run == "pp_convg") {
        pp_uq_convg(std::move(config));
    }
    else if (run == "pp_convg_wrt_samples") {
        pp_uq_convg_wrt_samples(std::move(config));
    }
    else if (run == "pp_norm_meanVar") {
        pp_uq_meanVarNorm(std::move(config));
    }
    else if (run == "pp_ensemble") {
        pp_uq_ensemble(std::move(config));
    }
    else if (run == "pp_measure") {
        pp_uq_measure_at_points(std::move(config));
    }
    else if (run == "pp_prepare_dataSet") {
        pp_uq_cf_prepare_dataSet(std::move(config));
    }

    return 0;
}

// End of file
