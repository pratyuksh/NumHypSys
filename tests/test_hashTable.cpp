#include <gtest/gtest.h>

#include "mfem.hpp"
using namespace mfem;

#include <iostream>
#include <Eigen/Core>

#include "../include/core/config.hpp"
#include "../include/mymfem/utilities.hpp"
#include "../include/uq/stats/cells_hash_table.hpp"


//! Test creating a cells hash table for a given mesh
//! Unit Square Mesh, coarse
TEST(CellsHashTable, test1)
{
    int lx = 2;
    const std::string mesh_file
            = "../input/poisson_smooth_unitSquare/mesh_lx"
            +std::to_string(lx);
    
    int Nx = 4;
    double xl = 0;
    double xr = 1;
    
    int Ny = 4;
    double yl = 0;
    double yr = 1;

    //std::cout << mesh_file << std::endl;
    std::shared_ptr<Mesh> mesh
            = std::make_shared<Mesh>(mesh_file.c_str());
    
    auto cellsHashTable
            = make_hashTable(mesh, Nx, xl, xr, Ny, yl, yr);
    cellsHashTable->display();

    int numElements = cellsHashTable->get_numElements();
    int true_numElements = mesh->GetNE();
    ASSERT_EQ(numElements, true_numElements);

    Eigen::Vector2d coords;
    coords(0) = 0.51;
    coords(1) = 0.50;
    int true_idx = 2;
    int true_idy = 1;
    int idx, idy;
    std::tie(idx, idy) = cellsHashTable->search(coords);
    std::cout << "Coords " << coords(0) << "," << coords(1)
              << " are in cell "
              << idx << "," << idy << std::endl;
    ASSERT_EQ(idx, true_idx);
    ASSERT_EQ(idy, true_idy);
}

TEST(CellsHashTable, test2)
{
    const std::string mesh_file
            = "../meshes/channel_L1pt5/refined"
              "/tri_mesh_l0.msh";

    int Nx = 10;
    double xl = 0;
    double xr = 1.5;

    int Ny = 10;
    double yl = 0;
    double yr = 0.5;

    //std::cout << mesh_file << std::endl;
    std::shared_ptr<Mesh> mesh
            = std::make_shared<Mesh>(mesh_file.c_str());

    auto cellsHashTable
            = make_hashTable(mesh, Nx, xl, xr, Ny, yl, yr);

    Eigen::Vector2d coords;
    coords(0) = 0.51;
    coords(1) = 0.49;
    int idx, idy;
    std::tie(idx, idy) = cellsHashTable->search(coords);
    std::cout << "Coords " << coords(0) << "," << coords(1)
              << " are in cell "
              << idx << "," << idy << std::endl;

    int numElements = cellsHashTable->get_numElements();
    int true_numElements = mesh->GetNE();
    ASSERT_EQ(numElements, true_numElements);
}

// End of file
