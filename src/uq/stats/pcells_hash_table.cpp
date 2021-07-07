#include "../../../include/uq/stats/pcells_hash_table.hpp"
#include "../../../include/mymfem/utilities.hpp"
#include <iostream>


//! Constructor
ParCellsHashTable
:: ParCellsHashTable (MPI_Comm& comm,
                      int Nx, double xl, double xr,
                      int Ny, double yl, double yr)
    : m_Nx(Nx), m_xl(xl), m_xr(xr),
      m_Ny(Ny), m_yl(yl), m_yr(yr)
{
    // check input bounds
    assert(m_xr > m_xl);
    assert(m_yr > m_yl);

    // communicator size
    int nprocs;
    MPI_Comm_size(comm, &nprocs);

    // create Cartesian communicator group
    int dims[2] = {0,0};
    int periodic[2] = {false, false};
    int coords[2] = {0,0};

    MPI_Dims_create(nprocs, 2, dims);
    MPI_Cart_create(comm, 2, dims, periodic, true,
                    &m_cartComm);
    MPI_Comm_rank(m_cartComm, &m_cartRank);
    MPI_Cart_coords(m_cartComm, m_cartRank, 2, coords);
    if (m_cartRank == 0) {
        std::cout << "\nProcessor Cartesian grid: "
                  << dims[0] << ","
                  << dims[1] << std::endl;
        /*std::cout << "\nGlobal x-dim: "
                  << xl << "," << xr << "," << Nx
                  << "\tGlobal y-dim: "
                  << yl << "," << yr << "," << Ny
                  << "\nGrid: "
                  << dims[0] << ","
                  << dims[1] << std::endl;*/
    }
    /*std::cout << "\nCart rank: " << m_cartRank
              << ", Coords: "
              << coords[0] << ","
              << coords[1] << std::endl;*/

    // define local hash table parameters

    // total number of cells to the left of current proc
    // in x-dim
    int sumNxl_ = 0;
    for (int i=0; i<coords[0]; i++) {
        sumNxl_ += Nx/dims[0];
        if (i < Nx%dims[0]) { sumNxl_++; }
    }

    // number of cells for current proc in x-dim
    int Nx_ = Nx/dims[0];
    if (coords[0] < Nx%dims[0]) { Nx_++; }

    // cell bounds in x-dim
    double xl_ = xl + sumNxl_*(xr-xl)/Nx;
    double xr_ = xl_ + Nx_*(xr-xl)/Nx;

    // extend with offsets for interior boundaries in x-dim
    m_iStart = 0;
    m_iEnd = Nx_-1;
    if (coords[0] > 0) {
        Nx_++;
        xl_ -= (xr-xl)/Nx;
        m_iStart++;
        m_iEnd++;
    }
    if (coords[0] < dims[0]-1) {
        Nx_++;
        xr_ += (xr-xl)/Nx;
    }

    // total number of cells to the left of current proc
    // in y-dim
    int sumNyl_ = 0;
    for (int j=0; j<coords[1]; j++) {
        sumNyl_ += Ny/dims[1];
        if (j < Ny%dims[1]) { sumNyl_++; }
    }

    // number of cells for current proc in y-dim
    int Ny_ = Ny/dims[1];
    if (coords[1] < Ny%dims[1]) { Ny_++; }

    // cell bounds in y-dim
    double yl_ = yl + sumNyl_*(yr-yl)/Ny;
    double yr_ = yl_ + Ny_*(yr-yl)/Ny;

    // extend with offsets for interior boundaries in y-dim
    m_jStart = 0;
    m_jEnd = Ny_-1;
    if (coords[1] > 0) {
        Ny_++;
        yl_ -= (yr-yl)/Ny;
        m_jStart++;
        m_jEnd++;
    }
    if (coords[1] < dims[1]-1) {
        Ny_++;
        yr_ += (yr-yl)/Ny;
    }

    /*std::cout << "\nCart rank: " << m_cartRank
              << "\tx-dim: "
              << xl_ << "," << xr_ << "," << Nx_
              << "\ty-dim: "
              << yl_ << "," << yr_ << "," << Ny_
              << std::endl;*/

    m_cellsHashTable = std::make_shared<CellsHashTable>
            (Nx_, xl_, xr_, Ny_, yl_, yr_);
}


//! Searches key in the hash table
std::pair<int, int> ParCellsHashTable
:: search(Eigen::Vector2d& coords)
{
    int localIndex
            = m_cellsHashTable->hash_function(coords);

    int globalIndex = 0;
    MPI_Allreduce(&localIndex, &globalIndex, 1,
                  MPI_INT, MPI_MAX, m_cartComm);

    if (globalIndex < 0) {
        std::cout << "\nGiven coordinate "
                     "outside the domain!\n";
        abort();
    }

    int xCoords_ = -1, yCoords_ = -1;
    if (localIndex >= 0) {
        std::tie(xCoords_, yCoords_) = m_cellsHashTable
                ->get_cell_indices(localIndex);
    }
    int xCoords, yCoords;
    MPI_Allreduce(&xCoords_, &xCoords, 1,
                  MPI_INT, MPI_MAX, m_cartComm);
    MPI_Allreduce(&yCoords_, &yCoords, 1,
                  MPI_INT, MPI_MAX, m_cartComm);

    return {xCoords, yCoords};
}

//! Inserts a key in the parallel hash table
//! id: index of point
//! weight: weight/area associated with point
//! data: (coords, (vx,vy))
void ParCellsHashTable
:: insert(int id, double weight, Eigen::Vector4d& data)
{
    m_cellsHashTable->insert(id, weight, data);
}

//! Modifies a key in the hash table
//! id: index of point
//! data: (coords, (vx,vy))
int ParCellsHashTable
:: modify(int id, Eigen::Vector4d& data)
{
    int info_ = m_cellsHashTable->modify(id, data);
    int info;
    MPI_Allreduce(&info_, &info, 1,
                  MPI_INT, MPI_MAX, m_cartComm);
    return info;
}


//! Makes parallel hash table for a given mesh
std::shared_ptr<ParCellsHashTable>
make_hashTable
(MPI_Comm& comm,
 std::shared_ptr<Mesh>& mesh,
 int Nx, double xl, double xr,
 int Ny, double yl, double yr)
{
    // create hash table local to the processor
    std::shared_ptr<ParCellsHashTable> cellsHashTable
            = std::make_shared<ParCellsHashTable>
            (comm, Nx, xl, xr, Ny, yl, yr);

    Eigen::Vector4d data(4);
    data.setZero();

    Vector coords(data.data(), 2);
    for(int i=0; i<mesh->GetNE(); i++)
    {
        get_element_center(mesh, i, coords); // coords
        double weight = mesh->GetElementVolume(i); // area
        if (isnan(weight)) {
            std::cout << "Bad mesh element: "
                      << i << std::endl;
        }
        cellsHashTable->insert(i, weight, data);
    }

    return cellsHashTable;
}

//! Updates the velocity data in given hash table
//! Assumes vx and vy are defined on the mesh provided
//! and are mesh element-wise constant
void update_hashTable
(std::shared_ptr<ParCellsHashTable>& pcellsHashTable,
 std::shared_ptr<Mesh>& mesh,
 GridFunction *vx, GridFunction *vy)
{
    Eigen::Vector4d data(4);
    data.setZero();

    Vector coords(data.data(), 2);
    Vector v(data.data()+2, 2);
    for(int i=0; i<mesh->GetNE(); i++)
    {
        get_element_center(mesh, i, coords);
        int geom = mesh->GetElementBaseGeometry(i);
        auto ip = Geometries.GetCenter(geom);
        v(0) = vx->GetValue(i, ip);
        v(1) = vy->GetValue(i, ip);
        int info = pcellsHashTable->modify(i, data);
        if (info < 0) {
            std::cout << "Error when updating "
                         "the parallel hash table"
                      << std::endl;
        }
    }
}


// End of file
