#include "../../../include/uq/stats/cells_hash_table.hpp"
#include "../../../include/mymfem/utilities.hpp"
#include <iostream>


//! Constructor
CellsHashTable
:: CellsHashTable (int Nx, double xl, double xr,
                   int Ny, double yl, double yr)
    : m_Nx(Nx), m_xl(xl), m_xr(xr),
      m_Ny(Ny), m_yl(yl), m_yr(yr),
      m_numCells (Nx*Ny)
{
    assert(m_xr > m_xl);
    assert(m_yr > m_yl);

    m_tol = 1E-8;

    table = new std::list<tabEl>
            [static_cast<unsigned int>(m_numCells)];
}

//! Hash function to map key to values
int CellsHashTable
:: hash_function(Eigen::Vector2d& coords)
{
    // add some tolerance for fail safe
    double sx = (coords(0) - m_xl)/(m_xr - m_xl);
    double sy = (coords(1) - m_yl)/(m_yr - m_yl);
    auto i = int(std::floor(m_Nx*(sx - m_tol)));
    auto j = int(std::floor(m_Ny*(sy - m_tol)));

    // if given coordinate is outside the domain
    if ((i < 0 || i >= m_Nx) ||
            (j < 0 || j >= m_Ny)) {
        //std::cout << "\n" << i << "\t"
        //          << j << "\t"
        //          << coords.transpose() << std::endl;
        //std::cout << "\nGiven coordinate "
        //             "outside the domain!\n";
        return -1;
    }

    return std::move(get_cell_number(i,j));
}

//! Searches key in the hash table
std::pair<int, int> CellsHashTable
:: search(Eigen::Vector2d& coords)
{
    int index = hash_function(coords);
    if (index < 0) {
        std::cout << "\nGiven coordinate "
                     "outside the domain!\n";
        abort();
    }
    return get_cell_indices(index);
}

//! Inserts a key in the hash table
//! id: index of point
//! weight: weight/area associated with point
//! data: (coords, (vx,vy))
void CellsHashTable
:: insert(int id, double weight, Eigen::Vector4d& data)
{
    Eigen::Vector2d coords(data.data());
    int index = hash_function(coords);

    if (index >= 0)
    {
        HashTableElement el;
        el.id = id;
        el.weight = weight;
        el.coords = coords;

        table[index].push_back(el);
    }
}

//! Modifies a key in the hash table
//! id: index of point
//! data: (coords, (vx,vy))
int CellsHashTable
:: modify(int id, Eigen::Vector4d& data)
{
    Eigen::Vector2d coords(data.data());
    Eigen::Vector2d velocity(data.data()+2);
    int index = hash_function(coords);
    int info = -1;

    if (index >= 0)
    {
        std::list <tabEl> :: iterator el;
        for (el = table[index].begin();
                 el != table[index].end(); el++) {
          if (el->id == id)
            break;
        }

        // modify velocity
        if (el != table[index].end()) {
            el->velocity = velocity;
            info = 1;
        }
    }

    return info;
}


//! Creates the hash table for a given mesh
std::shared_ptr<CellsHashTable>
make_hashTable (std::shared_ptr<Mesh>& mesh,
                int Nx, double xl, double xr,
                int Ny, double yl, double yr)
{
    std::shared_ptr<CellsHashTable> cellsHashTable
            = std::make_shared<CellsHashTable>
            (Nx, xl, xr, Ny, yl, yr);

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
(std::shared_ptr<CellsHashTable>& cellsHashTable,
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
        cellsHashTable->modify(i, data);
    }
}

//! Updates the velocity fluctuations data in hash table
//! Assumes vx and vy are defined on the mesh provided
//! and are mesh element-wise constant
void update_hashTable
(std::shared_ptr<CellsHashTable>& cellsHashTable,
 std::shared_ptr<Mesh>& mesh,
 GridFunction *vx, GridFunction *vxMean,
 GridFunction *vy, GridFunction *vyMean)
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
        v(0) = vx->GetValue(i, ip) - vxMean->GetValue(i, ip);
        v(1) = vy->GetValue(i, ip) - vyMean->GetValue(i, ip);
        cellsHashTable->modify(i, data);
    }
}

// End of file
