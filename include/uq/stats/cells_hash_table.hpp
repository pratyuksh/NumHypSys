#ifndef UQ_CELLS_HASH_TABLE_HPP
#define UQ_CELLS_HASH_TABLE_HPP

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "mfem.hpp"

using namespace mfem;


//! Defines an element type of the hash list
struct HashTableElement
{
    HashTableElement () {
        velocity.setZero();
    }

    void display() {
        std::cout << id << ", "
                  << weight << ", "
                  << coords.transpose() << ", "
                  << velocity.transpose();
    }

    int id;
    double weight;
    Eigen::Vector2d coords;
    Eigen::Vector2d velocity;
};
typedef struct HashTableElement tabEl;


//! Defines the hash table
class CellsHashTable
{
public:
    //! Constructor
    CellsHashTable(int Nx, double xl, double xr,
                   int Ny, double yl, double yr);

    //! Destructor
    ~ CellsHashTable() { destroy(); }

    //! Hash function to map key to values
    int hash_function(Eigen::Vector2d&);

    //! Searches key in the hash table
    //! returns the cell indices
    std::pair<int, int> search(Eigen::Vector2d&);

    //! Inserts a key in the hash table
    void insert(int, double, Eigen::Vector4d&);

    //! Modifies a key in the hash table
    int modify(int, Eigen::Vector4d&);

    //! Gives cell number from coords
    inline int get_cell_number(int i, int j)
    {
        return std::move(i + j*m_Nx);
    }

    //! Gives cell coords from number
    inline std::pair<int, int> get_cell_indices(int index)
    {
        int i = index%m_Nx;
        int j = index/m_Nx;
        return {std::move(i), std::move(j)};
    }

    //! Displays the hash table
    void display()
    {
        for (int i = 0; i < m_numCells; i++) {
            std::cout << i;
            for (tabEl data : table[i])
            {
                std::cout << " --> ";
                data.display();
            }
        std::cout << std::endl;
      }
    }

    //! Deletes the hash table
    void destroy()
    {
        for (int i=0; i<m_numCells; i++) {
            table[i].clear();
        }
    }

    //! Operators to access table
    std::list<tabEl> operator()(int i, int j) {
        return get_data(i,j);
    }

    std::list<tabEl> operator()(int index) {
        return get_data(index);
    }

    //! Returns table
    std::list<tabEl> *get_table() {
        return table;
    }

    //! Returns the list of data for cell i,j
    //! data = (coords, area)
    std::list<tabEl> get_data(int i, int j) {
        return table[get_cell_number(i,j)];
    }

    //! Returns the list of data for cell i,j
    //! data = (coords, area)
    std::list<tabEl> get_data(int index) {
        return table[index];
    }

    //! Returns the total number of elements
    inline int get_numElements() {
        return get_numElements(0, m_Nx-1, 0, m_Ny-1);
    }

    //! Returns the total number of elements
    inline int get_numElements(int iStart, int iEnd,
                               int jStart, int jEnd)
    {
        int numElements = 0;
        for (int i=iStart; i<=iEnd; i++)
            for (int j=jStart; j<=jEnd; j++) {
                numElements += get_data_size(i,j);
            }
        return numElements;
    }

    //! Returns the number of elements in the list
    //! for cell i,j
    inline int get_data_size(int i, int j) {
        return table[get_cell_number(i,j)].size();
    }

    //! Returns the number of cells along x-axis
    inline int get_numCellsInX() {
        return m_Nx;
    }

    //! Returns the number of cells along y-axis
    inline int get_numCellsInY() {
        return m_Ny;
    }

    //! Returns x-axis left bound
    inline double get_xLeft() {
        return m_xl;
    }

    //! Returns x-axis right bound
    inline double get_xRight() {
        return m_xr;
    }

    //! Returns y-axis left bound
    inline double get_yLeft() {
        return m_yl;
    }

    //! Returns y-axis right bound
    inline double get_yRight() {
        return m_yr;
    }

private:
    int m_Nx;
    double m_xl, m_xr;

    int m_Ny;
    double m_yl, m_yr;

    double m_tol;

    int m_numCells;
    std::list<tabEl> *table;
};


//! Makes the hash table for a given mesh
std::shared_ptr<CellsHashTable> make_hashTable
(std::shared_ptr<Mesh>& mesh,
 int Nx, double xl, double xr,
 int Ny, double yl, double yr);

//! Updates the velocity data in hash table
void update_hashTable
(std::shared_ptr<CellsHashTable>&,
 std::shared_ptr<Mesh>& mesh,
 GridFunction *vx, GridFunction *vy);

//! Updates the velocity fluctuations data in hash table
void update_hashTable
(std::shared_ptr<CellsHashTable>&,
 std::shared_ptr<Mesh>& mesh,
 GridFunction *vx, GridFunction *vxMean,
 GridFunction *vy, GridFunction *vyMean);

#endif /// UQ_CELLS_HASH_TABLE_HPP
