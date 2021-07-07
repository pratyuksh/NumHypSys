#ifndef UQ_PCELLS_HASH_TABLE_HPP
#define UQ_PCELLS_HASH_TABLE_HPP

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "mfem.hpp"

#include "cells_hash_table.hpp"

using namespace mfem;


//! Defines parallel hash table
class ParCellsHashTable
{
public:
    //! Constructor
    ParCellsHashTable(MPI_Comm& comm,
                      int Nx, double xl, double xr,
                      int Ny, double yl, double yr);

    //! Destructor
    ~ ParCellsHashTable() {}

    //! Searches key in the hash table
    //! returns the cell indices
    std::pair<int, int> search(Eigen::Vector2d&);

    //! Inserts a key in the hash table
    void insert(int, double, Eigen::Vector4d&);

    //! Modifies a key in the hash table
    int modify(int, Eigen::Vector4d&);

    //! Displays the hash table
    void display()
    {
        if (m_cartRank == 0) { m_cellsHashTable->display(); }
    }

    //! Returns the total number of elements
    inline int get_numElements() {

        int numElements_ = m_cellsHashTable
                ->get_numElements(m_iStart, m_iEnd,
                                  m_jStart, m_jEnd);

        int numElements = 0;
        MPI_Allreduce(&numElements_, &numElements, 1,
                      MPI_INT, MPI_SUM, m_cartComm);

        return numElements;
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

    //! Returns the communicator
    inline MPI_Comm get_comm() {
        return m_cartComm;
    }

    //! Returns local hash table
    inline std::shared_ptr<CellsHashTable>
    get_localHashTable() {
        return m_cellsHashTable;
    }

private:
    MPI_Comm m_cartComm;
    int m_cartRank;

    int m_Nx;
    double m_xl, m_xr;
    int m_iStart, m_iEnd;

    int m_Ny;
    double m_yl, m_yr;
    int m_jStart, m_jEnd;

    std::shared_ptr<CellsHashTable> m_cellsHashTable;
};


//! Makes parallel hash table for a given mesh
std::shared_ptr<ParCellsHashTable>
make_hashTable
(MPI_Comm& comm,
 std::shared_ptr<Mesh>& mesh,
 int Nx, double xl, double xr,
 int Ny, double yl, double yr);

//! Updates the velocity data in hash table
void update_hashTable
(std::shared_ptr<ParCellsHashTable>&,
 std::shared_ptr<Mesh>& mesh,
 GridFunction *vx, GridFunction *vy);


#endif /// UQ_PCELLS_HASH_TABLE_HPP
