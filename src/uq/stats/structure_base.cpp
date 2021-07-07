#include "../../../include/uq/stats/structure_base.hpp"
#include <iostream>
#include "mpi.h"


//! Evaluates the structure function on structured grid
Eigen::VectorXd StructureBase
:: eval(const Eigen::MatrixXd &U,
        Eigen::VectorXi stencil, int p)
{
    auto Ns = stencil.size();
    int max_stencil = stencil(Ns-1);

    auto Nx = U.rows()/2 - 2*max_stencil;
    auto Ny = U.cols() - 2*max_stencil;
    //std::cout << Nx << "\t" << Ny << "\t"
    //          << max_stencil << std::endl;

    Eigen::VectorXd sval_raw(stencil.size());
    Eigen::VectorXd sval(stencil.size());

    // compute outer shells of stencil
    sval_raw.setZero();
    for (int j=0; j<Ny; j++) {
        for (int i=0; i<Nx; i++) {
            for (int ns=0; ns<Ns; ns++) {
                sval_raw(ns)
                        += evalAtPoint(U,
                                       i+max_stencil,
                                       j+max_stencil,
                                       stencil(ns), p);
            }
        }
    }

    // post-process
    double factor = 1;
    for (int ns=0; ns<Ns; ns++) {
        factor = 1./std::pow(2*stencil(ns)+1,2);
        sval(ns) = sval_raw.head(ns+1).sum()*factor;
    }

    return sval;
}

//! Evaluates the structure function for a point/cell
//! of the structured grid
double StructureBase
:: evalAtPoint(const Eigen::MatrixXd &U,
               int i, int j,
               int stencil, int p)
{
    double sval = 0;

    int istart = i - stencil;
    int iend = i + stencil;

    int jstart = j - stencil;
    int jend = j + stencil;

    // solution at i,j
    auto Uij = getSol(U, i, j);

    // bottom of the shell
    int l = jstart;
    for (int k=istart; k<=iend; k++) {
        auto Ukl = getSol(U, k, l);
        sval += functional(Uij, Ukl, p);
    }

    // top of the shell
    l = jend;
    for (int k=istart; k<=iend; k++) {
        auto Ukl = getSol(U, k, l);
        sval += functional(Uij, Ukl, p);
    }

    // left of the shell
    int k = istart;
    for (int l=jstart+1; l<jend; l++) {
        auto Ukl = getSol(U, k, l);
        sval += functional(Uij, Ukl, p);
    }

    // right of the shell
    k = iend;
    for (int l=jstart+1; l<jend; l++) {
        auto Ukl = getSol(U, k, l);
        sval += functional(Uij, Ukl, p);
    }

    return sval;
}

//! Evaluates functional
double StructureBase
::functional(const Eigen::Vector2d& Uij,
             const Eigen::Vector2d& Ukl,
             int p)
{
    auto val = std::pow(std::abs(Uij[0] - Ukl[0]), p);
    val += std::pow(std::abs(Uij[1] - Ukl[1]), p);
    return val;
}

//! Returns solution U(k,l)
Eigen::Vector2d StructureBase
:: getSol(const Eigen::MatrixXd &U, int k, int l)
{
    Eigen::Vector2d Ukl;
    Ukl(0) = U(2*k,l);
    Ukl(1) = U(2*k+1,l);
    return Ukl;
}


//! Evaluates the structure function for a cell
//! of the hash table
Eigen::VectorXd StructureBase
:: evalInCell
(std::shared_ptr<CellsHashTable>& cellsHashTable,
 int i, int j, Eigen::VectorXd offsets, int p)
{
    auto M = offsets.size()-1;
    Eigen::VectorXd sval(M);
    sval.setZero();

    Eigen::VectorXd sval_raw(M);
    Eigen::VectorXd sumW(M);

    // hash table object
    auto table = cellsHashTable->get_table();

    int curCellId = cellsHashTable->get_cell_number(i, j);
    std::list <tabEl> :: iterator curPt;
    for (curPt = table[curCellId].begin();
             curPt != table[curCellId].end(); curPt++)
    {
        sval_raw.setZero(); sumW.setZero();

        auto curW = curPt->weight;
        auto curCoords = curPt->coords;
        auto curV = curPt->velocity;

        // loop over all neighbouring cells
        for (int jj=j-1; jj<=j+1; jj++)
        {
            for (int ii=i-1; ii<=i+1; ii++)
            {
                // neighbour cell id
                int nbCellId = cellsHashTable->
                        get_cell_number(ii, jj);

                // loop over all points in the neighbor
                std::list <tabEl> :: iterator nbPt;
                for (nbPt = table[nbCellId].begin();
                     nbPt != table[nbCellId].end();
                     nbPt++)
                {
                    auto nbW = nbPt->weight;
                    auto nbCoords = nbPt->coords;
                    auto nbV = nbPt->velocity;

                    // loop over offsets
                    for (int m=0; m<M; m++)
                    {
                        double minOffset = offsets(m);
                        double maxOffset = offsets(m+1);
                        bool inside = neighbour_in_shell
                                (curCoords, nbCoords,
                                 minOffset, maxOffset);

                        if (inside)
                        {
                            // compute functional
                            //sval_raw(m) += functional
                            //        (curV, nbV, p);
                            sval_raw(m) += nbW*functional
                                    (curV, nbV, p);
                            sumW(m) += nbW;
                            break;
                        }
                    }

                    /*int m;
                    bool inside;
                    std::tie(inside, m)
                            = neighbour_in_shells
                            (curCoords, nbCoords, offsets);
                    if (inside)
                    {
                        sval_raw(m) += nbW*functional
                                (curV, nbV, p);
                        sumW(m) += nbW;
                    }*/
                }
            }
        }
        // post-process shell computations
        for (int m=0; m<M; m++) {
            // exclude the case when raw data not updated
            if (sumW(m) > 1E-12) {
                sval(m) += curW*(sval_raw.head(m+1).sum()
                        /sumW.head(m+1).sum());
                //sval(m) += sval_raw.head(m+1).sum();
            }
        }
        /*std::cout << sval_raw.transpose() << "\t"
                  << sumW.transpose() << "\t"
                  << curW << "\t"
                  << sval.transpose() << std::endl;
        if (isnan(sval(0)))
            abort();*/
    }

    return sval;
}

//! Checks if the neighbour lies within the shell offset
//! Returns true if inside the shell
bool StructureBase
:: neighbour_in_shell(Eigen::Vector2d& coords,
                      Eigen::Vector2d& nbCoords,
                      double minOffset, double maxOffset)
{
    // bottom or top of shell
    if (std::fabs(coords(1)-nbCoords(1))
            > minOffset + m_tol)
        if (std::fabs(coords(1)-nbCoords(1))
                <= maxOffset + m_tol)
        {
            if (std::fabs(coords(0)-nbCoords(0))
                    <= maxOffset + m_tol) {
                return true;
            }
        }

    // left or right of shell
    if (std::fabs(coords(0)-nbCoords(0))
            > minOffset + m_tol)
        if (std::fabs(coords(0)-nbCoords(0))
                <= maxOffset + m_tol)
        {
            if (std::fabs(coords(1)-nbCoords(1))
                    <= minOffset + m_tol) {
                return true;
            }
        }

    return false;
}

std::pair<bool, int> StructureBase
:: neighbour_in_shells
(Eigen::Vector2d& curCoords, Eigen::Vector2d& nbCoords,
 Eigen::VectorXd& offsets)
{
    auto M = offsets.size()-1;
    bool inside = false;

    int m;
    for (m=0; m<M; m++)
    {
        double minOffset = offsets(m);
        double maxOffset = offsets(m+1);
        inside = neighbour_in_shell
                (curCoords, nbCoords,
                 minOffset, maxOffset);

        if (inside) { break; }
    }

    return {inside, m};
}

//! Evaluates the structure function on unstructured meshes
//! using uniform grid hash table
//! Skips the cells of the uniform grid next to boundary
Eigen::VectorXd StructureBase
:: eval(std::shared_ptr<CellsHashTable>& cellsHashTable,
        Eigen::VectorXd offsets, int p)
{
    auto M = offsets.size()-1;
    double maxOffset = offsets(M);

    // read hash table parameters
    auto Nx = cellsHashTable->get_numCellsInX();
    auto xl = cellsHashTable->get_xLeft();
    auto xr = cellsHashTable->get_xRight();
    if (maxOffset > (xr - xl)/Nx) {
        std::cout << "Maximum offset is not compatible "
                     "with Nx of hash table" << std::endl;
        abort();
    }

    auto Ny = cellsHashTable->get_numCellsInY();
    auto yl = cellsHashTable->get_yLeft();
    auto yr = cellsHashTable->get_yRight();
    if (maxOffset > (yr - yl)/Ny) {
        std::cout << "Maximum offset is not compatible "
                     "with Ny of hash table" << std::endl;
        abort();
    }

    // compute
    Eigen::VectorXd sval(M);
    sval.setZero();
    for (int j=1; j<Ny-1; j++)
        for (int i=1; i<Nx-1; i++) {
            sval += evalInCell(cellsHashTable,
                               i, j, offsets, p);
        }

    return sval;
}

//! Evaluates the structure function on unstructured meshes
//! using uniform grid hash table
//! Skips the cells of the uniform grid next to boundary
Eigen::VectorXd StructureBase
:: eval(std::shared_ptr<ParCellsHashTable>& pcellsHashTable,
        Eigen::VectorXd offsets, int p)
{
    auto M = offsets.size()-1;
    double maxOffset = offsets(M);

    // read hash table parameters
    auto Nx = pcellsHashTable->get_numCellsInX();
    auto xl = pcellsHashTable->get_xLeft();
    auto xr = pcellsHashTable->get_xRight();
    if (maxOffset > (xr - xl)/Nx) {
        std::cout << "Maximum offset is not compatible "
                     "with Nx of hash table" << std::endl;
        abort();
    }

    auto Ny = pcellsHashTable->get_numCellsInY();
    auto yl = pcellsHashTable->get_yLeft();
    auto yr = pcellsHashTable->get_yRight();
    if (maxOffset > (yr - yl)/Ny) {
        std::cout << "Maximum offset is not compatible "
                     "with Ny of hash table" << std::endl;
        abort();
    }

    // compute process local
    auto cellsHashTable
            = pcellsHashTable->get_localHashTable();
    auto sval_local = eval(cellsHashTable, offsets, p);

    Eigen::VectorXd sval(M);
    auto comm = pcellsHashTable->get_comm();
    MPI_Allreduce(sval_local.data(), sval.data(), int(M),
                  MPI_DOUBLE, MPI_SUM, comm);

    return sval;
}

// End of file
