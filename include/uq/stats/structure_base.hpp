#ifndef UQ_STRUCTURE_BASE_HPP
#define UQ_STRUCTURE_BASE_HPP

#include "pcells_hash_table.hpp"
#include <Eigen/Dense>


class StructureBase
{
public:
    explicit StructureBase() {}

    //! Evaluates the structure function on structured grid
    Eigen::VectorXd eval(const Eigen::MatrixXd& U,
                         Eigen::VectorXi stencil, int p);

    //! Evaluates the structure function for a point/cell
    //! of the structured grid
    double evalAtPoint(const Eigen::MatrixXd& U,
                       int i, int j,
                       int stencil, int p);


    //! Evaluates the structure function for a cell
    //! in the uniform grid hash table
    Eigen::VectorXd evalInCell
    (std::shared_ptr<CellsHashTable>& cellsHashTable,
     int i, int j, Eigen::VectorXd offsets, int p);

    //! Evaluates the structure function
    //! on unstructured meshes
    //! using uniform grid hash table
    Eigen::VectorXd eval
    (std::shared_ptr<CellsHashTable>& cellsHashTable,
     Eigen::VectorXd offsets, int p);

    //! Evaluates the structure function
    //! on unstructured meshes
    //! using uniform grid hash table
    Eigen::VectorXd eval
    (std::shared_ptr<ParCellsHashTable>& cellsHashTable,
     Eigen::VectorXd offsets, int p);

private:
    //! Evaluates functional
    inline double functional(const Eigen::Vector2d& Uij,
                             const Eigen::Vector2d& Ukl,
                             int p);

    //! Returns solution U(k,l)
    inline Eigen::Vector2d getSol(const Eigen::MatrixXd& U,
                                  int k, int l);

    //! Checks if the neighbour lies within the shell offset
    inline bool neighbour_in_shell
    (Eigen::Vector2d& coords, Eigen::Vector2d& nbCoords,
     double minOffset, double maxOffset);

    //! Checks if the neighbour lies within the shells
    std::pair<bool, int> neighbour_in_shells
    (Eigen::Vector2d& coords, Eigen::Vector2d& nbCoords,
     Eigen::VectorXd& offsets);

    double m_tol=1E-12;
};


#endif /// UQ_STRUCTURE_BASE_HPP
