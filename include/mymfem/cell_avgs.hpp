#ifndef MYMFEM_CELL_AVERAGES_HPP
#define MYMFEM_CELL_AVERAGES_HPP

//#include "utilities.hpp"

#include "mfem.hpp"
using namespace mfem;


//! Class CellAverages
//! Computes the cell-averaged velocity components
class CellAverages
{
public:
    //! Constructors
    CellAverages(Mesh *mesh);

    CellAverages (FiniteElementSpace *);

    //! Destructor
    ~CellAverages() {
        if (m_sfes0) { delete m_sfes0; }
        if (m_l2_coll) { delete m_l2_coll; }
    }

    //! Evaluates x-component and y-component of velocity
    std::pair<Vector, Vector> eval(GridFunction *v);

    //! Evaluates x-component of velocity
    Vector eval_xComponent(GridFunction *v);

    //! Evaluates y-component of velocity
    Vector eval_yComponent(GridFunction *v);

    //! Returns scalar FE space
    inline FiniteElementSpace * get_sfes() {
        return m_sfes0;
    }

private:
    //! Evaluates velocity component along direction provided
    Vector compute(const GridFunction *, Vector&);

    //! Evaluates velocity component in direction n
    //! for given element
    double compute_fe(const FiniteElement&,
                      const Vector &el_dofs,
                      ElementTransformation&,
                      Vector&);

private:
    FiniteElementCollection *m_l2_coll = nullptr;
    FiniteElementSpace *m_sfes0 = nullptr;

    //std::unique_ptr<VelocityFunction> m_vel;
};

#endif /// MYMFEM_CELL_AVERAGES_HPP
