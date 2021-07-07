#ifndef MYMFEM_PCELL_AVERAGES_HPP
#define MYMFEM_PCELL_AVERAGES_HPP

#include "mfem.hpp"
using namespace mfem;


//! Class ParCellAverages
//! Computes the cell-averaged velocity components
class ParCellAverages
{
public:
    //! Constructors
    ParCellAverages (ParMesh *);

    ParCellAverages (ParFiniteElementSpace *);

    //! Destructor
    ~ParCellAverages () {
        if (m_sfes0) { delete m_sfes0; }
        if (m_l2_coll) { delete m_l2_coll; }
    }

    //! Evaluates x-component of velocity
    Vector eval_xComponent(const ParGridFunction *);

    //! Evaluates y-component of velocity
    Vector eval_yComponent(const ParGridFunction *);

    //! Evaluates x-component and y-component of velocity
    std::pair<Vector, Vector> eval(const ParGridFunction *v)
    {
        return {eval_xComponent(v), eval_yComponent(v)};
    }

    //! Returns scalar FE space
    inline ParFiniteElementSpace * get_sfes() {
        return m_sfes0;
    }

private:
    //! Evaluates velocity component along direction provided
    Vector compute(const ParGridFunction *, Vector&);

    //! Evaluates velocity component in direction n
    //! for given element
    double compute_fe(const FiniteElement&,
                      const Vector &el_dofs,
                      ElementTransformation&,
                      Vector&);

private:
    FiniteElementCollection *m_l2_coll = nullptr;
    ParFiniteElementSpace *m_sfes0 = nullptr;
};


#endif /// MYMFEM_PCELL_AVERAGES_HPP
