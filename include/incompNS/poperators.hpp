#ifndef INCOMPNS_POPERATORS_HPP
#define INCOMPNS_POPERATORS_HPP

#include "mfem.hpp"
using namespace mfem;


namespace mymfem {

//! Block triangular Preconditioner
class IncompNSParBlockTriPr : public Solver
{
public:
    //! Constructor
    IncompNSParBlockTriPr(const Array<int> & offsets);

    //! Destructor
    ~ IncompNSParBlockTriPr() {}

    //! Sets the solver components
    void set(Operator* matB1T,
             Operator* invMatS,
             Operator* invMatD,
             Operator* meanFreePressureOp=nullptr);

    //! Applies the solver
    void Mult(const Vector &x, Vector& y) const override;

    //! Needed by the abstract class Solver
    virtual void SetOperator(const Operator &op) { }

private:
    Operator *m_matB1T = nullptr;
    Operator *m_invMatS = nullptr;
    Operator *m_invMatD = nullptr;
    Operator *m_meanFreePressureOp = nullptr;

    int nRowBlocks;
    int nColBlocks;

    Array<int> row_offsets;
    Array<int> col_offsets;

    //! Temporary Vectors used to efficiently
    //! apply the Mult and MultTranspose methods.
    mutable BlockVector xblock;
    mutable BlockVector yblock;
    mutable Vector tmp;
};

}

#endif /// INCOMPNS_POPERATORS_HPP
