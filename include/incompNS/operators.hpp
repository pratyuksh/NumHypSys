#ifndef INCOMPNS_OPERATORS_HPP
#define INCOMPNS_OPERATORS_HPP

#include "mfem.hpp"
using namespace mfem;


namespace mymfem {

//! Block(0,0) of inverse of Preconditioner P1
//! Operator: I - invD * B1^T * invS * B2
class InvPrP1Block00Op : public Operator
{
public:
    //! Constructor
    InvPrP1Block00Op(Operator* matB1T,
                     Operator* matB2,
                     Operator* invMatS,
                     Operator* invMatD)
    : m_matB1T (matB1T),
      m_matB2 (matB2),
      m_invMatS (invMatS),
      m_invMatD(invMatD)
    {
        height = invMatD->NumRows();
        width = matB2->NumCols();
    }

    //! Applies operator
    void Mult(const Vector &x, Vector& y) const override;

private:
    Operator *m_matB1T = nullptr;
    Operator *m_matB2 = nullptr;
    Operator *m_invMatS = nullptr;
    Operator *m_invMatD = nullptr;

    mutable Vector tmp1, tmp2;
};

//! Block(1,0) of inverse of Preconditioner P1
//! Operator: invS * B2
class InvPrP1Block10Op : public Operator
{
public:
    //! Constructor
    InvPrP1Block10Op(Operator* matB2,
                     Operator* invMatS)
    : m_matB2 (matB2),
      m_invMatS (invMatS)
    {
        height = invMatS->NumRows();
        width = matB2->NumCols();
    }

    //! Applies operator
    void Mult(const Vector &x, Vector& y) const override;

private:
    Operator *m_matB2 = nullptr;
    Operator *m_invMatS = nullptr;

    mutable Vector tmp;
};

//! Block(0,1) of inverse of Preconditioner P1
//! Operator: invD * B^T
class InvPrP1Block01Op : public Operator
{
public:
    //! Constructor
    InvPrP1Block01Op(Operator* matB1T, Operator* invMatD)
    : m_matB1T (matB1T), m_invMatD (invMatD)
    {
        height = invMatD->NumRows();
        width = matB1T->NumCols();
    }

    //! Applies operator
    void Mult(const Vector &x, Vector& y) const override;

private:
    Operator *m_matB1T = nullptr;
    Operator *m_invMatD = nullptr;

    mutable Vector tmp;
};

//! Inverse of Preconditioner P1
class InvPrP1Op : public Operator
{
public:
    //! Constructor
    InvPrP1Op(const Array<int> & offsets);

    //! Destructor
    ~ InvPrP1Op();

    //! Sets operator components
    void update(Operator* matB1T,
                Operator* matB2,
                Operator* invMatS,
                Operator* invMatD);

    //! Applies (inverse(P1))(x)
    void Mult(const Vector &x, Vector& y) const override;

    //! Returns Block(0,0)
    Operator * get_block00() { return m_invPrP1block00; }

    //! Returns Block(1,0)
    Operator * get_block10() { return m_invPrP1block10; }

    //! Returns Block(0,1)
    Operator * get_block01() { return m_invPrP1block01; }

private:
    InvPrP1Block00Op* m_invPrP1block00 = nullptr;
    InvPrP1Block10Op* m_invPrP1block10 = nullptr;
    InvPrP1Block01Op* m_invPrP1block01 = nullptr;

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


//! Preconditioner P2
//! Operator: invD * E := invD * (A - D) = invD * A - I
class PrP2Op : public Operator
{
public:
    //! Constructor
    PrP2Op(Operator* matA, Operator* invMatD)
        : m_matA (matA), m_invMatD (invMatD)
    {
        height = invMatD->NumRows();
        width = matA->NumCols();
    }

    //! Applies operator
    void Mult(const Vector &x, Vector& y) const override;

private:
    Operator *m_matA = nullptr;
    Operator *m_invMatD = nullptr;

    mutable Vector tmp;
};


//! System Operator with embedded preconditioner
//! Operator: I + invP1.block(0,0) * P2
class IncompNSPrOp : public Operator
{
public:
    //! Constructor
    IncompNSPrOp(InvPrP1Op* invPrP1, PrP2Op* prP2)
    : m_invPrP1 (invPrP1), m_prP2 (prP2)
    {
        height = invPrP1->get_block00()->NumRows();
        width = prP2->NumCols();
    }

    //! Applies operator
    void Mult(const Vector &x, Vector& y) const override;

private:
    InvPrP1Op *m_invPrP1 = nullptr;
    PrP2Op* m_prP2 = nullptr;

    mutable Vector tmp;
};


//! Block triangular Preconditioner
class IncompNSBlockTriPr : public Solver
{
public:
    //! Constructor
    IncompNSBlockTriPr(const Array<int> & offsets);

    //! Destructor
    ~ IncompNSBlockTriPr() {}

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
