#include "../../include/incompNS/operators.hpp"


//! Block(0,0) of inverse of Preconditioner P1
//! Operator: I - invD * B1^T * invS * B2
void mymfem :: InvPrP1Block00Op ::
Mult (const Vector &x, Vector& y) const
{
    assert(x.Size() == width);
    assert(y.Size() == height);

    // tmp1 <- B2 * x
    tmp1.SetSize(m_matB2->NumRows());
    m_matB2->Mult(x, tmp1);

    // tmp2 <- invS * tmp1
    tmp2.SetSize(m_invMatS->NumRows());
    m_invMatS->Mult(tmp1, tmp2);

    // tmp1 <- B1^T * tmp2
    tmp1.SetSize(m_matB1T->NumRows());
    m_matB1T->Mult(tmp2, tmp1);

    // y <- invD * tmp1
    m_invMatD->Mult(tmp1, y);

    // y <- x-y
    y *= -1;
    y.Add(1, x);

    // release memory
    tmp1.Destroy();
    tmp2.Destroy();
}

//! Block(1,0) of inverse of Preconditioner P1
//! Operator: invS * B2
void mymfem :: InvPrP1Block10Op ::
Mult (const Vector &x, Vector& y) const
{
    assert(x.Size() == width);
    assert(y.Size() == height);

    // tmp <- B2 * x
    tmp.SetSize(m_matB2->NumRows());
    m_matB2->Mult(x, tmp);

    // y <- invS * tmp
    m_invMatS->Mult(tmp, y);

    // release memory
    tmp.Destroy();
}

//! Block(0,1) of inverse of Preconditioner P1
//! Operator: invD * B^T
void mymfem :: InvPrP1Block01Op ::
Mult (const Vector &x, Vector& y) const
{
    assert(x.Size() == width);
    assert(y.Size() == height);

    // tmp <- B1^T * x
    tmp.SetSize(m_matB1T->NumRows());
    m_matB1T->Mult(x, tmp);

    // y <- invD * tmp
    m_invMatD->Mult(tmp, y);

    // release memory
    tmp.Destroy();
}

//! Inverse of Preconditioner P1
//! Constructor
mymfem :: InvPrP1Op
:: InvPrP1Op (const Array<int> & offsets)
    : nRowBlocks(offsets.Size() - 1),
      nColBlocks(offsets.Size() - 1),
      row_offsets(0),
      col_offsets(0)
{
    assert(nRowBlocks == 2);
    assert(nColBlocks == 2);

    row_offsets.MakeRef(offsets);
    col_offsets.MakeRef(offsets);

    height = row_offsets[2] - row_offsets[0];
    width = col_offsets[2] - col_offsets[0];
}

//! Destructor
mymfem :: InvPrP1Op
:: ~ InvPrP1Op()
{
    if (m_invPrP1block00) { delete m_invPrP1block00; }
    if (m_invPrP1block10) { delete m_invPrP1block10; }
    if (m_invPrP1block01) { delete m_invPrP1block01; }
}

//! Sets operator components
void mymfem :: InvPrP1Op
:: update(Operator* matB1T,
          Operator* matB2,
          Operator* invMatS,
          Operator* invMatD)
{
    if (m_invPrP1block00) {
        delete m_invPrP1block00;
        m_invPrP1block00 = nullptr;
    }
    m_invPrP1block00 = new InvPrP1Block00Op
            (matB1T, matB2, invMatS, invMatD);

    if (m_invPrP1block10) {
        delete m_invPrP1block10;
        m_invPrP1block10 = nullptr;
    }
    m_invPrP1block10 = new InvPrP1Block10Op
            (matB2, invMatS);

    if (m_invPrP1block01) {
        delete m_invPrP1block01;
        m_invPrP1block01 = nullptr;
    }
    m_invPrP1block01 = new InvPrP1Block01Op
            (matB1T, invMatD);
}

//! Applies (inverse(P1))(x)
void mymfem :: InvPrP1Op ::
Mult (const Vector &x, Vector& y) const
{
    assert(x.Size() == width);
    assert(y.Size() == height);

    x.Read();
    y.Write(); y = 0.0;

    xblock.Update(const_cast<Vector&>(x),
                  col_offsets);
    yblock.Update(y,row_offsets);

    // block 00
    tmp.SetSize(row_offsets[1] - row_offsets[0]);
    m_invPrP1block00->Mult(xblock.GetBlock(0), tmp);
    yblock.GetBlock(0).Add(1, tmp);

    // block 01
    m_invPrP1block01->Mult(xblock.GetBlock(1), tmp);
    yblock.GetBlock(0).Add(1, tmp);

    // block 10
    tmp.SetSize(row_offsets[2] - row_offsets[1]);
    m_invPrP1block10->Mult(xblock.GetBlock(0), tmp);
    yblock.GetBlock(1).Add(1, tmp);

    // block 11
    yblock.GetBlock(1).Add(-1, xblock.GetBlock(1));

    // Destroy alias vectors to prevent dangling aliases
    // when the base vectors are deleted
    for (int i=0; i < nColBlocks; ++i)
    { xblock.GetBlock(i).Destroy(); }
    for (int i=0; i < nRowBlocks; ++i)
    { yblock.GetBlock(i).Destroy(); }
    tmp.Destroy();
}


//! Preconditioner P2
//! Operator: invD * E := invD * (A - D) = invD * A - I
void mymfem :: PrP2Op ::
Mult (const Vector &x, Vector& y) const
{
    assert(x.Size() == width);
    assert(y.Size() == height);

    // tmp <- A * x
    tmp.SetSize(m_matA->NumRows());
    m_matA->Mult(x, tmp);

    // y <- invD * tmp
    m_invMatD->Mult(tmp, y);

    // y <- (y-x)
    y.Add(-1, x);

    // release memory
    tmp.Destroy();
}


//! System Operator with embedded preconditioner
//! Operator: I + invP1.block(0,0) * P2
void mymfem :: IncompNSPrOp ::
Mult (const Vector &x, Vector& y) const
{
    assert(x.Size() == width);
    assert(y.Size() == height);

    // tmp <- P2 * x
    tmp.SetSize(m_prP2->NumRows());
    m_prP2->Mult(x, tmp);

    // y <- invP1 * tmp
    m_invPrP1->get_block00()->Mult(tmp, y);

    // y <- y + x
    y.Add(1, x);

    // release memory
    tmp.Destroy();
}


//! Block triangular Preconditioner
//! Constructor
mymfem :: IncompNSBlockTriPr
:: IncompNSBlockTriPr (const Array<int> & offsets)
    : nRowBlocks(offsets.Size() - 1),
      nColBlocks(offsets.Size() - 1),
      row_offsets(0),
      col_offsets(0)
{
    assert(nRowBlocks == 2);
    assert(nColBlocks == 2);

    row_offsets.MakeRef(offsets);
    col_offsets.MakeRef(offsets);

    height = row_offsets[2] - row_offsets[0];
    width = col_offsets[2] - col_offsets[0];
}

//! Sets the solver components
void mymfem :: IncompNSBlockTriPr
:: set(Operator* matB1T,
       Operator* invMatS,
       Operator* invMatD,
       Operator* meanFreePressureOp)
{
    m_matB1T = matB1T;
    m_invMatS = invMatS;
    m_invMatD = invMatD;
    m_meanFreePressureOp = meanFreePressureOp;
}

//! Applies the solver
void mymfem :: IncompNSBlockTriPr
:: Mult (const Vector &x, Vector& y) const
{
    assert(x.Size() == width);
    assert(y.Size() == height);

    x.Read();
    y.Write(); y = 0.0;

    xblock.Update(const_cast<Vector&>(x),
                  col_offsets);
    yblock.Update(y,row_offsets);

    // block 1
    m_invMatS->Mult(xblock.GetBlock(1),
                    yblock.GetBlock(1));

    // block 0
    tmp.SetSize(row_offsets[1] - row_offsets[0]);
    m_matB1T->Mult(yblock.GetBlock(1), tmp);
    tmp.Add(1, xblock.GetBlock(0));
    m_invMatD->Mult(tmp, yblock.GetBlock(0));

    yblock.GetBlock(1) *= -1;

    if (m_meanFreePressureOp)
    {
        tmp.SetSize(row_offsets[2] - row_offsets[1]);
        tmp = yblock.GetBlock(1);
        m_meanFreePressureOp->Mult
                (tmp, yblock.GetBlock(1));
    }

    // Destroy alias vectors to prevent dangling aliases
    // when the base vectors are deleted
    for (int i=0; i < nColBlocks; ++i)
    { xblock.GetBlock(i).Destroy(); }
    for (int i=0; i < nRowBlocks; ++i)
    { yblock.GetBlock(i).Destroy(); }
}

// End of file
