#include "../../include/incompNS/poperators.hpp"

//! Block triangular Preconditioner
//! Constructor
mymfem :: IncompNSParBlockTriPr
:: IncompNSParBlockTriPr (const Array<int> & offsets)
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
void mymfem :: IncompNSParBlockTriPr
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
void mymfem :: IncompNSParBlockTriPr
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
    tmp.Destroy();
}

// End of file
