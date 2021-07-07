#ifndef INCOMPNS_ASSEMBLY_HPP
#define INCOMPNS_ASSEMBLY_HPP

#include "mfem.hpp"
using namespace mfem;

#include "../mymfem/mybilininteg.hpp"


namespace mymfem {

//! Convection Integrator
class ConvectionIntegrator
        : public MyBilinearFormIntegrator
{
public:
    void MyAssembleElementMatrix
    (const FiniteElement &el, const Vector &el_dofs,
     ElementTransformation &Trans, DenseMatrix &elmat)
    const override;

    inline void MyAssembleBlock
    (const int dim, const int row_ndofs, const int col_ndofs,
     const DenseMatrix &row_vshape,
     const DenseTensor &col_gradvshape,
     const Vector vCoeff, const double sCoeff,
     DenseMatrix &elmat) const;

private:
#ifndef MFEM_THREAD_SAFE
    mutable DenseMatrix vshape,
    mutable DenseTensor gradvshape;
#endif
};


//! Central Numerical Flux Integrator
class CentralNumFluxIntegrator
        : public MyBilinearFormIntegrator
{
public:
    void MyAssembleFaceMatrix
    (const FiniteElement &fe1, const FiniteElement &fe2,
     const Vector &fe1_dofs,
     FaceElementTransformations &ftr, DenseMatrix &elmat)
    const override;

    inline void MyAssembleBlock
    (const int dim, const int row_ndofs, const int col_ndofs,
     const int row_offset, const int col_offset,
     const DenseMatrix &row_vshape,
     const DenseMatrix &col_vshape,
     const double sCoeff, DenseMatrix &elmat) const;

private:
#ifndef MFEM_THREAD_SAFE
    mutable DenseMatrix vshape1, vshape2;
    mutable Vector nor, u;
#endif
};


//! Upwind Numerical Flux Integrator
class UpwindNumFluxIntegrator
        : public MyBilinearFormIntegrator
{
public:
    void MyAssembleFaceMatrix
    (const FiniteElement &fe1, const FiniteElement &fe2,
     const Vector &fe1_dofs,
     FaceElementTransformations &ftr, DenseMatrix &elmat)
    const override;

    inline void MyAssembleBlock
    (const int dim, const int row_ndofs, const int col_ndofs,
     const int row_offset, const int col_offset,
     const DenseMatrix &row_vshape,
     const DenseMatrix &col_vshape,
     const double sCoeff, DenseMatrix &elmat) const;

private:
#ifndef MFEM_THREAD_SAFE
    mutable DenseMatrix vshape1, vshape2;
    mutable Vector nor, u;
#endif
};


//! Matrix-free Convection + Numerical flux operator
class ApplyMFConvNFluxOperator
{
public:

    Vector operator()(Vector &U, Vector &W,
                      FiniteElementSpace *fes) const;

    Vector operator()(Vector &U, GridFunction *w) const;

    Vector operator()(GridFunction *u,
                      GridFunction *w) const;

    void AssembleElementRhsVector
    (const FiniteElement &el,
     const Vector &u_dofs, const Vector &w_dofs,
     ElementTransformation &Trans, Vector &elvec)
    const;

    void AssembleInteriorFaceRhsVector
    (const FiniteElement &fe1, const FiniteElement &fe2,
     const Vector &u_dofs1, const Vector &u_dofs2,
     const Vector &w_dofs,
     FaceElementTransformations &ftr, Vector &elvec)
    const;

protected:
    inline void AssembleBlockVector
    (const int ndofs,
     const int offset,
     const DenseMatrix &vshape,
     const Vector& u,
     const double sCoeff, Vector &elvec) const;

private:
#ifndef MFEM_THREAD_SAFE
    DenseTensor gradvshape;
    mutable DenseMatrix vshape, gradu;
    mutable Vector nor, u, w;

    mutable DenseMatrix vshape1, vshape2;
#endif
};

}

#endif /// INCOMPNS_ASSEMBLY_HPP
