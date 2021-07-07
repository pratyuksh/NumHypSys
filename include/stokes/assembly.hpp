#ifndef STOKES_ASSEMBLY_HPP
#define STOKES_ASSEMBLY_HPP

#include "../mymfem/mybilininteg.hpp"

#include "mfem.hpp"
using namespace mfem;


namespace mymfem { 

// Gradient Integrator
class GradientIntegrator
        : public BilinearFormIntegrator
{
public:
    void AssembleElementMatrix
    (const FiniteElement &, ElementTransformation &,
     DenseMatrix &) override {}

    void AssembleElementMatrix2
    (const FiniteElement &trial_fe,
     const FiniteElement &test_fe,
     ElementTransformation &Trans, DenseMatrix &elmat)
    override;

private:
#ifndef MFEM_THREAD_SAFE
    DenseMatrix vshape, dshape;
#endif
};


// Diffusion Integrator
class DiffusionIntegrator
        : public MyBilinearFormIntegrator
{
public:
    void MyAssembleElementMatrix (const FiniteElement &el,
                                  const Vector &el_dofs,
                                  ElementTransformation &tr,
                                  DenseMatrix &elmat) const override;

    inline void MyAssembleBlock (const int ndofs,
                                 const DenseTensor &gradu,
                                 const double sCoeff,
                                 DenseMatrix &elmat) const;

private:
#ifndef MFEM_THREAD_SAFE
    mutable DenseTensor gradvshape;
#endif

};


// Diffusion Consistency Integrator
class DiffusionConsistencyIntegrator
        : public MyBilinearFormIntegrator
{
public:
    void MyAssembleFaceMatrix (const FiniteElement &fe1,
                               const FiniteElement &fe2,
                               const Vector &fe1_dofs,
                               FaceElementTransformations &ftr,
                               DenseMatrix &elmat) const override;

    inline void MyAssembleBlock (const int dim,
                                 const int row_ndofs,
                                 const int col_ndofs,
                                 const int row_offset,
                                 const int col_offset,
                                 const DenseMatrix &row_vshape,
                                 const DenseTensor &col_gradvshape,
                                 const double sCoeff,
                                 const Vector &nor,
                                 DenseMatrix &elmat) const;

private:
#ifndef MFEM_THREAD_SAFE
    mutable Vector nor;
    mutable DenseMatrix vshape1, vshape2;
    mutable DenseTensor gradvshape1, gradvshape2;
#endif
};


// Diffusion Symmetry Integrator
class DiffusionSymmetryIntegrator
        : public MyBilinearFormIntegrator
{
public:
    void MyAssembleFaceMatrix (const FiniteElement &fe1,
                               const FiniteElement &fe2,
                               const Vector &fe1_dofs,
                               FaceElementTransformations &ftr,
                               DenseMatrix &elmat) const override;

    inline void MyAssembleBlock (const int dim,
                                 const int row_ndofs,
                                 const int col_ndofs,
                                 const int row_offset,
                                 const int col_offset,
                                 const DenseTensor &row_gradvshape,
                                 const DenseMatrix &col_vshape,
                                 const double sCoeff,
                                 const Vector &nor,
                                 DenseMatrix &elmat) const;

private:
#ifndef MFEM_THREAD_SAFE
    mutable Vector nor;
    mutable DenseMatrix vshape1, vshape2;
    mutable DenseTensor gradvshape1, gradvshape2;
#endif
};


// Diffusion Penalty Integrator
class DiffusionPenaltyIntegrator
        : public MyBilinearFormIntegrator
{
public:
    DiffusionPenaltyIntegrator (double penalty) : m_penalty(penalty) {}

    void MyAssembleFaceMatrix (const FiniteElement &fe1,
                               const FiniteElement &fe2,
                               const Vector &fe1_dofs,
                               FaceElementTransformations &ftr,
                               DenseMatrix &elmat) const override;

    inline void MyAssembleBlock (const int dim,
                                 const int row_ndofs,
                                 const int col_ndofs,
                                 const int row_offset,
                                 const int col_offset,
                                 const DenseMatrix &row_vshape,
                                 const DenseMatrix &col_vshape,
                                 const double sCoeff,
                                 const Vector &nor,
                                 DenseMatrix &elmat) const;

private:
    double m_penalty;
#ifndef MFEM_THREAD_SAFE
    mutable Vector nor;
    mutable DenseMatrix vshape1, vshape2;
#endif
};


// Boundary Diffusion Consistency Integrator
class BdryDiffusionConsistencyIntegrator
        : public LinearFormIntegrator
{
public:
    BdryDiffusionConsistencyIntegrator(VectorCoefficient &Q_)
        : Q(Q_) { }

    virtual void AssembleRHSElementVect(const FiniteElement &,
                                        FaceElementTransformations &,
                                        Vector &);

    virtual void AssembleRHSElementVect(const FiniteElement &,
                                        ElementTransformation &,
                                        Vector &) {}

private:
    VectorCoefficient &Q;
#ifndef MFEM_THREAD_SAFE
    DenseTensor gradvshape;
    Vector nor, buf_vec;
    DenseMatrix buf_mat;
#endif
};


// Boundary Diffusion PenaltyIntegrator
class BdryDiffusionPenaltyIntegrator
        : public LinearFormIntegrator
{
public:
    BdryDiffusionPenaltyIntegrator(VectorCoefficient &Q_, double penalty)
        : Q(Q_), m_penalty(penalty) { }

    virtual void AssembleRHSElementVect(const FiniteElement &,
                                        FaceElementTransformations &,
                                        Vector &);

    virtual void AssembleRHSElementVect(const FiniteElement &,
                                        ElementTransformation &,
                                        Vector &) {}

private:
    VectorCoefficient &Q;
    double m_penalty;
#ifndef MFEM_THREAD_SAFE
    DenseMatrix vshape;
    Vector nor, buf_vec;
    DenseMatrix buf_mat1, buf_mat2;
#endif
};


// Open boundary integrator
class OpenBdryIntegrator : public MyBilinearFormIntegrator
{
public:
    OpenBdryIntegrator() {}

    void MyAssembleFaceMatrix
        (const FiniteElement &trial_fe,
         const FiniteElement &test_fe,
         FaceElementTransformations &ftr, DenseMatrix &elmat)
        const override;

private:
#ifndef MFEM_THREAD_SAFE
    mutable Vector shape, nor;
    mutable DenseMatrix vshape;
#endif
};

}

#endif /// STOKES_ASSEMBLY_HPP
