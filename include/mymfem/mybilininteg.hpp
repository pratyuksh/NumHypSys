#ifndef MYMFEM_BILININTEG_HPP
#define MYMFEM_BILININTEG_HPP

#include "mfem.hpp"
using namespace mfem;


/// Abstract base class MyBilinearFormIntegrator
class MyBilinearFormIntegrator
        : public BilinearFormIntegrator
{
public:
    virtual void MyAssembleElementMatrix
    (const FiniteElement &el, const Vector &el_dofs,
     ElementTransformation &Trans, DenseMatrix &elmat)
    const
    {
        MFEM_ABORT("MyAssembleElementMatrix is not "
                   "implemented for this Integrator class.");
    }

    virtual void MyAssembleFaceMatrix
    (const FiniteElement &fe1, const FiniteElement &fe2,
     const Vector &fe1_dofs,
     FaceElementTransformations &ftr, DenseMatrix &elmat)
    const
    {
        MFEM_ABORT("MyAssembleFaceMatrix is not implemented "
                   "for this Integrator class.");
    }

    virtual void MyAssembleFaceMatrix
    (const FiniteElement &trial_fe,
     const FiniteElement &test_fe,
     FaceElementTransformations &ftr, DenseMatrix &elmat)
    const
    {
        MFEM_ABORT("MyAssembleFaceMatrix (mixed form) "
                   "is not implemented "
                   "for this Integrator class.");
    }

    /*virtual void MyAssembleFaceMatrix
    (const FiniteElement &el1, const FiniteElement &el2,
     FaceElementTransformations &ftr, DenseMatrix &elmat)
    const
    {
        MFEM_ABORT("MyAssembleFaceMatrix is not implemented "
                   "for this Integrator class.");
    }

    virtual void MyAssembleFaceMatrix
    (const FiniteElement &trial_fe1,
     const FiniteElement &trial_fe2,
     const FiniteElement &test_fe1,
     const FiniteElement &test_fe2,
     FaceElementTransformations &ftr, DenseMatrix &elmat)
    const
    {
        MFEM_ABORT("MyAssembleFaceMatrix (mixed form) "
                   "is not implemented "
                   "for this Integrator class.");
    }

    virtual void MyAssembleFaceMatrix
    (const FiniteElement &trial_fe1,
     const FiniteElement &trial_fe2,
     const FiniteElement &test_fe,
     FaceElementTransformations &ftr, DenseMatrix &elmat)
    const
    {
        MFEM_ABORT("MyAssembleFaceMatrix (mixed form) "
                   "is not implemented "
                   "for this Integrator class.");
    }*/

    virtual ~MyBilinearFormIntegrator() {}
};


#endif /// MYMFEM_BILININTEG_HPP
