#ifndef MYMFEM_ASSEMBLY_HPP
#define MYMFEM_ASSEMBLY_HPP

#include "mfem.hpp"
using namespace mfem;


namespace mymfem { 

// Velocity x-component Projection Integrator
class VelocityXProjectionIntegrator
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
    DenseMatrix vshape;
    Vector shape;
#endif
};

// Velocity y-component Projection Integrator
class VelocityYProjectionIntegrator
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
    DenseMatrix vshape;
    Vector shape;
#endif
};

// Velocity to Vorticity Projection Integrator
class VorticityProjectionIntegrator
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
    DenseTensor gradvshape;
    Vector shape;
#endif
};

}

#endif /// MYMFEM_ASSEMBLY_HPP
