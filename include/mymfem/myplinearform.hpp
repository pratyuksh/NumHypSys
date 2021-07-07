#ifndef MYMFEM_PLINEARFORM_HPP
#define MYMFEM_PLINEARFORM_HPP

#include "mfem.hpp"
using namespace mfem;


class MyParLinearForm : public ParLinearForm
{
public:
    //! Constructor
    MyParLinearForm(ParFiniteElementSpace *pf)
        : ParLinearForm(pf) {mypfes = pf; myfes=pf;}
    
    //! Adds boundary integrator
    //! with boundary marker
    void MyAddBoundaryIntegrator
    (LinearFormIntegrator *lfi, Array<int> &bdr_attr_marker)
    {
        myblfi.Append (lfi);
        myblfi_marker.Append(&bdr_attr_marker);
    }

    //! Adds boundary face integrator
    //! with boundary marker
    void MyAddBdrFaceIntegrator(LinearFormIntegrator *lfi,
                                Array<int> &bdr_attr_marker)
    {
        mybflfi.Append (lfi);
        mybflfi_marker.Append(&bdr_attr_marker);
    }

    //! Adds internal mesh skeleton integrator
    void MyAddIntMeshSkeletonIntegrator
    (LinearFormIntegrator * lfi)
    {
        myislfi.Append (lfi);
    }

    //! Adds external mesh skeleton integrator
    //! no boundary marker
    void MyAddExtMeshSkeletonIntegrator
    (LinearFormIntegrator * lfi)
    {
        myeslfi.Append (lfi);
        myeslfi_marker.Append(nullptr);
    }

    //! Adds external mesh skeleton integrator
    //! with boundary marker
    void MyAddExtMeshSkeletonIntegrator
    (LinearFormIntegrator * lfi, Array<int> &bdr_attr_marker)
    {
        myeslfi.Append (lfi);
        myeslfi_marker.Append(&bdr_attr_marker);
    }

    //! Assembles the integrators
    void MyAssemble();

    //! Assembles in parallel
    void MyParallelAssemble(Vector &tv);
    HypreParVector *MyParallelAssemble();

protected:
    FiniteElementSpace *myfes;
    ParFiniteElementSpace *mypfes;

    Array<LinearFormIntegrator*> myblfi;
    Array<LinearFormIntegrator*> mybflfi;
    Array<LinearFormIntegrator*> myislfi;
    Array<LinearFormIntegrator*> myeslfi;

    Array<Array<int>*> myblfi_marker;
    Array<Array<int>*> mybflfi_marker;

    Array<Array<int>*> myeslfi_marker;
};


#endif /// MYMFEM_PLINEARFORM_HPP
