#ifndef MYMFEM_PBILINEARFORM_HPP
#define MYMFEM_PBILINEARFORM_HPP

#include "../includes.hpp"
#include "mybilininteg.hpp"
#include "mybilinearform.hpp"


namespace mymfem {

class MyParBilinearForm : public ParBilinearForm
{
public:
    //! Constructor
    MyParBilinearForm(ParFiniteElementSpace *pfes)
        : ParBilinearForm(pfes) { }

    //! Adds domain integrator
    void MyAddDomainIntegrator
    (MyBilinearFormIntegrator * bfi) {
            mydbfi.Append (bfi);
    }

    //! Adds interior face integrator
    void MyAddInteriorFaceIntegrator
    (MyBilinearFormIntegrator * bfi) {
            myifbfi.Append (bfi);
    }

    //! Adds boundary face integrator
    //! no boundary marker
    void MyAddBoundaryFaceIntegrator
    (MyBilinearFormIntegrator * bfi, bool useFaceInt=false)
    {
            mybfbfi.Append (bfi);
            mybfmarker.Append(nullptr);
            m_useFaceInts.Append(useFaceInt);
    }

    //! Adds boundary face integrator
    //! with boundary marker
    void MyAddBoundaryFaceIntegrator
    (MyBilinearFormIntegrator * bfi,
     Array<int> &bdr_attr_marker,
     bool useFaceInt=false)
    {
        mybfbfi.Append (bfi);
        mybfmarker.Append(&bdr_attr_marker);
        m_useFaceInts.Append(useFaceInt);
    }

    //! Adds face integrator
    //! no bounadry marker
    void MyAddFaceIntegrator (MyBilinearFormIntegrator * bfi)
    {
        bool useFaceInt = true;
        MyAddInteriorFaceIntegrator(bfi);
        MyAddBoundaryFaceIntegrator(bfi, useFaceInt);
    }

    //! Adds face integrator
    //! with boundary marker
    void MyAddFaceIntegrator (MyBilinearFormIntegrator * bfi,
                              Array<int> &bdr_attr_marker)
    {
        bool useFaceInt = true;
        MyAddInteriorFaceIntegrator(bfi);
        MyAddBoundaryFaceIntegrator(bfi, bdr_attr_marker,
                                    useFaceInt);
    }

    //! Assembles the integrators
    void MyAssemble(const ParGridFunction *u,
                    int skip_zeros=1);

    //! Assmbles the shared faces
    void MyAssembleSharedFaces(const ParGridFunction *u,
                               int skip_zeros=1);

    //! Assembles in parallel
    HypreParMatrix* MyParallelAssemble();
    HypreParMatrix* MyParallelAssemble(SparseMatrix *m);
    void MyParallelAssemble(OperatorHandle &A,
                            SparseMatrix *A_local);

    //! Clear the integrators
    void MyClear()
    {
        for (int k=0; k < mydbfi.Size(); k++)
        { delete mydbfi[k]; }
        for (int k=0; k < myifbfi.Size(); k++)
        { delete myifbfi[k]; }
        for (int k=0; k < mybfbfi.Size(); k++)
        {
            if (!m_useFaceInts[k]) {
                delete mybfbfi[k];
            }
        }
    }

private:
    Array<MyBilinearFormIntegrator *> mydbfi;
    Array<MyBilinearFormIntegrator *> myifbfi;
    Array<MyBilinearFormIntegrator *> mybfbfi;
    Array<Array<int>*> mybfmarker;

    Array<bool> m_useFaceInts;
};

}

#endif /// MYMFEM_PBILINEARFORM_HPP
