#ifndef MYMFEM_MIXEDBILINEARFORM_HPP
#define MYMFEM_MIXEDBILINEARFORM_HPP

#include "mybilininteg.hpp"


namespace mymfem {

class MyMixedBilinearForm : public MixedBilinearForm
{
public:
    //! Constructor
    MyMixedBilinearForm(FiniteElementSpace *tr_fes,
                        FiniteElementSpace *te_fes)
        : MixedBilinearForm(tr_fes, te_fes) { }
    
    //! Adds boundary face integrator
    //! no boundary marker
    void MyAddBoundaryFaceIntegrator
    (MyBilinearFormIntegrator * bfi)
    {
        mybfbfi.Append (bfi);
        mybfmarker.Append(nullptr);
    }

    //! Adds boundary face integrator
    //! with boundary marker
    void MyAddBoundaryFaceIntegrator
    (MyBilinearFormIntegrator * bfi,
     Array<int> &bdr_attr_marker)
    {
        mybfbfi.Append (bfi);
        mybfmarker.Append(&bdr_attr_marker);
    }
    
    //! Assembles the integrators
    void MyAssemble(int skip_zeros=1);
    
    //! Clears the integrators
    void MyClear()
    {
        for (int k=0; k < mybfbfi.Size(); k++)
        { delete mybfbfi[k]; }
    }

private:
    Array<MyBilinearFormIntegrator *> mybfbfi;
    Array<Array<int>*> mybfmarker;
};

}

#endif /// MYMFEM_MIXEDBILINEARFORM_HPP
