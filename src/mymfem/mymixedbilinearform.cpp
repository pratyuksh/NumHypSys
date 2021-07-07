#include "../../include/mymfem/mymixedbilinearform.hpp"
#include "../../include/mymfem/utilities.hpp"


//! Assembles the integrators
void mymfem::MyMixedBilinearForm
:: MyAssemble (int skip_zeros)
{
    // allocate memory
    if (mat == nullptr) {
        mat = new SparseMatrix(height, width);
    }

    // assemble boundary face integrators
    if (mybfbfi.Size())
    {
        DenseMatrix elmat;
        Array<int> trial_vdofs, test_vdofs;
        FaceElementTransformations *ftr = nullptr;
        Mesh *mesh = test_fes -> GetMesh();
        
        // set boundary markers
        Array<int> bdr_attr_marker
                (mesh->bdr_attributes.Size() ?
                     mesh->bdr_attributes.Max() : 0);
        bdr_attr_marker = 0;
        for (int k = 0; k < mybfbfi.Size(); k++)
        {
            if (mybfmarker[k] == nullptr)
            {
                bdr_attr_marker = 1;
                break;
            }
            Array<int> &bdr_marker = *mybfmarker[k];
            MFEM_ASSERT(bdr_marker.Size()
                        == bdr_attr_marker.Size(),
                        "invalid boundary marker "
                        "for boundary face "
                        "integrator #" << k
                        << ", counting from zero");
            for (int i = 0; i < bdr_attr_marker.Size(); i++)
            {
                bdr_attr_marker[i] |= bdr_marker[i];
            }
        }

        int nbfaces = mesh->GetNBE();
        for (int i = 0; i < nbfaces; i++)
        {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr-1] == 0) {continue;}
            
            ftr = mesh -> GetBdrFaceTransformations(i);
            const FiniteElement &trial_fe
                    = *(trial_fes->GetFE (ftr->Elem1No));
            const FiniteElement &test_fe
                    = *(test_fes->GetFE (ftr->Elem1No));
            trial_fes->GetElementVDofs
                    (ftr -> Elem1No, trial_vdofs);
            test_fes->GetElementVDofs
                    (ftr->Elem1No, test_vdofs);

            for (int k = 0; k < mybfbfi.Size(); k++)
            {
                mybfbfi[k] -> MyAssembleFaceMatrix
                        (trial_fe, test_fe, *ftr, elmat);
                mat -> AddSubMatrix (test_vdofs, trial_vdofs,
                                     elmat, skip_zeros);
            }
        }
    }
}


// End of file
