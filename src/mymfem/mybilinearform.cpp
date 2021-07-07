#include "../../include/mymfem/mybilinearform.hpp"
#include "../../include/mymfem/utilities.hpp"


//! Assembles the integrators
void mymfem::MyBilinearForm
:: MyAssemble (const GridFunction *u, int skip_zeros)
{
    // allocate memory
    if (mat == nullptr) {
        AllocMat();
    }

    // assemble domain integrators
    if (mydbfi.Size())
    {
        DenseMatrix elmat;
        ElementTransformation *eltrans = nullptr;
        Vector el_dofs;

        for (int i = 0; i < fes -> GetNE(); i++)
        {
            fes->GetElementVDofs(i, vdofs);
            el_dofs.SetSize(vdofs.Size());
            if (u) { get_dofs(*u, vdofs, el_dofs); }

            const FiniteElement &fe = *fes->GetFE(i);
            eltrans = fes->GetElementTransformation(i);
            for (int k = 0; k < mydbfi.Size(); k++)
            {
                mydbfi[k]->MyAssembleElementMatrix
                        (fe, el_dofs, *eltrans, elmat);
                mat->AddSubMatrix(vdofs, vdofs, elmat,
                                  skip_zeros);
            }
        }
    }

    // assemble interior face integrators
    if (myifbfi.Size())
    {
        DenseMatrix elmat;
        FaceElementTransformations *ftr = nullptr;
        Array<int> vdofs2;
        Vector fe1_dofs;
        Mesh *mesh = fes -> GetMesh();

        int nfaces = mesh->GetNumFaces();
        for (int i = 0; i < nfaces; i++)
        {
            ftr = mesh -> GetInteriorFaceTransformations(i);
            if (ftr != nullptr)
            {
                const FiniteElement &fe1
                        = *(fes->GetFE (ftr->Elem1No));
                const FiniteElement &fe2
                        = *(fes->GetFE (ftr->Elem2No));

                fes -> GetElementVDofs
                        (ftr -> Elem1No, vdofs);
                fe1_dofs.SetSize(vdofs.Size());
                if (u) { get_dofs(*u, vdofs, fe1_dofs); }

                fes -> GetElementVDofs
                        (ftr -> Elem2No, vdofs2);
                vdofs.Append (vdofs2);
                for (int k = 0; k < myifbfi.Size(); k++)
                {
                    myifbfi[k] -> MyAssembleFaceMatrix
                            (fe1, fe2, fe1_dofs,
                             *ftr, elmat);

                    mat -> AddSubMatrix (vdofs, vdofs,
                                         elmat, skip_zeros);
                }
            }
        }
    }

    // assemble boundary face integrators
    if (mybfbfi.Size())
    {
        DenseMatrix elmat;
        FaceElementTransformations *ftr = nullptr;
        Vector fe_dofs;
        Mesh *mesh = fes -> GetMesh();

        int nbfaces = mesh->GetNBE();
        for (int i = 0; i < nbfaces; i++)
        {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            ftr = mesh -> GetBdrFaceTransformations(i);
            const FiniteElement &fe
                    = *(fes->GetFE (ftr->Elem1No));

            fes -> GetElementVDofs (ftr -> Elem1No, vdofs);
            fe_dofs.SetSize(vdofs.Size());
            if (u) { get_dofs(*u, vdofs, fe_dofs); }

            for (int k = 0; k < mybfbfi.Size(); k++)
            {
                if (mybfmarker[k]) {
                    Array<int> &bdr_marker = *mybfmarker[k];
                    if (bdr_marker[bdr_attr-1] == 0)
                    { continue; }
                }

                mybfbfi[k] -> MyAssembleFaceMatrix
                        (fe, fe, fe_dofs, *ftr, elmat);
                mat -> AddSubMatrix (vdofs, vdofs, elmat,
                                     skip_zeros);
            }
        }
    }
}


// End of file
