#include "../../include/mymfem/myplinearform.hpp"


//! Assembles the integrators
void MyParLinearForm::MyAssemble()
{
    Array<int> vdofs;
    ElementTransformation *eltrans;
    Vector elemvect;

    Vector::operator=(0.0);

    if (myblfi.Size())
    {
        Mesh *mesh = myfes->GetMesh();

        // Which boundary attributes need to be processed?
        Array<int> bdr_attr_marker
                (mesh->bdr_attributes.Size() ?
                     mesh->bdr_attributes.Max() : 0);
        bdr_attr_marker = 0;
        for (int k = 0; k < myblfi.Size(); k++)
        {
            if (myblfi_marker[k] == nullptr)
            {
                bdr_attr_marker = 1;
                break;
            }
            Array<int> &bdr_marker = *myblfi_marker[k];
            MFEM_ASSERT(bdr_marker.Size()
                        == bdr_attr_marker.Size(),
                        "invalid boundary marker for "
                        "boundary integrator #"
                        << k << ", counting from zero");
            for (int i = 0; i < bdr_attr_marker.Size(); i++)
            {
                bdr_attr_marker[i] |= bdr_marker[i];
            }
        }

        for (int i = 0; i < myfes -> GetNBE(); i++)
        {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr-1] == 0)
            { continue; }
            myfes -> GetBdrElementVDofs (i, vdofs);
            eltrans = myfes->GetBdrElementTransformation(i);
            for (int k=0; k < myblfi.Size(); k++)
            {
                myblfi[k]->AssembleRHSElementVect
                        (*myfes->GetBE(i),
                         *eltrans, elemvect);
                AddElementVector (vdofs, elemvect);
            }
        }
    }

    if (myislfi.Size())
    {
        FaceElementTransformations *ftr;
        Mesh *mesh = myfes->GetMesh();

        for (int i = 0; i < mesh -> GetNumFaces(); i++)
        {
            ftr = mesh -> GetInteriorFaceTransformations(i);
            if (ftr != nullptr)
            {
                myfes -> GetFaceVDofs(i, vdofs);
                for (int k=0; k < myislfi.Size(); k++)
                {
                    myislfi[k]->AssembleRHSElementVect
                            (*myfes->GetFaceElement(i),
                             *ftr, elemvect);
                    AddElementVector (vdofs, elemvect);
                }
            }
        }
    }

    if (myeslfi.Size())
    {
        FaceElementTransformations *ftr;
        Mesh *mesh = myfes->GetMesh();

        // Which boundary attributes need to be processed?
        Array<int> bdr_attr_marker
                (mesh->bdr_attributes.Size() ?
                     mesh->bdr_attributes.Max() : 0);
        bdr_attr_marker = 0;
        for (int k = 0; k < myeslfi.Size(); k++)
        {
            if (myeslfi_marker[k] == nullptr)
            {
                bdr_attr_marker = 1;
                break;
            }
            Array<int> &bdr_marker = *myeslfi_marker[k];
            MFEM_ASSERT(bdr_marker.Size()
                        == bdr_attr_marker.Size(),
                        "invalid boundary marker for "
                        "boundary integrator #"
                        << k << ", counting from zero");
            for (int i = 0; i < bdr_attr_marker.Size(); i++)
            {
                bdr_attr_marker[i] |= bdr_marker[i];
            }
        }

        for (int i = 0; i < mesh -> GetNBE(); i++)
        {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            ftr = mesh -> GetBdrFaceTransformations(i);

            if ((bdr_attr_marker[bdr_attr-1] != 0)
                    && ftr != nullptr)
            {
                myfes -> GetBdrElementVDofs (i, vdofs);
                for (int k=0; k < myeslfi.Size(); k++)
                {
                    myeslfi[k]->AssembleRHSElementVect
                            (*myfes->GetFaceElement(i),
                             *ftr, elemvect);
                    AddElementVector (vdofs, elemvect);
                }
            }
        }
    }
}

//! Assembles in parallel
void MyParLinearForm::MyParallelAssemble(Vector &tv)
{
    mypfes->GetProlongationMatrix()
            ->MultTranspose(*this, tv);
}

HypreParVector *MyParLinearForm::MyParallelAssemble()
{
   HypreParVector *tv = mypfes->NewTrueDofVector();
   mypfes->GetProlongationMatrix()
           ->MultTranspose(*this, *tv);
   return tv;
}

// End of file
