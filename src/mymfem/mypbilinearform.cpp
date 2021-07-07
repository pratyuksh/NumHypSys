#include "../../include/mymfem/mypbilinearform.hpp"
#include "../../include/mymfem/utilities.hpp"


//! Assembles the integrators
void mymfem::MyParBilinearForm
:: MyAssemble (const ParGridFunction *u, int skip_zeros)
{
    // allocate memory
    if (mat == nullptr && (myifbfi.Size() > 0))
    {
        pfes->ExchangeFaceNbrData();
        pAllocMat();
    }
    else if (mat == nullptr)
    {
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

            const FiniteElement &fe = *(fes->GetFE(i));
            eltrans = fes->GetElementTransformation(i);
            for (int k = 0; k < mydbfi.Size(); k++)
            {
                mydbfi[k]->MyAssembleElementMatrix
                        (fe, el_dofs, *eltrans, elmat);
                mat->AddSubMatrix(vdofs, vdofs,
                                  elmat, skip_zeros);
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
            ftr = mesh -> GetInteriorFaceTransformations (i);
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
        MyAssembleSharedFaces(u, skip_zeros);
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

            for (int k = 0; k < myifbfi.Size(); k++)
            {
                if (mybfmarker[k]) {
                    Array<int> &bdr_marker = *mybfmarker[k];
                    if (bdr_marker[bdr_attr-1] == 0)
                    { continue; }
                }

                mybfbfi[k] -> MyAssembleFaceMatrix
                        (fe, fe, fe_dofs, *ftr, elmat);
                mat -> AddSubMatrix (vdofs, vdofs,
                                     elmat, skip_zeros);
            }
        }
    }
}

//! Assembles shared faces
#if MFEM_VERSION == 40100
void mymfem::MyParBilinearForm
:: MyAssembleSharedFaces(const ParGridFunction *u,
                         int skip_zeros)
{
    ParMesh *pmesh = pfes->GetParMesh();
    FaceElementTransformations *T;
    Array<int> vdofs1, vdofs2, vdofs_all;
    DenseMatrix elemmat;
    Vector fe1_dofs;

    int myrank;
    MPI_Comm_rank(pfes->GetComm(), &myrank);

    int nfaces = pmesh->GetNSharedFaces();
    for (int i = 0; i < nfaces; i++)
    {
        T = pmesh->GetSharedFaceTransformations(i);
        //cout << "\n" << myrank << "\t"
        //     << i << "\t"
        //     << nfaces << "\t"
        //     << T->Elem1No << "\t"
        //     << T->Elem2No << endl;

        pfes->GetElementVDofs(T->Elem1No, vdofs1);
        fe1_dofs.SetSize(vdofs1.Size());
        if (u) { get_dofs(*u, vdofs1, fe1_dofs); }

        pfes->GetFaceNbrElementVDofs(T->Elem2No, vdofs2);
        vdofs1.Copy(vdofs_all);

        for (int j = 0; j < vdofs2.Size(); j++)
        {
            if (vdofs2[j] >= 0) {
                vdofs2[j] += height;
            } else {
                vdofs2[j] -= height;
            }
        }
        vdofs_all.Append(vdofs2);

        for (int k = 0; k < myifbfi.Size(); k++)
        {
            myifbfi[k]->MyAssembleFaceMatrix
                    (*pfes->GetFE(T->Elem1No),
                     *pfes->GetFaceNbrFE(T->Elem2No),
                     fe1_dofs, *T, elemmat);
            if (keep_nbr_block)
            {
                mat->AddSubMatrix(vdofs_all, vdofs_all,
                                  elemmat, skip_zeros);
            }
            else
            {
                mat->AddSubMatrix(vdofs1, vdofs_all,
                                  elemmat, skip_zeros);
            }
        }
    }
}

#elif MFEM_VERSION == 40200

void mymfem::MyParBilinearForm
:: MyAssembleSharedFaces(const ParGridFunction *u,
                         int skip_zeros)
{
   ParMesh *pmesh = pfes->GetParMesh();
   FaceElementTransformations *T;
   Array<int> vdofs1, vdofs2, vdofs_all;
   DenseMatrix elemmat;
   Vector fe1_dofs;

   int nfaces = pmesh->GetNSharedFaces();
   for (int i = 0; i < nfaces; i++)
   {
      T = pmesh->GetSharedFaceTransformations(i);
      int Elem2NbrNo = T->Elem2No - pmesh->GetNE();

      pfes->GetElementVDofs(T->Elem1No, vdofs1);
      fe1_dofs.SetSize(vdofs1.Size());
      if (u) { get_dofs(*u, vdofs1, fe1_dofs); }

      pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);
      vdofs1.Copy(vdofs_all);

      for (int j = 0; j < vdofs2.Size(); j++)
      {
         if (vdofs2[j] >= 0)
         {
            vdofs2[j] += height;
         }
         else
         {
            vdofs2[j] -= height;
         }
      }
      vdofs_all.Append(vdofs2);

      for (int k = 0; k < myifbfi.Size(); k++)
      {
         myifbfi[k]->MyAssembleFaceMatrix
                 (*pfes->GetFE(T->Elem1No),
                  *pfes->GetFaceNbrFE(Elem2NbrNo),
                  fe1_dofs, *T, elemmat);
         if (keep_nbr_block)
         {
            mat->AddSubMatrix(vdofs_all, vdofs_all,
                              elemmat, skip_zeros);
         }
         else
         {
            mat->AddSubMatrix(vdofs1, vdofs_all,
                              elemmat, skip_zeros);
         }
      }
   }
}
#endif

//! Assembles in parallel
HypreParMatrix* mymfem::MyParBilinearForm
:: MyParallelAssemble()
{
    return MyParallelAssemble(mat);
}

HypreParMatrix* mymfem::MyParBilinearForm
:: MyParallelAssemble(SparseMatrix *m)
{
    OperatorHandle Mh(Operator::Hypre_ParCSR);
    MyParallelAssemble(Mh, m);
    Mh.SetOperatorOwner(false);
    return Mh.As<HypreParMatrix>();
}

void mymfem::MyParBilinearForm
:: MyParallelAssemble (OperatorHandle &A,
                       SparseMatrix *A_local)
{
    A.Clear();

    if (A_local == nullptr) { return; }
    MFEM_VERIFY(A_local->Finalized(),
                "the local matrix must be finalized")

    OperatorHandle dA(A.Type()), Ph(A.Type()), hdA;

    if (myifbfi.Size() == 0)
    {
        // construct a parallel block-diagonal matrix 'A'
        // based on 'a'
        dA.MakeSquareBlockDiag(pfes->GetComm(),
                               pfes->GlobalVSize(),
                               pfes->GetDofOffsets(),
                               A_local);
    }
    else
    {
        // handle the case when 'a' contains offdiagonal
        int lvsize = pfes->GetVSize();
        const HYPRE_Int *face_nbr_glob_ldof
                = pfes->GetFaceNbrGlobalDofMap();
        HYPRE_Int ldof_offset = pfes->GetMyDofOffset();

        Array<HYPRE_Int> glob_J(A_local->NumNonZeroElems());
        int *J = A_local->GetJ();
        for (int i = 0; i < glob_J.Size(); i++)
        {
            if (J[i] < lvsize)
            {
                glob_J[i] = J[i] + ldof_offset;
            }
            else
            {
                glob_J[i]
                        = face_nbr_glob_ldof[J[i] - lvsize];
            }
        }

        // TODO - construct dA directly in the A format
        hdA.Reset(
          new HypreParMatrix(pfes->GetComm(),
                             lvsize,
                             pfes->GlobalVSize(),
                             pfes->GlobalVSize(),
                             A_local->GetI(),
                             glob_J,
                             A_local->GetData(),
                             pfes->GetDofOffsets(),
                             pfes->GetDofOffsets()));
        // - hdA owns the new HypreParMatrix
        // - the above constructor copies all input arrays
        glob_J.DeleteAll();
        dA.ConvertFrom(hdA);
    }

    Ph.ConvertFrom(pfes->Dof_TrueDof_Matrix());
    A.MakePtAP(dA, Ph);
}

// End of file
