#include "../../include/incompNS/passembly.hpp"


#if MFEM_VERSION == 40200
//! Matrix-free application of
//! convection + upwind numerical flux operators
Vector mymfem::ApplyParMFConvNFluxOperator
:: operator()(ParGridFunction* u, ParGridFunction* w) const
{
    auto pfes = u->ParFESpace();
    pfes->ExchangeFaceNbrData();

    auto fes = u->FESpace();
    Mesh *mesh = fes -> GetMesh();

    Vector rhs(fes->GetVSize());
    rhs = 0.;

    // assemble domain integrators
    {
        Vector elvec;
        Array<int> vdofs;
        Vector u_dofs, w_dofs;
        ElementTransformation *eltrans = nullptr;

        for (int i = 0; i < fes->GetNE(); i++)
        {
            // get dof values
            fes->GetElementVDofs(i, vdofs);
            u_dofs.SetSize(vdofs.Size());
            if (u) { get_dofs(*u, vdofs, u_dofs); }
            w_dofs.SetSize(vdofs.Size());
            if (w) { get_dofs(*w, vdofs, w_dofs); }

            const FiniteElement &fe = *fes->GetFE(i);
            eltrans = fes->GetElementTransformation(i);

            AssembleElementRhsVector(fe, u_dofs, w_dofs,
                                     *eltrans, elvec);
            rhs.AddElementVector(vdofs, elvec);
        }
    }

    // assemble interior face integrators
    {
        Vector elvec;
        Array<int> vdofs, vdofs2;
        Vector u_dofs1, u_dofs2;
        Vector w_dofs;
        FaceElementTransformations *ftr = nullptr;

        int nfaces = mesh->GetNumFaces();
        for (int i = 0; i < nfaces; i++)
        {
            ftr = mesh->GetInteriorFaceTransformations (i);
            if (ftr != nullptr)
            {
                // get dof values of function w
                fes -> GetElementVDofs
                       (ftr -> Elem1No, vdofs);
                w_dofs.SetSize(vdofs.Size());
                if (w) { get_dofs(*w, vdofs, w_dofs); }

                // get dof values of function u
                fes -> GetElementVDofs
                       (ftr -> Elem2No, vdofs2);
                u_dofs1.SetSize(vdofs.Size());
                u_dofs2.SetSize(vdofs2.Size());
                if (u) {
                    get_dofs(*u, vdofs, u_dofs1);
                    get_dofs(*u, vdofs2, u_dofs2);
                }
                vdofs.Append (vdofs2);

                const FiniteElement &fe1
                        = *(fes->GetFE (ftr->Elem1No));
                const FiniteElement &fe2
                        = *(fes->GetFE (ftr->Elem2No));

                AssembleInteriorFaceRhsVector
                        (fe1, fe2, u_dofs1, u_dofs2, w_dofs,
                         *ftr, elvec);
                rhs.AddElementVector(vdofs, elvec);
            }
        }
        // assemble shared faces
        AssembleSharedFaces(u, w, rhs);
    }

    // get true dofs
    Vector tv(pfes->GetTrueVSize());
    const Operator* prolong
            = pfes->GetProlongationMatrix();
    prolong->MultTranspose(rhs, tv);

    return tv;
}

//! Assembles shared interior faces
void mymfem::ApplyParMFConvNFluxOperator
:: AssembleSharedFaces(ParGridFunction* u,
                       ParGridFunction* w,
                       Vector& rhs) const
{
    auto pfes = u->ParFESpace();
    auto pmesh = pfes->GetParMesh();

    int myrank;
    MPI_Comm_rank(pfes->GetComm(), &myrank);

    // face neighbour data
    u->ExchangeFaceNbrData();
    m_face_nbr_data = u->FaceNbrData();

    Vector elvec;
    Array<int> vdofs1, vdofs2;
    Vector u_dofs1, u_dofs2;
    Vector w_dofs;
    FaceElementTransformations *ftr = nullptr;

    int nfaces = pmesh->GetNSharedFaces();
    for (int i = 0; i < nfaces; i++)
    {
        ftr = pmesh->GetSharedFaceTransformations(i);
        int Elem2NbrNo = ftr->Elem2No - pmesh->GetNE();

        // get dof values of function w
        pfes -> GetElementVDofs
               (ftr -> Elem1No, vdofs1);
        w_dofs.SetSize(vdofs1.Size());
        if (w) { get_dofs(*w, vdofs1, w_dofs); }

        // get dof values of function u
        pfes -> GetFaceNbrElementVDofs
               (Elem2NbrNo, vdofs2);
        u_dofs1.SetSize(vdofs1.Size());
        u_dofs2.SetSize(vdofs2.Size());
        if (u) {
            get_dofs(*u, vdofs1, u_dofs1);
            get_face_nbr_dofs(vdofs2, u_dofs2);
        }

        const FiniteElement &fe1
                = *(pfes->GetFE (ftr->Elem1No));
        const FiniteElement &fe2
                = *(pfes->GetFaceNbrFE(Elem2NbrNo));

        AssembleInteriorFaceRhsVector
                (fe1, fe2, u_dofs1, u_dofs2, w_dofs,
                 *ftr, elvec);
        Vector tmp(elvec.GetData(), vdofs1.Size());
        rhs.AddElementVector(vdofs1, tmp);
    }
}

#endif

// End of file
