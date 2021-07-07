#include "../../include/incompNS/assembly.hpp"
#include "../../include/mymfem/utilities.hpp"


//! Convection Integrator
void mymfem::ConvectionIntegrator
:: MyAssembleElementMatrix (const FiniteElement &el,
                            const Vector &el_dofs,
                            ElementTransformation &Trans,
                            DenseMatrix &elmat) const
{
    int dim = el.GetDim();
    int ndofs = el.GetDof();
     
#ifdef MFEM_THREAD_SAFE
    DenseMatrix vshape(ndofs, dim);
    DenseTensor gradvshape(dim, dim, ndofs);
#else
    vshape.SetSize(ndofs, dim);
    gradvshape.SetSize(dim, dim, ndofs);
#endif
    elmat.SetSize(ndofs, ndofs);
    
    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
       int order = 3*el.GetOrder()-1;
       ir = &IntRules.Get(el.GetGeomType(), order);
    }
    //cout << el.GetOrder() << "\t" << ndofs << endl;
    
    elmat = 0.0;
    Vector u(dim);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);

        el.CalcVShape(Trans, vshape);
        el.CalcGradVShape(Trans, gradvshape);
        vshape.MultTranspose(el_dofs, u);

        double coeff = ip.weight*Trans.Weight();
        MyAssembleBlock (dim, ndofs, ndofs,
                         vshape, gradvshape,
                         u, coeff, elmat);
    }
}

void mymfem::ConvectionIntegrator
:: MyAssembleBlock (const int dim,
                    const int row_ndofs,
                    const int col_ndofs,
                    const DenseMatrix &row_vshape,
                    const DenseTensor &col_gradvshape,
                    const Vector vCoeff,
                    const double sCoeff,
                    DenseMatrix &elmat) const
{
    Vector graduMultVCoeff(dim);
    Vector v(row_ndofs);
    for (int j=0; j < col_ndofs; j++)
    {
        // gradu * vCoeff
        col_gradvshape(j).Mult(vCoeff, graduMultVCoeff);
        row_vshape.Mult(graduMultVCoeff, v);
        for (int i=0; i < row_ndofs; i++)
            elmat(i, j) += sCoeff*v(i);
    }
}


//! Central Numerical Flux Integrator
void mymfem::CentralNumFluxIntegrator
:: MyAssembleFaceMatrix (const FiniteElement &fe1,
                         const FiniteElement &fe2,
                         const Vector &fe1_dofs,
                         FaceElementTransformations &ftr,
                         DenseMatrix &elmat) const
{
#ifdef MFEM_THREAD_SAFE
    DenseMatrix vshape1, vshape2;
    Vector nor, u;
#endif

    int dim = fe1.GetDim();
    int ndofs1 = fe1.GetDof();
    int ndofs2 = (ftr.Elem2No >= 0) ? fe2.GetDof() : 0;
    int ndofs = ndofs1 + ndofs2;

    elmat.SetSize(ndofs, ndofs);
    elmat = 0.0;

    //cout << "\n\n" << fe1.GetOrder() << "\t"
    //     << ndofs << endl;

    u.SetSize(dim);
    nor.SetSize(dim);
    vshape1.SetSize(ndofs1, dim);
    if (ndofs2)
        vshape2.SetSize(ndofs2, dim);

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
       int order = 3*std::max(fe1.GetOrder(), ndofs2 ?
                                  fe2.GetOrder() : 0);
       ir = &IntRules.Get(ftr.FaceGeom, order);
    }

    IntegrationPoint eip1, eip2;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        ftr.Face->SetIntPoint(&ip);
        CalcOrtho(ftr.Face->Jacobian(), nor);

        ftr.Loc1.Transform(ip, eip1);
        ftr.Elem1->SetIntPoint(&eip1);
        fe1.CalcVShape(*(ftr.Elem1), vshape1);
        vshape1.MultTranspose(fe1_dofs, u);
        if (ndofs2) {
            ftr.Loc2.Transform(ip, eip2);
            ftr.Elem2->SetIntPoint(&eip2);
            fe2.CalcVShape(*(ftr.Elem2), vshape2);
        }

        double coeff = 0.5*ip.weight*(u*nor);

        // (1,1) block
        MyAssembleBlock(dim, ndofs1, ndofs1,
                        0, 0,
                        vshape1, vshape1, -coeff, elmat);

        if (ndofs2) {
            // (1,2) block
            MyAssembleBlock(dim, ndofs1, ndofs2,
                            0, ndofs1,
                            vshape1, vshape2, +coeff, elmat);
            // (2,1) block
            MyAssembleBlock(dim, ndofs2, ndofs1,
                            ndofs1, 0,
                            vshape2, vshape1, -coeff, elmat);
            // (2,2) block
            MyAssembleBlock(dim, ndofs2, ndofs2,
                            ndofs1, ndofs1,
                            vshape2, vshape2, +coeff, elmat);
        }
    }
}

void mymfem::CentralNumFluxIntegrator
:: MyAssembleBlock (const int dim,
                    const int row_ndofs,
                    const int col_ndofs,
                    const int row_offset,
                    const int col_offset,
                    const DenseMatrix &row_vshape,
                    const DenseMatrix &col_vshape,
                    const double sCoeff,
                    DenseMatrix &elmat) const
{
    DenseMatrix rvMultcv(row_ndofs, col_ndofs);
    MultABt(row_vshape, col_vshape, rvMultcv);
    elmat.AddMatrix(sCoeff, rvMultcv,
                    row_offset, col_offset);

    /*for (int j=0; j < col_ndofs; j++)
    {
        for (int i=0; i < row_ndofs; i++) {
            double dot=0;
            for (int k=0; k < dim; k++) {
                dot += row_vshape(i,k)*col_vshape(j,k);
            }
            elmat(i+row_offset, j+col_offset) += sCoeff*dot;
        }
    }*/
}


//! Upwind Numerical Flux Integrator
void mymfem::UpwindNumFluxIntegrator
:: MyAssembleFaceMatrix (const FiniteElement &fe1,
                         const FiniteElement &fe2,
                         const Vector &fe1_dofs,
                         FaceElementTransformations &ftr,
                         DenseMatrix &elmat) const
{
#ifdef MFEM_THREAD_SAFE
    DenseMatrix vshape1, vshape2;
    Vector nor, u;
#endif
    int dim = fe1.GetDim();
    int ndofs1 = fe1.GetDof();
    int ndofs2 = (ftr.Elem2No >= 0) ? fe2.GetDof() : 0;
    int ndofs = ndofs1 + ndofs2;

    elmat.SetSize(ndofs, ndofs);
    elmat = 0.0;

    //cout << "\n\n" << fe1.GetOrder() << "\t"
    //     << ndofs << endl;

    u.SetSize(dim);
    nor.SetSize(dim);
    vshape1.SetSize(ndofs1, dim);
    if (ndofs2)
        vshape2.SetSize(ndofs2, dim);

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
       int order = 3*std::max(fe1.GetOrder(), ndofs2 ?
                                  fe2.GetOrder() : 0);
       ir = &IntRules.Get(ftr.FaceGeom, order);
    }

    IntegrationPoint eip1, eip2;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        ftr.Face->SetIntPoint(&ip);
        CalcOrtho(ftr.Face->Jacobian(), nor);

        ftr.Loc1.Transform(ip, eip1);
        ftr.Elem1->SetIntPoint(&eip1);
        fe1.CalcVShape(*(ftr.Elem1), vshape1);
        vshape1.MultTranspose(fe1_dofs, u);
        if (ndofs2) {
            ftr.Loc2.Transform(ip, eip2);
            ftr.Elem2->SetIntPoint(&eip2);
            fe2.CalcVShape(*(ftr.Elem2), vshape2);
        }

        /*double w = ip.weight;
        double coeff1 = fabs(u*nor);
        double coeff2 = 0.5*(u*nor);

        // (1,1) block
        MyAssembleBlock(dim, ndofs1, ndofs1,
                        0, 0,
                        vshape1, vshape1,
                        +w*(coeff1-coeff2), elmat);

        if (ndofs2) {
            // (1,2) block
            MyAssembleBlock(dim, ndofs1, ndofs2,
                            0, ndofs1,
                            vshape1, vshape2,
                            -w*(coeff1-coeff2), elmat);
            // (2,1) block
            MyAssembleBlock(dim, ndofs2, ndofs1,
                            ndofs1, 0,
                            vshape2, vshape1,
                            -w*(coeff1+coeff2), elmat);
            // (2,2) block
            MyAssembleBlock(dim, ndofs2, ndofs2,
                            ndofs1, ndofs1,
                            vshape2, vshape2,
                            +w*(coeff1+coeff2), elmat);
        }*/

        double un = (u*nor);
        double coeff1 = ip.weight*(fabs(un) - 0.5*un);
        double coeff2 = ip.weight*(fabs(un) + 0.5*un);

        MyAssembleBlock(dim, ndofs1, ndofs1,
                        0, 0,
                        vshape1, vshape1,
                        coeff1, elmat);

        if (ndofs2) {
            // (1,2) block
            MyAssembleBlock(dim, ndofs1, ndofs2,
                            0, ndofs1,
                            vshape1, vshape2,
                            -coeff1, elmat);
            // (2,1) block
            MyAssembleBlock(dim, ndofs2, ndofs1,
                            ndofs1, 0,
                            vshape2, vshape1,
                            -coeff2, elmat);
            // (2,2) block
            MyAssembleBlock(dim, ndofs2, ndofs2,
                            ndofs1, ndofs1,
                            vshape2, vshape2,
                            +coeff2, elmat);
        }
    }
}

void mymfem::UpwindNumFluxIntegrator
:: MyAssembleBlock (const int dim,
                    const int row_ndofs,
                    const int col_ndofs,
                    const int row_offset,
                    const int col_offset,
                    const DenseMatrix &row_vshape,
                    const DenseMatrix &col_vshape,
                    const double sCoeff,
                    DenseMatrix &elmat) const
{
    DenseMatrix rvMultcv(row_ndofs, col_ndofs);
    MultABt(row_vshape, col_vshape, rvMultcv);
    elmat.AddMatrix(sCoeff, rvMultcv,
                    row_offset, col_offset);

    /*for (int j=0; j < col_ndofs; j++)
    {
        for (int i=0; i < row_ndofs; i++) {
            double dot=0;
            for (int k=0; k < dim; k++)
                dot += row_vshape(i,k)*col_vshape(j,k);

            elmat(i+row_offset, j+col_offset) += sCoeff*dot;
        }
    }*/
}


//! Matrix-free application of
//! convection + upwind numerical flux operators
Vector mymfem::ApplyMFConvNFluxOperator
:: operator()(Vector &U, Vector &W,
              FiniteElementSpace *fes) const
{
    auto u = new GridFunction{};
    u->MakeRef(fes, U);

    auto w = new GridFunction{};
    w->MakeRef(fes, W);

    auto B = (*this)(u, w);
    delete u;
    delete w;

    return B;
}

Vector mymfem::ApplyMFConvNFluxOperator
:: operator()(Vector &U, GridFunction* w) const
{
    auto fes = w->FESpace();

    auto u = new GridFunction{};
    u->MakeRef(fes, U);
    auto B = (*this)(u, w);
    delete u;

    return B;
}

Vector mymfem::ApplyMFConvNFluxOperator
:: operator()(GridFunction* u, GridFunction* w) const
{
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
                fes -> GetElementVDofs
                        (ftr -> Elem1No, vdofs);
                w_dofs.SetSize(vdofs.Size());
                if (w) { get_dofs(*w, vdofs, w_dofs); }

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
    }

    return std::move(rhs);
}

void mymfem::ApplyMFConvNFluxOperator
:: AssembleElementRhsVector (const FiniteElement &el,
                             const Vector &u_dofs,
                             const Vector &w_dofs,
                             ElementTransformation &Trans,
                             Vector &elvec) const
{
    int dim = el.GetDim();
    int ndofs = el.GetDof();

#ifdef MFEM_THREAD_SAFE
    DenseTensor gradvshape(dim, dim, ndofs);
    DenseMatrix vshape(ndofs, dim);
    DenseMatrix gradu(dim, dim);
    Vector w(dim);
#else
    gradvshape.SetSize(dim, dim, ndofs);
    vshape.SetSize(ndofs, dim);
    gradu.SetSize(dim, dim);
    w.SetSize(dim);
#endif
    elvec.SetSize(ndofs);

    int order = 3*el.GetOrder()-1;
    const IntegrationRule *ir
            = &IntRules.Get(el.GetGeomType(), order);
    //cout << el.GetOrder() << "\t" << ndofs << endl;

    elvec = 0.0;

    Vector buf(ndofs);
    Vector graduMultW(dim);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);
        el.CalcVShape(Trans, vshape);
        vshape.MultTranspose(w_dofs, w);
        el.CalcGradVShape(Trans, gradvshape);
        gradu = 0.;
        for (int j=0; j<ndofs; j++)
            gradu.Add(u_dofs(j), gradvshape(j));

        double coeff = ip.weight*Trans.Weight();
        gradu.Mult(w, graduMultW);
        vshape.Mult(graduMultW, buf);
        elvec.Add(coeff, buf);
    }
}

void mymfem::ApplyMFConvNFluxOperator
:: AssembleInteriorFaceRhsVector
(const FiniteElement &fe1, const FiniteElement &fe2,
 const Vector &u_dofs1, const Vector &u_dofs2,
 const Vector &w_dofs,
 FaceElementTransformations &ftr, Vector &elvec) const
{
#ifdef MFEM_THREAD_SAFE
    DenseMatrix vshape1, vshape2;
    Vector nor, u, w;
#endif

    int dim = fe1.GetDim();
    int ndofs1 = fe1.GetDof();
    int ndofs2 = (ftr.Elem2No >= 0) ? fe2.GetDof() : 0;
    int ndofs = ndofs1 + ndofs2;

    elvec.SetSize(ndofs);
    elvec = 0.;

    u.SetSize(dim);
    w.SetSize(dim);
    nor.SetSize(dim);
    vshape1.SetSize(ndofs1, dim);
    if (ndofs2)
        vshape2.SetSize(ndofs2, dim);

    int order = 3*std::max(fe1.GetOrder(), ndofs2 ?
                               fe2.GetOrder() : 0);
    const IntegrationRule *ir
            = &IntRules.Get(ftr.FaceGeom, order);

    IntegrationPoint eip1, eip2;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        ftr.Face->SetIntPoint(&ip);
        CalcOrtho(ftr.Face->Jacobian(), nor);

        ftr.Loc1.Transform(ip, eip1);
        ftr.Elem1->SetIntPoint(&eip1);
        fe1.CalcVShape(*(ftr.Elem1), vshape1);
        vshape1.MultTranspose(w_dofs, w);
        if (ndofs2) {
            ftr.Loc2.Transform(ip, eip2);
            ftr.Elem2->SetIntPoint(&eip2);
            fe2.CalcVShape(*(ftr.Elem2), vshape2);
        }

        double wn = (w*nor);
        double coeff1 = ip.weight*(fabs(wn) - 0.5*wn);
        double coeff2 = ip.weight*(fabs(wn) + 0.5*wn);

        vshape1.MultTranspose(u_dofs1, u);
        AssembleBlockVector(ndofs1, 0,
                            vshape1, u,
                            coeff1, elvec);

        if (ndofs2) {
            // (2,1) block
            AssembleBlockVector(ndofs2, ndofs1,
                                vshape2, u,
                                -coeff2, elvec);

            vshape2.MultTranspose(u_dofs2, u);
            // (1,2) block
            AssembleBlockVector(ndofs1, 0,
                                vshape1, u,
                                -coeff1, elvec);
            // (2,2) block
            AssembleBlockVector(ndofs2, ndofs1,
                                vshape2, u,
                                +coeff2, elvec);
        }
    }
}

void mymfem::ApplyMFConvNFluxOperator
:: AssembleBlockVector (const int ndofs,
                        const int offset,
                        const DenseMatrix &vshape,
                        const Vector& u,
                        const double sCoeff,
                        Vector &elvec) const
{
    Vector vshapeMultU(ndofs);
    vshape.Mult(u, vshapeMultU);
    for (int j=0; j < ndofs; j++) {
        elvec(j+offset) += sCoeff*vshapeMultU(j);
    }
}


// End of file
