#include "../../include/stokes/assembly.hpp"


// Gradient Integrator
void mymfem::GradientIntegrator
:: AssembleElementMatrix2 (const FiniteElement &trial_fe,
                           const FiniteElement &test_fe,
                           ElementTransformation &Trans,
                           DenseMatrix &elmat)
{
    int dim = trial_fe.GetDim();
    int trial_nd = trial_fe.GetDof();
    int test_nd = test_fe.GetDof();

#ifdef MFEM_THREAD_SAFE
    DenseMatrix vshape(test_nd, dim);
    DenseMatrix dshape(trial_nd, dim);
#else
    vshape.SetSize(test_nd, dim);
    dshape.SetSize(trial_nd, dim);
#endif
    elmat.SetSize(test_nd, trial_nd);

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
        int order = trial_fe.GetOrder()
                + test_fe.GetOrder() - 1;
        ir = &IntRules.Get(trial_fe.GetGeomType(), order);
    }

    elmat = 0.0;
    DenseMatrix tmp(test_nd, trial_nd);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);

        trial_fe.CalcPhysDShape(Trans, dshape);
        test_fe.CalcVShape(Trans, vshape);

        double w = ip.weight*Trans.Weight();
        MultABt(vshape, dshape, tmp);
        elmat.Add(w, tmp);
    }
}


// Diffusion Integrator
void mymfem::DiffusionIntegrator
:: MyAssembleElementMatrix (const FiniteElement &el,
                            const Vector &,
                            ElementTransformation &Trans,
                            DenseMatrix &elmat) const
{
    int dim = el.GetDim();
    int ndofs = el.GetDof();

#ifdef MFEM_THREAD_SAFE
    DenseTensor gradvshape(dim, dim, ndofs);
#else
    gradvshape.SetSize(dim, dim, ndofs);
#endif
    elmat.SetSize(ndofs, ndofs);

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
       int order = 2*el.GetOrder()-2;
       ir = &IntRules.Get(el.GetGeomType(), order);
    }
    //std::cout << el.GetOrder() << "\t"
    //          << ndofs << std::endl;

    DenseMatrix vshape_buf(ndofs, dim);

    elmat = 0.0;
    Vector u(dim);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);
        el.CalcGradVShape(Trans, gradvshape);

        double coeff = ip.weight*Trans.Weight();
        MyAssembleBlock (ndofs, gradvshape, coeff, elmat);

        /*DenseMatrix J = Trans.Jacobian();
        for (int k=0; k<dim; k++) {
            for (int l=0; l<dim; l++)
                std::cout << J(k,l) << " ";
            std::cout << "\n";
        }*/
    }
}

void mymfem::DiffusionIntegrator
:: MyAssembleBlock (const int ndofs,
                    const DenseTensor &gradu,
                    const double sCoeff,
                    DenseMatrix &elmat) const
{
    // gradu(i)*gradu(j) := trace(gradu(i)^T*gradu(j))
    // i.e. inner product of matrices
    for (int j=0; j < ndofs; j++) {
        DenseMatrix col_buf = gradu(j);
        for (int i=0; i < ndofs; i++) {
            elmat(i,j) += sCoeff*(gradu(i)*col_buf);
        }
    }
}


// Diffusion Consistency Integrator
void mymfem::DiffusionConsistencyIntegrator
:: MyAssembleFaceMatrix (const FiniteElement &fe1,
                         const FiniteElement &fe2,
                         const Vector &,
                         FaceElementTransformations &ftr,
                         DenseMatrix &elmat) const
{
#ifdef MFEM_THREAD_SAFE
    DenseMatrix vshape1, vshape2;
    DenseTensor gradvshape1, gradvshape2;
    Vector nor;
#endif

    int dim = fe1.GetDim();
    int ndofs1 = fe1.GetDof();
    int ndofs2 = (ftr.Elem2No >= 0) ? fe2.GetDof() : 0;
    int ndofs = ndofs1 + ndofs2;

    elmat.SetSize(ndofs, ndofs);

    nor.SetSize(dim);
    vshape1.SetSize(ndofs1, dim);
    gradvshape1.SetSize(dim, dim, ndofs1);
    if (ndofs2) {
        vshape2.SetSize(ndofs2, dim);
        gradvshape2.SetSize(dim, dim, ndofs2);
    }

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
       int order = 2*fe1.GetOrder() - 1;
       ir = &IntRules.Get(ftr.FaceGeom, order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        ftr.Face->SetIntPoint(&ip);
        CalcOrtho(ftr.Face->Jacobian(), nor);

        IntegrationPoint eip1, eip2;
        ftr.Loc1.Transform(ip, eip1);
        ftr.Elem1->SetIntPoint(&eip1);
        fe1.CalcVShape(*(ftr.Elem1), vshape1);
        fe1.CalcGradVShape(*(ftr.Elem1), gradvshape1);
        if (ndofs2) {
            ftr.Loc2.Transform(ip, eip2);
            ftr.Elem2->SetIntPoint(&eip2);
            fe2.CalcVShape(*(ftr.Elem2), vshape2);
            fe2.CalcGradVShape(*(ftr.Elem2), gradvshape2);
        }

        /*std::cout << "\n\nvshape:" << std::endl;
        for (int k=0; k<ndofs1; k++) {
            for (int l=0; l<dim; l++)
                std::cout << vshape1(k,l) << " ";
            std::cout << "\n";
        }
        std::cout << "\n" << nor(0)
                  << "\t" << nor(1) << std::endl;*/


        double coeff = (ndofs2) ? 0.5*ip.weight : ip.weight;
        // (1,1) block
        MyAssembleBlock(dim, ndofs1, ndofs1,
                        0, 0,
                        vshape1, gradvshape1,
                        -coeff, nor, elmat);
        if (ndofs2)
        {
            // (1,2) block
            MyAssembleBlock(dim, ndofs1, ndofs2,
                            0, ndofs1,
                            vshape1, gradvshape2,
                            -coeff, nor, elmat);
            // (2,1) block
            MyAssembleBlock(dim, ndofs2, ndofs1,
                            ndofs1, 0,
                            vshape2, gradvshape1,
                            +coeff, nor, elmat);
            // (2,2) block
            MyAssembleBlock(dim, ndofs2, ndofs2,
                            ndofs1, ndofs1,
                            vshape2, gradvshape2,
                            +coeff, nor, elmat);
        }
    }

    /*std::cout << "\n\nelmat:" << std::endl;
    for (int k=0; k<ndofs; k++) {
        for (int l=0; l<ndofs; l++)
            std::cout << elmat(k,l) << " ";
        std::cout << "\n";
    }*/
}

void mymfem::DiffusionConsistencyIntegrator
:: MyAssembleBlock (const int dim,
                    const int row_ndofs,
                    const int col_ndofs,
                    const int row_offset,
                    const int col_offset,
                    const DenseMatrix &row_vshape,
                    const DenseTensor &col_gradvshape,
                    const double sCoeff,
                    const Vector &nor,
                    DenseMatrix &elmat) const
{
    DenseMatrix row_mat(dim, dim);

    for (int i=0; i < row_ndofs; i++)
    {
        // (row_vshape for i^{th} dof) \kronecker n
        for (int l=0; l<dim; l++) {
            for (int k=0; k<dim; k++) {
                row_mat(k,l) = row_vshape(i,k)*nor(l);
            }
        }
        /*std::cout << "\n\nrowmat:" << std::endl;
        for (int k=0; k<dim; k++) {
            for (int l=0; l<dim; l++)
                std::cout << row_mat(k,l) << " ";
            std::cout << "\n";
        }*/

        for (int j=0; j < col_ndofs; j++)
        {
            /*std::cout << "\n\ncol_gradvshape:"
                      << std::endl;
            for (int k=0; k<dim; k++) {
                for (int l=0; l<dim; l++)
                    std::cout << col_gradvshape(k,l,j)
                              << " ";
                std::cout << "\n";
            }*/

            // row_mat*col_gradvshape
            // := trace(row_mat^T*col_gradvshape)
            // i.e. inner product of matrices
            elmat(i+row_offset, j+col_offset)
                    += sCoeff*(row_mat*col_gradvshape(j));
        }
    }
}


// Diffusion Symmetry Integrator
void mymfem::DiffusionSymmetryIntegrator
:: MyAssembleFaceMatrix (const FiniteElement &fe1,
                         const FiniteElement &fe2,
                         const Vector &,
                         FaceElementTransformations &ftr,
                         DenseMatrix &elmat) const
{
#ifdef MFEM_THREAD_SAFE
    DenseMatrix vshape1, vshape2;
    DenseTensor gradvshape1, gradvshape2;
    Vector nor;
#endif

    int dim = fe1.GetDim();
    int ndofs1 = fe1.GetDof();
    int ndofs2 = (ftr.Elem2No >= 0) ? fe2.GetDof() : 0;
    int ndofs = ndofs1 + ndofs2;

    elmat.SetSize(ndofs, ndofs);

    nor.SetSize(dim);
    vshape1.SetSize(ndofs1, dim);
    gradvshape1.SetSize(dim, dim, ndofs1);
    if (ndofs2) {
        vshape2.SetSize(ndofs2, dim);
        gradvshape2.SetSize(dim, dim, ndofs2);
    }

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
       int order = 2*fe1.GetOrder() - 1;
       ir = &IntRules.Get(ftr.FaceGeom, order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        ftr.Face->SetIntPoint(&ip);
        CalcOrtho(ftr.Face->Jacobian(), nor);

        IntegrationPoint eip1, eip2;
        ftr.Loc1.Transform(ip, eip1);
        ftr.Elem1->SetIntPoint(&eip1);
        fe1.CalcVShape(*(ftr.Elem1), vshape1);
        fe1.CalcGradVShape(*(ftr.Elem1), gradvshape1);
        if (ndofs2) {
            ftr.Loc2.Transform(ip, eip2);
            ftr.Elem2->SetIntPoint(&eip2);
            fe2.CalcVShape(*(ftr.Elem2), vshape2);
            fe2.CalcGradVShape(*(ftr.Elem2), gradvshape2);
        }

        double coeff = (ndofs2) ? 0.5*ip.weight : ip.weight;
        // (1,1) block
        MyAssembleBlock(dim, ndofs1, ndofs1,
                        0, 0,
                        gradvshape1, vshape1,
                        -coeff, nor, elmat);
        if (ndofs2)
        {
            // (1,2) block
            MyAssembleBlock(dim, ndofs1, ndofs2,
                            0, ndofs1,
                            gradvshape1, vshape2,
                            +coeff, nor, elmat);
            // (2,1) block
            MyAssembleBlock(dim, ndofs2,
                            ndofs1, ndofs1, 0,
                            gradvshape2, vshape1,
                            -coeff, nor, elmat);
            // (2,2) block
            MyAssembleBlock(dim, ndofs2, ndofs2,
                            ndofs1, ndofs1,
                            gradvshape2, vshape2,
                            +coeff, nor, elmat);
        }
    }
}

void mymfem::DiffusionSymmetryIntegrator
:: MyAssembleBlock (const int dim,
                    const int row_ndofs,
                    const int col_ndofs,
                    const int row_offset,
                    const int col_offset,
                    const DenseTensor &row_gradvshape,
                    const DenseMatrix &col_vshape,
                    const double sCoeff,
                    const Vector &nor,
                    DenseMatrix &elmat) const
{
    DenseMatrix col_mat(dim, dim);

    for (int j=0; j < col_ndofs; j++)
    {
        // (col_vshape for j^{th} dof) \kronecker n
        for (int l=0; l<dim; l++) {
            for (int k=0; k<dim; k++) {
                col_mat(k,l) = col_vshape(j,k)*nor(l);
            }
        }

        for (int i=0; i < row_ndofs; i++)
        {
            // col_mat*row_gradvshape
            // := trace(col_mat^T*row_gradvshape)
            // i.e. inner product of matrices
            elmat(i+row_offset, j+col_offset)
                    += sCoeff*(col_mat*row_gradvshape(i));
        }
    }
}


// Diffusion Penalty Integrator
void mymfem::DiffusionPenaltyIntegrator
:: MyAssembleFaceMatrix (const FiniteElement &fe1,
                         const FiniteElement &fe2,
                         const Vector &,
                         FaceElementTransformations &ftr,
                         DenseMatrix &elmat) const
{
#ifdef MFEM_THREAD_SAFE
    DenseMatrix vshape1, vshape2;
    Vector nor;
#endif

    int dim = fe1.GetDim();
    int ndofs1 = fe1.GetDof();
    int ndofs2 = (ftr.Elem2No >= 0) ? fe2.GetDof() : 0;
    int ndofs = ndofs1 + ndofs2;

    elmat.SetSize(ndofs, ndofs);

    nor.SetSize(dim);
    vshape1.SetSize(ndofs1, dim);
    if (ndofs2) {
        vshape2.SetSize(ndofs2, dim);
    }

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
       int order = 2*fe1.GetOrder();
       ir = &IntRules.Get(ftr.FaceGeom, order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        ftr.Face->SetIntPoint(&ip);
        CalcOrtho(ftr.Face->Jacobian(), nor);
        nor /= ftr.Face->Weight(); // unit normal

        IntegrationPoint eip1, eip2;
        ftr.Loc1.Transform(ip, eip1);
        ftr.Elem1->SetIntPoint(&eip1);
        fe1.CalcVShape(*(ftr.Elem1), vshape1);
        if (ndofs2) {
            ftr.Loc2.Transform(ip, eip2);
            ftr.Elem2->SetIntPoint(&eip2);
            fe2.CalcVShape(*(ftr.Elem2), vshape2);
        }

        /*std::cout << "\n\nvshape:" << std::endl;
        for (int k=0; k<ndofs1; k++) {
            for (int l=0; l<dim; l++)
                std::cout << vshape1(k,l) << " ";
            std::cout << "\n";
        }
        std::cout << "\n" << nor(0) << "\t" << nor(1)
                  << "\t" << ftr.Face->Weight()
                  << std::endl;*/

        double coeff = ip.weight;//*ftr.Face->Weight();
        // (1,1) block
        MyAssembleBlock(dim, ndofs1, ndofs1,
                        0, 0,
                        vshape1, vshape1,
                        +coeff, nor, elmat);
        if (ndofs2)
        {
            // (1,2) block
            MyAssembleBlock(dim, ndofs1, ndofs2,
                            0, ndofs1,
                            vshape1, vshape2,
                            -coeff, nor, elmat);
            // (2,1) block
            MyAssembleBlock(dim, ndofs2,
                            ndofs1, ndofs1, 0,
                            vshape2, vshape1,
                            -coeff, nor, elmat);
            // (2,2) block
            MyAssembleBlock(dim, ndofs2, ndofs2,
                            ndofs1, ndofs1,
                            vshape2, vshape2,
                            +coeff, nor, elmat);
        }
    }
    elmat *= m_penalty;

    /*for (int m=0; m<elmat.NumRows(); m++) {
        for (int n=0; n<elmat.NumCols(); n++) {
            std::cout << elmat(m,n) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\n";*/
}

void mymfem::DiffusionPenaltyIntegrator
:: MyAssembleBlock (const int dim,
                    const int row_ndofs,
                    const int col_ndofs,
                    const int row_offset,
                    const int col_offset,
                    const DenseMatrix &row_vshape,
                    const DenseMatrix &col_vshape,
                    const double sCoeff,
                    const Vector &nor,
                    DenseMatrix &elmat) const
{
    DenseMatrix row_mat(dim, dim), col_mat(dim, dim);

    for (int j=0; j < col_ndofs; j++)
    {
        // (col_vshape for j^{th} dof) \kronecker n
        for (int l=0; l<dim; l++) {
            for (int k=0; k<dim; k++) {
                col_mat(k,l) = col_vshape(j,k)*nor(l);
            }
        }

        for (int i=0; i < row_ndofs; i++)
        {
            // (row_vshape for i^{th} dof) \kronecker n
            for (int l=0; l<dim; l++) {
                for (int k=0; k<dim; k++) {
                    row_mat(k,l) = row_vshape(i,k)*nor(l);
                }
            }

            // row_mat*col_mat := trace(row_mat^T*col_mat)
            // i.e. inner product of matrices
            elmat(i+row_offset, j+col_offset)
                    += sCoeff*(row_mat*col_mat);
        }
    }
}

/*void mymfem::DiffusionPenaltyIntegrator
:: MyAssembleBlock (const int dim,
                    const int row_ndofs,
                    const int col_ndofs,
                    const int row_offset,
                    const int col_offset,
                    const DenseMatrix &row_vshape,
                    const DenseMatrix &col_vshape,
                    const double sCoeff,
                    const Vector &nor,
                    DenseMatrix &elmat) const
{
    assert(dim == 2);

    Vector tang(dim);
    tang(0) = -nor(1);
    tang(1) = +nor(0);

    Vector row_v(row_ndofs), col_v(col_ndofs);
    row_vshape.Mult(tang, row_v);
    col_vshape.Mult(tang, col_v);

    for (int j=0; j < col_ndofs; j++) {
        for (int i=0; i < row_ndofs; i++)
        {
            elmat(i+row_offset, j+col_offset)
                    += sCoeff*(row_v(i)*col_v(j));
        }
    }
}*/


// Boundary Diffusion Consistency Integrator
void mymfem::BdryDiffusionConsistencyIntegrator
:: AssembleRHSElementVect (const FiniteElement &el,
                           FaceElementTransformations &ftr,
                           Vector &elvect)
{
    int dim = el.GetDim();
    int ndofs = el.GetDof();

#ifdef MFEM_THREAD_SAFE
    DenseTensor gradvshape(dim, dim, ndofs);
    Vector nor(dim);
    Vector buf_vec(dim);
    DenseMatrix buf_mat(dim, dim);
#else
    gradvshape.SetSize(dim, dim, ndofs);
    nor.SetSize(dim);
    buf_vec.SetSize(dim);
    buf_mat.SetSize(dim, dim);
#endif
    elvect.SetSize(ndofs);

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
       int order = 2*el.GetOrder();
       ir = &IntRules.Get(ftr.FaceGeom, order);
    }

    elvect = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        ftr.Face->SetIntPoint(&ip);
        CalcOrtho(ftr.Face->Jacobian(), nor);

        // evaluate vector coefficient
        // computes Q \kronecker nor
        Q.Eval(buf_vec, *(ftr.Face), ip);
        for (int l=0; l<dim; l++) {
            for (int k=0; k<dim; k++) {
                buf_mat(k,l) = buf_vec(k)*nor(l);
            }
        }

        /*std::cout << "\n\n";
        for (int m=0; m<nor.Size(); m++) {
            std::cout << nor(m) << " ";
        }
        std::cout << "\n";

        for (int m=0; m<nor.Size(); m++) {
            std::cout << buf_vec(m) << " ";
        }
        std::cout << "\n";*/

        IntegrationPoint eip;
        ftr.Loc1.Transform(ip, eip);
        ftr.Elem1->SetIntPoint(&eip);
        el.CalcGradVShape(*(ftr.Elem1), gradvshape);

        double coeff = -ip.weight;
        for (int j=0; j < ndofs; j++)
        {
            elvect(j) += coeff*(buf_mat*gradvshape(j));
            //std::cout << j << "\t"
            //          << coeff << "\t"
            //          << (buf_mat*gradvshape(j))
            //          << std::endl;
        }
    }

    /*for (int m=0; m<elvect.Size(); m++) {
        std::cout << elvect(m) << " ";
    }
    std::cout << "\n\n";*/
}


// Boundary Diffusion Penalty Integrator
void mymfem::BdryDiffusionPenaltyIntegrator
:: AssembleRHSElementVect (const FiniteElement &el,
                           FaceElementTransformations &ftr,
                           Vector &elvect)
{
    int dim = el.GetDim();
    int ndofs = el.GetDof();

#ifdef MFEM_THREAD_SAFE
    DenseMatrix vshape(ndofs, dim);
    Vector nor(dim);
    Vector buf_vec(dim);
    DenseMatrix buf_mat1(dim, dim);
    DenseMatrix buf_mat2(dim, dim);
#else
    vshape.SetSize(ndofs, dim);
    nor.SetSize(dim);
    buf_vec.SetSize(dim);
    buf_mat1.SetSize(dim, dim);
    buf_mat2.SetSize(dim, dim);
#endif
    elvect.SetSize(ndofs);

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
       int order = 2*el.GetOrder();
       ir = &IntRules.Get(ftr.FaceGeom, order);
    }

    elvect = 0.0;
    Vector u(dim);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        ftr.Face->SetIntPoint(&ip);
        CalcOrtho(ftr.Face->Jacobian(), nor);
        nor /= ftr.Face->Weight(); // unit normal

        // evaluate vector coefficient
        // computes Q \kronecker nor
        Q.Eval(buf_vec, *(ftr.Face), ip);
        for (int l=0; l<dim; l++) {
            for (int k=0; k<dim; k++) {
                buf_mat1(k,l) = buf_vec(k)*nor(l);
            }
        }

        IntegrationPoint eip;
        ftr.Loc1.Transform(ip, eip);
        ftr.Elem1->SetIntPoint(&eip);
        el.CalcVShape(*(ftr.Elem1), vshape);

        double coeff = ip.weight;
        for (int j=0; j < ndofs; j++)
        {
            // computes vshape.row(j) \kronecker nor
            for (int l=0; l<dim; l++) {
                for (int k=0; k<dim; k++) {
                    buf_mat2(k,l) = vshape(j,k)*nor(l);
                }
            }
            elvect(j) += coeff*(buf_mat1*buf_mat2);
        }
    }
    elvect *= m_penalty;
}

/*void mymfem::BdryDiffusionPenaltyIntegrator
:: AssembleRHSElementVect (const FiniteElement &el,
                           FaceElementTransformations &ftr,
                           Vector &elvect)
{
    int dim = el.GetDim();
    int ndofs = el.GetDof();
    assert(dim == 2);

#ifdef MFEM_THREAD_SAFE
    DenseMatrix vshape(ndofs, dim);
    Vector nor(dim);
    Vector tang(dim);
    Vector buf_vec(dim);
    DenseMatrix buf_mat1(dim, dim);
    DenseMatrix buf_mat2(dim, dim);
#else
    vshape.SetSize(ndofs, dim);
    nor.SetSize(dim);
    buf_vec.SetSize(dim);
    buf_mat1.SetSize(dim, dim);
    buf_mat2.SetSize(dim, dim);
#endif
    elvect.SetSize(ndofs);

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
       int order = 2*el.GetOrder();
       ir = &IntRules.Get(ftr.FaceGeom, order);
    }

    elvect = 0.0;
    Vector u(dim);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        ftr.Face->SetIntPoint(&ip);
        CalcOrtho(ftr.Face->Jacobian(), nor);
        nor /= ftr.Face->Weight(); // unit normal

        tang(0) = -nor(1);
        tang(1) = +nor(0);

        // evaluate vector coefficient
        // computes Q \kronecker nor
        Q.Eval(buf_vec, *(ftr.Face), ip);
        double q_tang = buf_vec*tang;

        IntegrationPoint eip;
        ftr.Loc1.Transform(ip, eip);
        ftr.Elem1->SetIntPoint(&eip);
        el.CalcVShape(*(ftr.Elem1), vshape);

        Vector v(ndofs);
        vshape.Mult(tang, v);

        double coeff = ip.weight;
        for (int j=0; j < ndofs; j++) {
            elvect(j) += coeff*v(j)*q_tang;
        }
    }
    elvect *= m_penalty;
}*/


// Open boundary integrator
void mymfem::OpenBdryIntegrator
:: MyAssembleFaceMatrix (const FiniteElement &trial_fe,
                         const FiniteElement &test_fe,
                         FaceElementTransformations &ftr,
                         DenseMatrix &elmat) const
{
#ifdef MFEM_THREAD_SAFE
    Vector shape;
    DenseMatrix vshape;
    Vector nor;
#endif
    int dim = trial_fe.GetDim();
    int trial_ndofs = trial_fe.GetDof();
    int test_ndofs = test_fe.GetDof();

    elmat.SetSize(test_ndofs, trial_ndofs);
    elmat = 0.0;

    //cout << "\n\n" << trial_fe.GetOrder() << "\t"
    //     << trial_ndofs << endl;

    nor.SetSize(dim);
    shape.SetSize(trial_ndofs);
    vshape.SetSize(test_ndofs, dim);
    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
       int order = trial_fe.GetOrder() + test_fe.GetOrder();
       ir = &IntRules.Get(ftr.FaceGeom, order);
    }

    Vector vn (test_ndofs);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        ftr.Face->SetIntPoint(&ip);
        CalcOrtho(ftr.Face->Jacobian(), nor);

        IntegrationPoint eip;
        ftr.Loc1.Transform(ip, eip);
        ftr.Elem1->SetIntPoint(&eip);
        trial_fe.CalcShape(eip, shape);
        test_fe.CalcVShape(*(ftr.Elem1), vshape);

        vshape.Mult(nor, vn);
        for (int l=0; l<trial_ndofs; l++) {
            for (int k=0; k<test_ndofs; k++) {
                elmat(k,l) += ip.weight*vn(k)*shape(l);
            }
        }
    }
}


// End of file
