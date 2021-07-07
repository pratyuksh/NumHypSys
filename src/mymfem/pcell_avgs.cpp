#include "../../include/mymfem/pcell_avgs.hpp"
#include "../../include/mymfem/utilities.hpp"


//! Constructors
ParCellAverages
:: ParCellAverages (ParMesh *pmesh)
{
    // piece-wise constant function space
    m_l2_coll = new L2_FECollection(0, pmesh->Dimension());
    m_sfes0 = new ParFiniteElementSpace(pmesh, m_l2_coll);
}

ParCellAverages
:: ParCellAverages (ParFiniteElementSpace *vfes)
{
    // piece-wise constant function space
    auto pmesh = vfes->GetParMesh();
    m_l2_coll = new L2_FECollection(0, pmesh->Dimension());
    m_sfes0 = new ParFiniteElementSpace(pmesh, m_l2_coll);
}

//! Evaluates x-component of velocity
Vector ParCellAverages
:: eval_xComponent(const ParGridFunction *v)
{
    // unit vector in x
    Vector nx(2);
    nx = 0.;
    nx(0) = 1;

    return std::move(compute(v, nx));
}

//! Evaluates y-component of velocity
Vector ParCellAverages
:: eval_yComponent(const ParGridFunction *v)
{
    // unit vector in y
    Vector ny(2);
    ny = 0.;
    ny(1) = 1;

    return std::move(compute(v, ny));
}

//! Evaluates velocity component in direction n
Vector ParCellAverages
:: compute(const ParGridFunction *v, Vector& n)
{
    Vector Vn(m_sfes0->GetTrueVSize());
    Array<int> sdofs;
    Vector el_sdofs;
    Vector shape;

    auto vfes = v->FESpace();
    Array<int> vdofs;
    Vector el_vdofs;

    for (int i=0; i<m_sfes0->GetNE(); i++)
    {
        vfes->GetElementVDofs(i, vdofs);
        el_vdofs.SetSize(vdofs.Size());
        get_dofs(*v, vdofs, el_vdofs);

        auto avg = compute_fe
                (*vfes->GetFE(i), el_vdofs,
                 *vfes->GetElementTransformation(i), n);

        const FiniteElement &fe = *(m_sfes0->GetFE(i));
        IntegrationPoint ip;
        ip.Set2(0, 0);

        m_sfes0->GetElementVDofs(i, sdofs);
        shape.SetSize(sdofs.Size());
        fe.CalcShape(ip, shape);

        Vn.SetSubVector(sdofs, avg/shape(0));
    }

    return Vn;
}

//! Evaluates velocity component in direction n
//! for given element
double ParCellAverages
:: compute_fe(const FiniteElement& el,
              const Vector &el_dofs,
              ElementTransformation& trans,
              Vector& n)
{
    int dim = el.GetDim();
    int ndofs = el.GetDof();

    Vector u(dim);
    DenseMatrix vshape(ndofs, dim);

    int order = el.GetOrder();
    const IntegrationRule *ir
            = &IntRules.Get(el.GetGeomType(), order);

    double vxAvg = 0;
    double area = 0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        trans.SetIntPoint(&ip);

        el.CalcVShape(trans, vshape);
        vshape.MultTranspose(el_dofs, u);

        double coeff = ip.weight*trans.Weight();
        vxAvg += coeff*(u*n);
        area += coeff;
    }
    vxAvg /= area;

    return vxAvg;
}

// End of file
