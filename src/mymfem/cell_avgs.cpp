#include "../../include/mymfem/cell_avgs.hpp"
#include "../../include/mymfem/utilities.hpp"


//! Constructors
CellAverages :: CellAverages (Mesh *mesh)
{
    // piece-wise constant function space
    m_l2_coll = new L2_FECollection(0, mesh->Dimension());
    m_sfes0 = new FiniteElementSpace(mesh, m_l2_coll);
}

CellAverages :: CellAverages (FiniteElementSpace *vfes)
{
    // piece-wise constant function space
    auto mesh = vfes->GetMesh();
    m_l2_coll = new L2_FECollection(0, mesh->Dimension());
    m_sfes0 = new FiniteElementSpace(mesh, m_l2_coll);
}

//! Evaluates x-component and y-component of velocity
std::pair<Vector, Vector> CellAverages
:: eval(GridFunction *v)
{
    return {eval_xComponent(v), eval_yComponent(v)};
}

//! Evaluates x-component of velocity
Vector CellAverages
:: eval_xComponent(GridFunction *v)
{
    // unit vector in x
    Vector nx(2);
    nx = 0.;
    nx(0) = 1;

    return std::move(compute(v, nx));
}

//! Evaluates y-component of velocity
Vector CellAverages
:: eval_yComponent(GridFunction *v)
{
    // unit vector in y
    Vector ny(2);
    ny = 0.;
    ny(1) = 1;

    return std::move(compute(v, ny));
}

//! Evaluates velocity component in direction n
Vector CellAverages
:: compute(const GridFunction *v, Vector& n)
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

        if (std::fabs(n(1) - 1) < 1E-8
             && (avg > 0.1 || avg < -0.1))
        {
            int geom = vfes->GetFE(i)->GetGeomType();
            auto trans = vfes->GetElementTransformation(i);
            auto ip = Geometries.GetCenter(geom);
            trans->SetIntPoint(&ip);
            Vector coords(2);
            trans->Transform(ip, coords);
            std::cout << i << "\t" << avg << std::endl;
            coords.Print();
        }
    }

    return Vn;
}

//! Evaluates velocity component in direction n
//! for given element
double CellAverages
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

    double vnAvg = 0;
    double area = 0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        trans.SetIntPoint(&ip);

        el.CalcVShape(trans, vshape);
        vshape.MultTranspose(el_dofs, u);

        double coeff = ip.weight*trans.Weight();
        vnAvg += coeff*(u*n);
        area += coeff;
    }
    vnAvg /= area;

    return vnAvg;
}

//! Evaluates x-component and y-component of velocity
/*std::pair<Vector, Vector> CellAverages
:: eval(GridFunction *v)
{
    m_vel.reset();
    m_vel = std::make_unique<VelocityFunction>
            (m_sfes0, v->FESpace());

    std::shared_ptr <GridFunction> vx
            = std::make_shared<GridFunction>(m_sfes0);
    std::shared_ptr <GridFunction> vy
            = std::make_shared<GridFunction>(m_sfes0);
    (*m_vel)(*v, *vx, *vy);

    Vector Vx, Vy;
    vx->GetTrueDofs(Vx);
    vy->GetTrueDofs(Vy);

    return {Vx, Vy};
}*/

// End of file
