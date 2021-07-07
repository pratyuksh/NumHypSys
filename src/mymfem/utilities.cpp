#include "../../include/mymfem/utilities.hpp"
#include "../../include/mymfem/assembly.hpp"

#include <assert.h>


//! Zero function
void zeroFn(const Vector&, Vector& f) {
   f = 0.0;
}

//! Returns cell center of element i of given mesh
void get_element_center(std::shared_ptr<Mesh>& mesh,
                        int i, Vector& center)
{
    int geom = mesh->GetElementBaseGeometry(i);
    ElementTransformation *trans
            = mesh->GetElementTransformation(i);
    trans->Transform(Geometries.GetCenter(geom), center);
}

//! Computes inner product of two dense matrices
double MatMatInnerProd(const DenseMatrix &A,
                       const DenseMatrix &B)
{
    assert(A.NumRows() == B.NumRows());
    assert(A.NumCols() == B.NumCols());

    double  val = 0;
    for (int j=0; j<A.NumCols(); j++)
        for (int i=0; i<A.NumRows(); i++)
            val += A(i,j)*B(i,j);

    return val;
}

//! Returns the upper-triangular, including the diagonal,
//! for the input matrix A
SparseMatrix& get_upper_triangle(const SparseMatrix& A)
{
    int numRows = A.NumRows();
    int numCols = A.NumCols();
    const int *iA = A.GetI();
    const int *jA = A.GetJ();
    const double *dA = A.GetData();

    SparseMatrix* UA = new SparseMatrix(numRows, numCols);
    for (int i=0; i<numRows; i++) {
        for(int k=iA[i]; k<iA[i+1]; k++) {
            int j = jA[k];
            if (j >= i) { UA->_Add_(i, j, dA[k]); }
        }
    }
    UA->Finalize();

    return *UA;
}

//! Computes the area of a domain
double compute_area (const FiniteElementSpace& fes)
{
    double area = 0;
    Mesh *mesh = fes.GetMesh();

    int geom_type = fes.GetFE(0)->GetGeomType();
    int order = fes.GetOrder(0);
    std::unique_ptr <const IntegrationRule> ir;
    ir = std::make_unique<IntegrationRule>
            (IntRules.Get(geom_type, order));

    int n = ir->GetNPoints();
    for (int i=0; i < mesh->GetNE(); i++)
    {
        ElementTransformation *trans
                = mesh->GetElementTransformation(i);

        for (int k=0; k < n; k++)
        {
            const IntegrationPoint &ip = ir->IntPoint(k);
            trans->SetIntPoint(&ip);

            double coeff = ip.weight*trans->Weight();
            area += coeff;
        }
    }

    return area;
}

//! Computes the mean of a function
double compute_mean (const GridFunction& u)
{
    double mean = 0;
    double area = 0;
    const FiniteElementSpace *fes = u.FESpace();
    Mesh *mesh = fes->GetMesh();

    int geom_type = fes->GetFE(0)->GetGeomType();
    int order = fes->GetOrder(0);
    std::unique_ptr <const IntegrationRule> ir;
    ir = std::make_unique<IntegrationRule>
            (IntRules.Get(geom_type, order));

    int n = ir->GetNPoints();
    for (int i=0; i < mesh->GetNE(); i++)
    {
        ElementTransformation *trans
                = mesh->GetElementTransformation(i);

        for (int k=0; k < n; k++)
        {
            const IntegrationPoint &ip = ir->IntPoint(k);
            trans->SetIntPoint(&ip);

            double coeff = ip.weight*trans->Weight();
            mean += coeff*u.GetValue(i, ip);
            area += coeff;
        }
    }

    return mean/area;
}

//! Read dofs for Raviart-Thomas Spaces from MFEM
void get_dofs(const GridFunction &u,
              const Array<int> &vdofs,
              Vector &el_dofs)
{
    int idof, s;
    //std::cout << "\n\n";
    for (int k=0; k<vdofs.Size(); k++) {
        if (vdofs[k] >= 0) {
            idof = vdofs[k];
            s = +1;
        }
        else {
            idof = -1-vdofs[k];
            s = -1;
        }
        el_dofs(k) = s*u.Elem(idof);
        //std::cout << k << "\t"
        //          << vdofs[k] << "\t"
        //          << idof << "\t" << s << std::endl;
    }
}

void get_dofs(const Vector &u,
              const Array<int> &vdofs,
              Vector &el_dofs)
{
    int idof, s;
    for (int k=0; k<vdofs.Size(); k++) {
        if (vdofs[k] >= 0) {
            idof = vdofs[k];
            s = +1;
        }
        else {
            idof = -1-vdofs[k];
            s = -1;
        }
        el_dofs(k) = s*u.Elem(idof);
    }
}

//! Projects the normal component at the boundary
//! for the given function
/*void project_bdry_coefficient_normal
(GridFunction &u, VectorCoefficient &vCoeff,
 const Array<int> &bdr_attr)
{
    // implementation for the case when the face dofs are
    // scaled point values of the normal component.
    int dim = vCoeff.GetVDim();
    FiniteElementSpace *fes = u.FESpace();

    Vector vC(dim), nor(dim), lvec;
    Array<int> dofs;

    const FiniteElement *fe = nullptr;
    ElementTransformation *trans = nullptr;

    for (int i = 0; i < fes->GetNBE(); i++)
    {
       if (bdr_attr[fes->GetBdrAttribute(i)-1] == 0)
       {
          continue;
       }
       fe = fes->GetBE(i);
       trans = fes->GetBdrElementTransformation(i);
       const IntegrationRule &ir = fe->GetNodes();
       lvec.SetSize(fe->GetDof());
       for (int j = 0; j < ir.GetNPoints(); j++)
       {
          const IntegrationPoint &ip = ir.IntPoint(j);
          trans->SetIntPoint(&ip);
          vCoeff.Eval(vC, *trans, ip);
          CalcOrtho(trans->Jacobian(), nor);
          lvec(j) = (vC * nor);
       }
       lvec.Print();
       fes->GetBdrElementDofs(i, dofs);
       u.SetSubVector(dofs, lvec);
    }
}*/
void project_bdry_coefficient_normal
(GridFunction &u, VectorCoefficient &vCoeff,
 const Array<int> &bdr_attr)
{
    int dim = vCoeff.GetVDim();
    FiniteElementSpace *fes = u.FESpace();
    Mesh *mesh = fes -> GetMesh();

    Vector vC(dim), nor(dim);
    Array<int> dofs;

    Vector vnshape, eldofs, elvec;
    DenseMatrix vshape, elmat;

    const FiniteElement *fe = nullptr;
    FaceElementTransformations *ftr = nullptr;

    fes->BuildDofToArrays();
    int nbfaces = mesh->GetNBE();
    for (int i = 0; i < nbfaces; i++)
    {
        if (bdr_attr[mesh->GetBdrAttribute(i)-1] == 0) {
           continue;
        }

        ftr = mesh -> GetBdrFaceTransformations(i);
        fe = fes->GetFE (ftr->Elem1No);

        int ndofs = fe->GetDof();
        vnshape.SetSize(ndofs);
        vshape.SetSize(ndofs, dim);
        elvec.SetSize(ndofs);
        elmat.SetSize(ndofs, ndofs);

        int order = 2*fe->GetOrder();
        const IntegrationRule &ir
                = IntRules.Get(ftr->FaceGeom, order);

        elvec = 0.0;
        elmat = 0.0;
        for (int j = 0; j < ir.GetNPoints(); j++)
        {
             const IntegrationPoint &ip = ir.IntPoint(j);
             ftr->Face->SetIntPoint(&ip);
             CalcOrtho(ftr->Face->Jacobian(), nor);
             nor *= (1./ftr->Face->Weight());

             IntegrationPoint eip;
             ftr->Loc1.Transform(ip, eip);
             ftr->Elem1->SetIntPoint(&eip);
             fe->CalcVShape(*(ftr->Elem1), vshape);
             vshape.Mult(nor, vnshape);

             double w = ip.weight*ftr->Face->Weight();
             // elmat
             for (int l=0; l<ndofs; l++) {
                 for (int k=0; k<ndofs; k++) {
                     elmat(k,l) += w*vnshape(k)*vnshape(l);
                 }
             }
             // elvec
             vCoeff.Eval(vC, *(ftr->Face), ip);
             elvec.Add(w*(vC*nor), vnshape);
        }

        int count = 0;
        double tol = 1E-12;
        int p = fe->GetOrder();
        eldofs.SetSize(p);
        for (int j=0; j<ndofs; j++) {
            if (std::abs(elmat(j,j)) > tol) {
                eldofs[count++] = elvec(j)/elmat(j,j);
                //std::cout << j << "\t"
                //          << elmat(j,j) << "\t"
                //          << elvec(j) << "\t"
                //          << elvec(j)/elmat(j,j)
                //          << std::endl;
            }
        }

        fes->GetBdrElementDofs(i, dofs);
        u.SetSubVector(dofs, eldofs);
    }
}

//! Velocity components function
//! Initializes components
void VelocityFunction :: init ()
{
    m_xComp.SetSize(m_sfes->GetTrueVSize());
    m_yComp.SetSize(m_sfes->GetTrueVSize());

    Vector *temp1=nullptr, *temp2=nullptr; // dummy

     // Assemble mass matrix
    BilinearForm *mass_form
            = new BilinearForm(m_sfes);
    mass_form->AddDomainIntegrator(new MassIntegrator);
    mass_form->Assemble();
    mass_form->Finalize();
    m_mass = mass_form->LoseMat();
    delete mass_form;

    // Assemble x-component L2-projection matrix
    // \int_{\Omega} (u_h * nx) w_h dx
    MixedBilinearForm *xProj_form
            = new MixedBilinearForm(m_vfes,
                                    m_sfes);
    xProj_form->AddDomainIntegrator
            (new mymfem::VelocityXProjectionIntegrator);
    xProj_form->Assemble();
    xProj_form->Finalize();
    if (m_ess_bdr_marker.Size()) {
        xProj_form->EliminateTrialDofs(m_ess_bdr_marker,
                                       *temp1, *temp2);
    }
    m_xProj = xProj_form->LoseMat();
    delete xProj_form;

    // Assemble y-component L2-projection matrix
    // \int_{\Omega} (u_h * ny) w_h dx
    MixedBilinearForm *yProj_form
            = new MixedBilinearForm(m_vfes,
                                    m_sfes);
    yProj_form->AddDomainIntegrator
            (new mymfem::VelocityYProjectionIntegrator);
    yProj_form->Assemble();
    yProj_form->Finalize();
    if (m_ess_bdr_marker.Size()) {
        yProj_form->EliminateTrialDofs(m_ess_bdr_marker,
                                       *temp1, *temp2);
    }
    m_yProj = yProj_form->LoseMat();
    delete yProj_form;

    // CG solver
    double absTol = 1E-10;
    double relTol = 1E-8;
    int maxIter = 10000;
    int verbose = 0;

    m_cgsolver = new CGSolver;
    m_cgsolver->SetAbsTol(absTol);
    m_cgsolver->SetRelTol(relTol);
    m_cgsolver->SetMaxIter(maxIter);
    m_cgsolver->SetPrintLevel(verbose);
    m_cgsolver->SetOperator(*m_mass);
}

//! Evaluates x-component of velocity
Vector VelocityFunction
:: eval_xComponent(const GridFunction *v)
{
    Vector B(m_sfes->GetTrueVSize());
    Vector V(v->Size());
    v->GetTrueDofs(V);

    m_xProj->Mult(V, B);
    m_xComp = 0.;
    m_cgsolver->Mult(B, m_xComp);

    return m_xComp;
}

//! Evaluates y-component of velocity
Vector VelocityFunction
:: eval_yComponent(const GridFunction *v)
{
    Vector B(m_sfes->GetTrueVSize());
    Vector V(v->Size());
    v->GetTrueDofs(V);

    m_yProj->Mult(V, B);
    m_yComp = 0.;
    m_cgsolver->Mult(B, m_yComp);

    return m_yComp;
}

//! Evaluates x-component and y-component of velocity
std::pair<Vector, Vector> VelocityFunction
:: eval(const GridFunction * v)
{
    return {eval_xComponent(v), eval_yComponent(v)};
}

//! Evaluates x-component and y-component of velocity
void VelocityFunction
:: operator()(const GridFunction& v,
              GridFunction& vx,
              GridFunction& vy) const
{
    Vector Bx(vx.Size());
    Vector By(vy.Size());
    Vector V(m_vfes->GetTrueVSize());
    v.GetTrueDofs(V);

    m_xProj->Mult(V, Bx);
    vx = 0;
    m_cgsolver->Mult(Bx, vx);

    m_yProj->Mult(V, By);
    vy = 0;
    m_cgsolver->Mult(By, vy);
}


//! Vorticity function
//! Initializes components
void VorticityFunction :: init ()
{
    Vector *temp1=nullptr, *temp2=nullptr; // dummy

    // Assemble vorticity mass matrix
    BilinearForm *mass_form
            = new BilinearForm(m_sfes);
    mass_form->AddDomainIntegrator(new MassIntegrator);
    mass_form->Assemble();
    mass_form->Finalize();
    m_mass = mass_form->LoseMat();
    delete mass_form;

    // Assemble Velocity to Vorticity L2-projection matrix
    // velocity to vorticity projection form:
    // \int_{\Omega} curl(u_h) w_h dx
    MixedBilinearForm *proj_form
            = new MixedBilinearForm(m_vfes,
                                    m_sfes);
    proj_form->AddDomainIntegrator
            (new mymfem::VorticityProjectionIntegrator);
    proj_form->Assemble();
    proj_form->Finalize();
    if (m_ess_bdr_marker.Size()) {
        proj_form->EliminateTrialDofs(m_ess_bdr_marker,
                                      *temp1, *temp2);
    }
    m_proj = proj_form->LoseMat();
    delete proj_form;

    // CG solver
    double absTol = 1E-10;
    double relTol = 1E-8;
    int maxIter = 10000;
    int verbose = 0;

    m_cgsolver = new CGSolver;
    m_cgsolver->SetAbsTol(absTol);
    m_cgsolver->SetRelTol(relTol);
    m_cgsolver->SetMaxIter(maxIter);
    m_cgsolver->SetPrintLevel(verbose);
    m_cgsolver->SetOperator(*m_mass);
}

//! Evaluates vorticity
void VorticityFunction
:: operator()(const GridFunction& v,
              GridFunction& w) const
{
    Vector B(w.Size());
    Vector V(m_vfes->GetTrueVSize());
    v.GetTrueDofs(V);

    m_proj->Mult(V, B);
    w = 0;
    m_cgsolver->Mult(B, w);
}


//! Sets the FE space to be considered
//! and the corresponding mass matrix and linear solver
void MeanFreePressure
:: set (FiniteElementSpace *fes) const
{
    SparseMatrix *mass = nullptr;
    m_lfone.SetSize(fes->GetTrueVSize());
    m_massInvLfone.SetSize(fes->GetTrueVSize());

    BilinearForm *mass_form = new BilinearForm(fes);
    mass_form->AddDomainIntegrator(new MassIntegrator);
    mass_form->Assemble();
    mass_form->Finalize();
    mass = mass_form->LoseMat();
    delete mass_form;

    LinearForm lfone_form(fes);
    ConstantCoefficient one(1.0);
    lfone_form.AddDomainIntegrator
            (new DomainLFIntegrator(one));
    lfone_form.Assemble();
    m_lfone = lfone_form.GetData();

    CGSolver cgsolver;
    cgsolver.SetOperator(*mass);
    cgsolver.Mult(m_lfone, m_massInvLfone);

    delete mass;

    m_area = compute_area(*fes);
}

//! Applies the mean-free constraint
//! to the input grid function
void MeanFreePressure
:: operator()(GridFunction &pressure) const
{
    double mean = get(pressure);
    pressure.Add(-mean, m_massInvLfone);
}

//! Returns mean of the given grid function
double MeanFreePressure
:: get(GridFunction& pressure) const {
    return (m_lfone*pressure)/m_area;
}


/// Locates a given physical point x to a mesh element
/// using the initial guess init_elId
/*std::pair <int, IntegrationPoint> PointLocator
:: operator() (const Vector& x, const int init_elId) const
{
    auto [info_, ip_] = compute_ref_point(x, init_elId);
    if (info_ == InverseElementTransformation::Inside) {
        return {init_elId, ip_};
    }

    bool found = false;
    int info;
    int elId = init_elId, old_elId = init_elId;
    IntegrationPoint ip;

    Array <int> vertices;
    Vector z (m_mesh->Dimension());
    double min_dist = std::numeric_limits<double>::max();

    while(!found)
    {
        /// find the element closest to point x amongst the
        /// neighbours of the vertices of the current element
        m_mesh->GetElementVertices(elId, vertices);
        for (int i=0; i < vertices.Size(); i++)
        {
            int v = vertices[i];
            int ne = m_vToEl->RowSize(v);
            const int* els = m_vToEl->GetRow(v);
            //std::cout << "\n\nFor vertex: " << i
            //          << std::endl;
            for (int j=0; j<ne; j++)
            {
                if (els[j] == elId) {continue;}

                m_mesh->GetElementTransformation(els[j])
                        ->Transform(Geometries.GetCenter
                         (m_mesh->GetElementBaseGeometry
                          (els[j])), z);
                double dist = z.DistanceTo(x.GetData());

                //std::cout << els[j] << "\t" << elId << "\t"
                //          << dist << "\t"
                //          << min_dist << std::endl;
                if (dist < min_dist)
                {
                    min_dist = dist;
                    elId = els[j];
                    //std::cout << "Minimum: "
                    //          << min_dist << "\t"
                    //          << elId << std::endl;
                }
            }
        }

        if (elId == old_elId) {
            /// if elId is not upadted, search
            /// all neighbours of its vertices
            for (int i=0; i < vertices.Size(); i++)
            {
                int v = vertices[i];
                int ne = m_vToEl->RowSize(v);
                const int* els = m_vToEl->GetRow(v);

                for (int j=0; j<ne; j++)
                {
                    if (els[j] == elId) {continue;}
                    //std::cout << "Search neighbours: "
                    //          << x(0) << "\t"
                    //          << x(1) << "\t"
                    //          << els[j] << std::endl;
                    std::tie (info, ip)
                        = compute_ref_point(x, els[j]);
                    if (info == InverseElementTransformation
                            ::Inside) {
                        //std::cout << "Exception: "
                        //          << els[j] << std::endl;
                        return {els[j],ip};
                    }
                }
            }
            /// in case the neighbour search fails,
            /// loop over other elements
            //if (elId < m_mesh->GetNE()-1) { elId++;}
            //else { elId = 0; }
        }
        else {
            /// if elId is upadted,
            /// check if it contains point x
            std::tie (info, ip) = compute_ref_point(x, elId);
            if (info == InverseElementTransformation
                    ::Inside) {
                found = true;
            }
            old_elId = elId;
        }
    }

    return {elId, ip};
}*/

//! Maps a set of physical points
//! to elements of the mesh
std::pair <int, IntegrationPoint> PointLocator
:: operator() (const Vector& x, const int init_elId) const
{
    int elId = -1;
    IntegrationPoint ip;
    (*this) (x, init_elId, elId, ip);

    return {elId, ip};
}

//! Maps a physical point
//! to an element of the mesh
int PointLocator
:: operator() (const Vector& x, const int init_elId,
               int& elId, IntegrationPoint& ip) const
{
    bool found = false;

    auto [info_, ip_] = compute_ref_point(x, init_elId);
    if (info_ == InverseElementTransformation::Inside) {
        found = true;
        elId = init_elId;
        ip = ip_;
        return found;
    }

    int info;
    int old_elId = init_elId;
    elId = init_elId;

    Array <int> vertices;
    Vector z (m_mesh->Dimension());
    double min_dist = std::numeric_limits<double>::max();

    while(!found)
    {
        /// find the element closest to point x amongst the
        /// neighbours of the vertices of the current element
        m_mesh->GetElementVertices(elId, vertices);
        for (int i=0; i < vertices.Size(); i++)
        {
            int v = vertices[i];
            int ne = m_vToEl->RowSize(v);
            const int* els = m_vToEl->GetRow(v);
            //std::cout << "\n\nFor vertex: " << i
            //          << std::endl;
            for (int j=0; j<ne; j++)
            {
                if (els[j] == elId) {continue;}

                m_mesh->GetElementTransformation(els[j])
                        ->Transform(Geometries.GetCenter
                         (m_mesh->GetElementBaseGeometry
                          (els[j])), z);
                double dist = z.DistanceTo(x.GetData());

                //std::cout << els[j] << "\t"
                //          << elId << "\t"
                //          << dist << "\t"
                //          << min_dist << std::endl;
                if (dist < min_dist)
                {
                    min_dist = dist;
                    elId = els[j];
                    //std::cout << "Minimum: "
                    //          << min_dist << "\t"
                    //          << elId << std::endl;
                }
            }
        }

        if (elId == old_elId) {
            /// if elId is not upadted,
            /// search all neighbours of its vertices
            for (int i=0; i < vertices.Size(); i++)
            {
                int v = vertices[i];
                int ne = m_vToEl->RowSize(v);
                const int* els = m_vToEl->GetRow(v);

                for (int j=0; j<ne; j++)
                {
                    if (els[j] == elId) {continue;}
                    //std::cout << "Search neighbours: "
                    //          << x(0) << "\t"
                    //          << x(1) << "\t"
                    //          << els[j] << std::endl;
                    std::tie (info, ip)
                            = compute_ref_point(x, els[j]);
                    if (info == InverseElementTransformation
                            ::Inside) {
                        //std::cout << "Exception: "
                        //          << els[j] << std::endl;
                        found = true;
                        elId = els[j];
                        return found;
                    }
                }
            }
            /// in case the neighbour search fails,
            /// loop over other elements
            //if (elId < m_mesh->GetNE()-1) { elId++;}
            //else { elId = 0; }
            elId = -1;
            break;
        }
        else {
            /// if elId is upadted,
            /// check if it contains point x
            std::tie (info, ip)
                    = compute_ref_point(x, elId);
            if (info == InverseElementTransformation
                    ::Inside) {
                found = true;
            }
            old_elId = elId;
        }
    }

    return found;
}


//! Generates a table of vertices,
//! which are shared by different processors.
//! Needed when a ParMesh is written to one file,
//! the same vertices at the partition interfaces
//! will have a different numbering for each processor
void PointLocator :: get_shared_vertices_table() const
{
    int nv = m_mesh->GetNV();
    int dim = m_mesh->Dimension();

    m_shared_vertices->MakeI (nv);
    for (int i=0; i<nv; i++)
    {
        Vector v1(m_mesh->GetVertex(i), dim);
        for (int j=i+1; j<nv; j++)
        {
            if (v1.DistanceTo(m_mesh->GetVertex(j))
                    < m_TOL) {
                m_shared_vertices->AddAColumnInRow(i);
                m_shared_vertices->AddAColumnInRow(j);
            }
        }
    }
    m_shared_vertices->MakeJ();

    for (int i=0; i<nv; i++)
    {
        Vector v1(m_mesh->GetVertex(i), dim);
        for (int j=i+1; j<nv; j++)
        {
            if (v1.DistanceTo(m_mesh->GetVertex(j))
                    < m_TOL) {
                m_shared_vertices->AddConnection(i, j);
                m_shared_vertices->AddConnection(j, i);
            }
        }
    }
    m_shared_vertices->ShiftUpI();
}

//! Generates the vertex to element table,
//! uses the shared vertices table
void PointLocator :: get_vertices_to_elements_table() const
{
    int ne = m_mesh->GetNE();
    int nv = m_mesh->GetNV();

    m_vToEl->MakeI(nv);

    Array<int> v, sv;
    for (int i=0; i<ne; i++)
    {
        m_mesh->GetElementVertices(i, v);
        for (int j=0; j<v.Size(); j++) {
            m_vToEl->AddAColumnInRow(v[j]);

            int nsv = m_shared_vertices->RowSize(v[j]);
            m_shared_vertices->GetRow(v[j], sv);
            for (int k=0; k<nsv; k++) {
                m_vToEl->AddAColumnInRow(sv[k]);
            }
        }
    }
    m_vToEl->MakeJ();

    for (int i=0; i<ne; i++)
    {
        m_mesh->GetElementVertices(i, v);
        for (int j=0; j<v.Size(); j++) {
            m_vToEl->AddConnection(v[j], i);

            int nsv = m_shared_vertices->RowSize(v[j]);
            m_shared_vertices->GetRow(v[j], sv);
            for (int k=0; k<nsv; k++) {
                m_vToEl->AddConnection(sv[k], i);
            }
        }
    }
    m_vToEl->ShiftUpI();
}


// End of file
