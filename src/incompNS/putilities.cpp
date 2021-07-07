#include "../../include/incompNS/putilities.hpp"
#include "../../include/stokes/assembly.hpp"


// Divergence measurement and cleaning routines

//! Measures divergence
double measure_divergence(HypreParMatrix* divMat,
                          ParGridFunction* v)
{
    Vector b(divMat->GetNumRows());
    Vector v_true;
    v->GetTrueDofs(v_true);
    divMat->Mult(v_true, b);
    double div_local = b.Norml2()*b.Norml2();
    double div = 0;
    MPI_Allreduce(&div_local, &div, 1,
                  MPI_DOUBLE, MPI_SUM,
                  divMat->GetComm());

    return std::sqrt(div);
}


//! Weak Divergence Integrator
//! Assembles the linear form for given element
void ParWeakDivergenceLfIntegrator ::
AssembleRHSElementVect (const FiniteElement &el,
                        ElementTransformation &Trans,
                        Vector &elvect)
{
    int ndofs = el.GetDof();

#ifdef MFEM_THREAD_SAFE
    Vector shape(ndofs);
#elif
    shape.SetSize(ndofs);
#endif
    elvect.SetSize(ndofs);

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
       int order = 2*el.GetOrder();
       ir = &IntRules.Get(el.GetGeomType(), order);
    }

    elvect = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);
        el.CalcShape(ip, shape);

        double div = m_gf->GetDivergence(Trans);
        double coeff = div*ip.weight*Trans.Weight();
        elvect.Add(coeff, shape);
    }
}


//! Velocity divergence cleaner
//! Constructor
ParDivergenceFreeVelocity
:: ParDivergenceFreeVelocity
(const nlohmann::json& config,
 const std::shared_ptr<ParMesh>& mesh)
{
    m_deg = config["deg_x"];

    // H^1 space
    FiniteElementCollection *h1_coll
            = new H1_FECollection(m_deg+2,
                                  mesh->Dimension());
    m_h1fes = new ParFiniteElementSpace(mesh.get(),
                                        h1_coll);

    // Raviart-Thomas space
    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(m_deg, mesh->Dimension());
    m_rtfes = new ParFiniteElementSpace(mesh.get(),
                                        hdiv_coll);

    // linear solvers
    double tol = 1E-12;
    int maxIter = 10000;
    int verbose = 0;

    m_cgsolver_stiff = new HyprePCG(mesh->GetComm());
    m_cgsolver_stiff->SetTol(tol);
    m_cgsolver_stiff->SetMaxIter(maxIter);
    m_cgsolver_stiff->SetPrintLevel(verbose);
    m_cgsolver_stiff->iterative_mode = true;

    m_cgsolver_mass = new HyprePCG(mesh->GetComm());
    m_cgsolver_mass->SetTol(tol);
    m_cgsolver_mass->SetMaxIter(maxIter);
    m_cgsolver_mass->SetPrintLevel(verbose);
    m_cgsolver_mass->iterative_mode = true;

    m_cgpr_stiff = new HypreBoomerAMG();

    // boundary
    if (mesh->bdr_attributes.Size()) {
        m_ess_bdr_marker_h1fes
                .SetSize(mesh->bdr_attributes.Max());
        m_ess_bdr_marker_h1fes = 1;
        m_h1fes->GetEssentialTrueDofs
                (m_ess_bdr_marker_h1fes,
                 m_ess_tdof_list_h1fes);
        m_ess_bdr_marker_rtfes
                .SetSize(mesh->bdr_attributes.Max());
        m_ess_bdr_marker_rtfes = 1;
        m_rtfes->GetEssentialTrueDofs
                (m_ess_bdr_marker_rtfes,
                 m_ess_tdof_list_rtfes);
    }

    // set operators
    set();
}

//! Destructor
ParDivergenceFreeVelocity
:: ~ParDivergenceFreeVelocity ()
{
    if (m_h1fes) { delete m_h1fes; }
    if (m_stiff_h1) { delete m_stiff_h1; }
    if (m_cgsolver_stiff) { delete m_cgsolver_stiff; }
    if (m_cgpr_stiff) { delete m_cgpr_stiff; }

    if (m_rtfes) { delete m_rtfes; }
    if (m_mass_rt) { delete m_mass_rt; }
    if (m_cgsolver_mass) { delete m_cgsolver_mass; }

    if (m_grad) { delete m_grad; }
    if (m_rhs) { delete m_rhs; }
}

//! Sets components
void ParDivergenceFreeVelocity
:: set () const
{
    // operators for H^1 space
    ParBilinearForm *stiff_form
            = new ParBilinearForm(m_h1fes);
    ConstantCoefficient one(1.0);
    stiff_form->AddDomainIntegrator
            (new DiffusionIntegrator(one));
    stiff_form->Assemble();
    stiff_form->Finalize();
    stiff_form->EliminateEssentialBC(m_ess_bdr_marker_h1fes);
    m_stiff_h1 = stiff_form->ParallelAssemble();
    delete stiff_form;

    static_cast<HypreBoomerAMG *>
            (m_cgpr_stiff)->SetOperator(*m_stiff_h1);
    static_cast<HypreBoomerAMG *>
            (m_cgpr_stiff)->SetPrintLevel(0);
    m_cgsolver_stiff->SetOperator(*m_stiff_h1);
    m_cgsolver_stiff->SetPreconditioner(*m_cgpr_stiff);

    // operators for RT space
    ParBilinearForm *mass_form
            = new ParBilinearForm(m_rtfes);
    mass_form->AddDomainIntegrator
            (new VectorFEMassIntegrator(one));
    mass_form->Assemble();
    mass_form->Finalize();
    //mass_form->
    //        EliminateEssentialBC(m_ess_bdr_marker_rtfes);
    m_mass_rt = mass_form->ParallelAssemble();
    delete mass_form;

    m_cgsolver_mass->SetOperator(*m_mass_rt);

    // mixed operator, trial space H^1, test space RT
    Vector *dum1=nullptr, *dum2=nullptr; // dummy
    ParMixedBilinearForm* grad_form
            = new ParMixedBilinearForm(m_h1fes, m_rtfes);
    grad_form->AddDomainIntegrator
            (new mymfem::GradientIntegrator);
    grad_form->Assemble();
    grad_form->Finalize();
    grad_form->EliminateTrialDofs(m_ess_bdr_marker_h1fes,
                                  *dum1, *dum2);
    //grad_form->EliminateTestDofs(m_ess_tdof_list_rtfes);
    m_grad = grad_form->ParallelAssemble();
    delete grad_form;

    // init rhs
    m_rhs = new Vector(m_h1fes->GetTrueVSize());
}

//! Cleans divergence
void ParDivergenceFreeVelocity
:: operator() (ParGridFunction *v) const
{
    // weak divergence form \int \div(v)*w dx
    ParLinearForm weak_div(m_h1fes);
    weak_div.AddDomainIntegrator
            (new ParWeakDivergenceLfIntegrator(v));
    weak_div.Assemble();
    weak_div.ParallelAssemble(*m_rhs);
    m_rhs->SetSubVector(m_ess_tdof_list_h1fes, 0.0);

    // eval phi
    Vector phi (m_h1fes->GetTrueVSize());
    phi = 0.;
    m_cgsolver_stiff->Mult(*m_rhs, phi);

    // project phi (in H^1 space) tp H(div) space
    Vector buf, delta;
    v->GetTrueDofs(buf); delta = buf;
    m_grad->Mult(phi, buf);
    //buf.SetSubVector(m_ess_tdof_list_rtfes, 0.0);
    m_cgsolver_mass->Mult(buf, delta);

    // update v <- v + delta
    v->AddDistribute(1.0, delta);
}

// End of file
