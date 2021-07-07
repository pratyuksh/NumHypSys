#include "../../include/incompNS/discretisation.hpp"
#include "../../include/incompNS/solver.hpp"
#include "../../include/stokes/assembly.hpp"

#include <fmt/format.h>


//----------------//
// Discretisation //
//----------------//

//! Constructors
IncompNSBdf1Rk2FEM
:: IncompNSBdf1Rk2FEM (const nlohmann::json& config)
    : IncompNSFEM(config) {}

IncompNSBdf1Rk2FEM
:: IncompNSBdf1Rk2FEM
(const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase)
    : IncompNSFEM(config, testCase) {}

//! Updates system
void IncompNSBdf1Rk2FEM
:: update_system(const double t,
                 const double,
                 GridFunction*)
{
    if (m_block00MatNoBCs) { update_BCs(t); }
}

//! Updates boundary conditions
void IncompNSBdf1Rk2FEM :: update_BCs(double t)
{
    update_vBdry(t);
    *m_rhsDirichlet = 0.0;
    m_block00MatNoBCs->AddMult(*m_vBdry,
                               m_rhsDirichlet->GetBlock(0),
                               -1);
    m_divNoBCs->AddMult(*m_vBdry,
                        m_rhsDirichlet->GetBlock(1), -1);
}

//! Initializes system operator
void IncompNSBdf1Rk2FEM:: init_system_op()
{
    if (!m_incompNSOp)
    {
        if (!m_block00Mat) {
            if (!m_bool_viscous) {
                m_block00Mat = new SparseMatrix(*m_mass);
            } else {
                m_block00Mat
                        = new SparseMatrix(*m_massDiffusion);
            }
            m_block00MatNoBCs
                    = new SparseMatrix(*m_block00Mat);
        }
        apply_BCs(*m_block00Mat);

        m_incompNSOp = new BlockOperator(m_block_offsets);
        m_incompNSOp->SetBlock(0,0, m_block00Mat);
        m_incompNSOp->SetBlock(0,1, m_block01Mat);
        m_incompNSOp->SetBlock(1,0, m_div);
    }
}

//! Updates system operator
void IncompNSBdf1Rk2FEM:: update_system_op() {}

//! Updates right-hand side
void IncompNSBdf1Rk2FEM
:: update_rhs(double t,
              double dt,
              Vector& V,
              BlockVector* B)
{
    auto dimR = B->GetBlock(0).Size();
    Vector &B0 = B->GetBlock(0);
    //assemble_source(t, B0);
    if (m_bool_viscous) {
        Vector buf(dimR);
        assemble_bdry(t, buf);
        B0.Add(m_viscosity, buf);
    }
    B0 *= dt;

    m_mass->AddMult(V, B0);
    apply_BCs(*B);
}


//--------//
// Solver //
//--------//

//! Constructor
IncompNSBdf1Rk2Solver
:: IncompNSBdf1Rk2Solver
(const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir, const int lx, const int Nt)
    : IncompNSSolver(config, testCase, mesh_dir, lx, Nt)
{
    m_discr = std::make_shared<IncompNSBdf1Rk2FEM>
            (config, testCase);
    m_dt = m_tEnd/m_Nt;
}

//! Initializes solver
void IncompNSBdf1Rk2Solver :: init ()
{
    const double dt = m_tEnd/m_Nt;

    // assemble system
    m_discr->set(m_mesh);
    m_discr->init(dt);

    // init linear solver and preconditioner
    init_linear_solver();
    init_preconditioner();

    // update linear solver and preconditioner
    m_incompNSOp = m_discr->get_incompNS_op();
    update_linear_solver();
    update_preconditioner(dt);

    m_block_offsets = m_discr->get_block_offsets();
    m_ess_bdr_marker = m_discr->get_ess_bdr_marker();

    // init observer
    m_observer->init(m_discr);

    // sets mass inverse for propagation
    m_discr->assemble_invMass();
    m_invMass = m_discr->get_invMass_matrix();
    //m_discr->apply_BCs(*m_invMass);

    // set auxiliary variables
    m_bufVn.SetSize(m_discr->
               get_fespaces()[0]->GetTrueVSize());
}

//! Solves one time step of BDF1-RK2 solver
void IncompNSBdf1Rk2Solver
:: solve_one_step (const int step_num,
                   GridFunction *v,
                   BlockVector *U,
                   BlockVector *B) const
{
    const double tn = (step_num-1)*m_dt;
    const double tnp1 = step_num*m_dt;
    std::cout << "\n"
              << step_num << "\t"
              << tnp1 << "\t"
              << m_dt << std::endl;

    m_bufVn = U->GetBlock(0);

    // explicit propagation
    Vector Vs(U->GetBlock(0));
    propagate(tn, Vs);

    // implicit diffusion
    m_discr->update_system(tnp1, m_dt, v);

    (*B) = 0.0;
    m_discr->update_rhs(tnp1, m_dt, Vs, B);
    //B->Print();
    m_solver->Mult(*B, *U);

    Vs.Destroy();

    /*SparseMatrix *divNoBCs
            = m_discr->get_divNoBCs_matrix();
    double div_old = measure_divergence(divNoBCs,
                                        U->GetBlock(0));
    std::cout << "Weak divergence check: "
              << div_old << std::endl;*/
}

//! Runs propagation
void IncompNSBdf1Rk2Solver
:: propagate(double tn, Vector& Vs) const
{
    auto vs = std::make_shared<GridFunction>();
    vs->MakeRef(m_discr->get_fespaces()[0], Vs);

    /*SparseMatrix *divNoBCs
            = m_discr->get_divNoBCs_matrix();
    double div_old = measure_divergence(divNoBCs,
                                        vs.get());
    std::cout << "Weak divergence old: "
              << div_old << std::endl;*/

    double s = 0;
    while (s < m_dt)
    {
        // cfl time step
        double ds_cfl = compute_time_step(vs);
        double ds = {ds_cfl > m_dt - s ? m_dt-s : ds_cfl};
        std::cout << s << "\t"
                  << ds_cfl << "\t"
                  << ds << std::endl;

        // one step
        propagate_one_step(s, ds, Vs);
        m_discr->update_vBdry(tn+s+ds);
        m_discr->apply_vBdry(Vs);
        s += ds;
    }
    /*double div_new = measure_divergence(divNoBCs,
                                        vs.get());
    std::cout << "Weak divergence new: "
              << div_new << std::endl;*/
}

//! Runs one time step of propagation
//! Explicit RK2, mid-point method
//! Applies matrix-free convection operator
void IncompNSBdf1Rk2Solver
:: propagate_one_step(double, double ds, Vector& V) const
{
    Vector buf(V.Size());
    std::unique_ptr<Vector> km
            = std::make_unique<Vector>(V.Size());

    auto fes = m_discr->get_fespaces()[0];
    mymfem::ApplyMFConvNFluxOperator applyMFConvNFlux;

    // stage 0
    buf = applyMFConvNFlux(V, m_bufVn, fes);
    m_invMass->Mult(buf, *km);

    // stage 1, mid-point rule
    (*km) *= -0.5*ds;
    km->Add(1, V);
    buf = applyMFConvNFlux(*km, m_bufVn, fes);
    m_invMass->Mult(buf, *km);

    // update solution
    V.Add(-ds, *km);
}

//! Runs one time step of propagation
//! Explicit RK4
//! Applies matrix-free convection operator
/*void IncompNSBdf1Rk2Solver
:: propagate_one_step(double, double ds, Vector& V) const
{
    Vector buf(V.Size());
    std::unique_ptr<Vector> k0
            = std::make_unique<Vector>(V.Size());
    std::unique_ptr<Vector> k1
            = std::make_unique<Vector>(V.Size());
    std::unique_ptr<Vector> k2
            = std::make_unique<Vector>(V.Size());
    std::unique_ptr<Vector> k3
            = std::make_unique<Vector>(V.Size());

    auto fes = m_discr->get_fespaces()[0];
    mymfem::ApplyMFConvNFluxOperator applyMFConvNFlux;

    // stage 0
    buf = applyMFConvNFlux(V, m_bufVn, fes);
    m_invMass->Mult(buf, *k0);

    // stage 1, mid-point rule
    k1->Set(-0.5*ds, *k0);
    k1->Add(1, V);
    buf = applyMFConvNFlux(*k1, m_bufVn, fes);
    m_invMass->Mult(buf, *k1);

    // stage 2
    k2->Set(-0.5*ds, *k1);
    k2->Add(1, V);
    buf = applyMFConvNFlux(*k2, m_bufVn, fes);
    m_invMass->Mult(buf, *k2);

    // stage 3
    k3->Set(-ds, *k2);
    k3->Add(1, V);
    buf = applyMFConvNFlux(*k3, m_bufVn, fes);
    m_invMass->Mult(buf, *k3);

    // update solution
    V.Add(-ds/6, *k0);
    V.Add(-ds/3, *k1);
    V.Add(-ds/3, *k2);
    V.Add(-ds/6, *k3);
}*/

//! Runs one time step of propagation
//! Explicit RK2, mid-point method
//! Slow version assembles the convection matrix
void IncompNSBdf1Rk2Solver
:: propagate_one_step_slow
(double, double ds, Vector& V) const
{
    Vector buf(V.Size());
    std::unique_ptr<Vector> km
            = std::make_unique<Vector>(V.Size());

    m_discr->update_convection(m_bufVn);
    auto conv = m_discr->get_convec_matrix();

    // stage 0
    conv->Mult(V, buf);
    m_invMass->Mult(buf, *km);

    // stage 1, mid-point rule
    (*km) *= -0.5*ds;
    km->Add(1, V);
    conv->Mult(*km, buf);
    m_invMass->Mult(buf, *km);

    // update solution
    V.Add(-ds, *km);

    buf.Destroy();
}

//! Constant exptrapolation
void IncompNSBdf1Rk2Solver
:: extrapolate(double, Vector& Vs) const
{
    Vs = m_bufVn;
}

// End of file
