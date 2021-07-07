#include "../../include/incompNS/discretisation.hpp"
#include "../../include/incompNS/solver.hpp"
#include "../../include/stokes/assembly.hpp"

#include <fmt/format.h>
#include <chrono>
using namespace std::chrono;


//----------------//
// Discretisation //
//----------------//

//! Constructors
IncompNSImexBdf2FEM
:: IncompNSImexBdf2FEM (const nlohmann::json& config)
    : IncompNSFEM(config) {}

IncompNSImexBdf2FEM
:: IncompNSImexBdf2FEM
(const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase)
    : IncompNSFEM(config, testCase) {}

//! Initializes discretisation for first step
void IncompNSImexBdf2FEM :: init_firstStep(const double dt)
{
    init_firstStep_assemble_system(dt);
    init_system_op();
}

//! Initial assembly of system for first step
//! Assembles mass, divergence, diffusion
//! Initializes convection
void IncompNSImexBdf2FEM
:: init_firstStep_assemble_system(const double dt)
{
    assemble_mass();
    assemble_divergence();
    init_convection();

    // diffusion SIP form
    if (m_bool_viscous) { assemble_diffusion(); }

    // matrix block 01
    if (m_bool_viscous) { m_block01Mat = m_divT; }
    else {
        assemble_openBdry();
        m_block01Mat = Add(*m_divT, *m_openBdry);
    }

    // M + dt*m_gamma*A
    if (m_bool_viscous)
    {
        double coeff = dt*m_gamma*m_viscosity;
        m_massDiffusion = Add(1, *m_mass,
                              coeff, *m_diffusion);
    }
}

//! Solves first step using Imex RK2
void IncompNSImexBdf2FEM
:: solve_firstStep(Solver* solver,
                   double dt,
                   BlockVector* U,
                   BlockVector* B)
{
    auto dimR = B->GetBlock(0).Size();
    Vector &B0 = B->GetBlock(0);
    m_buf.SetSize(dimR);

    // save Vn for next time steps
    update_oldSol(*U);

    // update BCs
    if (m_block00MatNoBCs) { update_BCs(dt); }

    /// RK2: stage 0
    BlockVector Us(*U);

    // compute rhs:
    // M*V_{n} - dt*gamma*( C*V_{n} - f_{n} )
    //assemble_source(0, B0);
    if (m_bool_viscous) {
        assemble_bdry(0, m_buf);
        B0.Add(m_viscosity, m_buf);
    }
    B0 *= dt*m_gamma;

    Vector &Vn = U->GetBlock(0);
    m_mass->AddMult(Vn, B0);
    m_buf = m_applyMFConvNFlux(Vn, Vn, m_fespaces[0]);
    B0.Add(-dt*m_gamma, m_buf);
    apply_BCs(*B);

    // solve stage 0
    solver->Mult(*B, Us);

    /// RK2: stage 1
    *B = 0;
    // compute rhs:
    // M*V_{n} - dt*delta*( C*V_{n} - f_{n} )
    // - dt*(1-delta)*( C*V_{s} - f_{n} )
    // - dt*(1-gamma)*A*V_{s}
    if (m_bool_viscous) {
        assemble_bdry(0, m_buf);
        B0.Add(m_viscosity*m_delta, m_buf);
        assemble_bdry(m_gamma*dt, m_buf);
        B0.Add(m_viscosity*(1-m_delta), m_buf);
    }
    B0 *= dt;

    Vector &Vs = Us.GetBlock(0);
    m_mass->AddMult(Vn, B0);
    m_buf = m_applyMFConvNFlux(Vn, Vn, m_fespaces[0]);
    B0.Add(-dt*m_delta, m_buf);
    m_buf = m_applyMFConvNFlux(Vs, Vs, m_fespaces[0]);
    B0.Add(-dt*(1-m_delta), m_buf);
    /*assemble_source(0, m_buf);
    B0.Add(dt*m_delta, m_buf);
    assemble_source(m_gamma*dt, m_buf);
    B0.Add(dt*(1-m_delta), m_buf);*/
    m_diffusion->AddMult(Vs, B0,
                         -dt*(1-m_gamma)*m_viscosity);
    apply_BCs(*B);

    // solve stage 1
    solver->Mult(*B, *U);
}

//! Re-initializes system operator and block 00
//! at the end of first step
void IncompNSImexBdf2FEM
:: partial_reinit_after_firstStep(double dt)
{
    if (m_incompNSOp) {
        delete m_incompNSOp;
        m_incompNSOp = nullptr;
    }
    if (m_block00Mat) {
        delete m_block00Mat;
        m_block00Mat = nullptr;
    }
    if (m_block00MatNoBCs) {
        delete m_block00MatNoBCs;
        m_block00MatNoBCs = nullptr;
    }
    init(dt);
}

//! Initializes discretisation
void IncompNSImexBdf2FEM :: init(const double dt)
{
    init_assemble_system(dt);
    init_system_op();
}

//! Initial assembly of system
//! Initializes massDiffusion
void IncompNSImexBdf2FEM
:: init_assemble_system(const double dt)
{
    // M + dt*m_cs*A
    if (m_bool_viscous)
    {
        if (m_massDiffusion) {delete m_massDiffusion;}
        double coeff = dt*m_cs*m_viscosity;
        m_massDiffusion = Add(1, *m_mass,
                              coeff, *m_diffusion);
    }
}

//! Updates system
void IncompNSImexBdf2FEM
:: update_system(const double t,
                 const double,
                 GridFunction *)
{
    if (m_block00MatNoBCs) { update_BCs(t); }
}

//! Updates boundary conditions
void IncompNSImexBdf2FEM :: update_BCs(double t)
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
void IncompNSImexBdf2FEM:: init_system_op()
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
void IncompNSImexBdf2FEM:: update_system_op() {}

//! Updates right-hand side
void IncompNSImexBdf2FEM
:: update_rhs(double tnp1,
              double dt,
              BlockVector* Un,
              BlockVector* B)
{
    auto dimR = B->GetBlock(0).Size();
    Vector &B0 = B->GetBlock(0);
    m_buf.SetSize(dimR);

    // dt*cs*f_{n+1}
    assemble_source(tnp1, m_buf);
    B0.Add(m_cs, m_buf);
    if (m_bool_viscous) {
        assemble_bdry(tnp1, m_buf);
        B0.Add(m_viscosity*m_cs, m_buf);
    }
    B0 *= dt;

    // a0*M*V_{n} + a1*M*V_{n-1}
    Vector &Vn = Un->GetBlock(0);
    m_mass->AddMult(Vn, B0, m_a0);
    m_mass->AddMult(m_Vnm1, B0, m_a1);

    // - dt*b0*C*V_{n} - dt*b1*C*V_{n-1}
    m_buf = m_applyMFConvNFlux(Vn, Vn,
                               m_fespaces[0]);
    B0.Add(-dt*m_b0, m_buf);
    m_buf = m_applyMFConvNFlux(m_Vnm1, m_Vnm1,
                               m_fespaces[0]);
    B0.Add(-dt*m_b1, m_buf);

    apply_BCs(*B);
}


//--------//
// Solver //
//--------//

//! Constructor
IncompNSImexBdf2Solver
:: IncompNSImexBdf2Solver
(const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir, const int lx, const int Nt)
    : IncompNSSolver(config, testCase, mesh_dir, lx, Nt)
{
    m_discr = std::make_shared<IncompNSImexBdf2FEM>
            (config, testCase);
}

//! Initializes solver
void IncompNSImexBdf2Solver :: init ()
{
    const double dt = m_tEnd/m_Nt;

    // assemble system
    m_discr->set(m_mesh);
    m_discr->init_firstStep(dt);

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
}

//! Solves one time step of IMEX Bdf2 solver
void IncompNSImexBdf2Solver
:: solve_one_step (const int step_num,
                   GridFunction *v,
                   BlockVector *U,
                   BlockVector *B) const
{
    const double dt = m_tEnd/m_Nt;
    const double t = step_num*dt;
    std::cout << "\n"
              << step_num << "\t"
              << t << "\t"
              << dt << std::endl;

    if (step_num == 1) // first step is Imex RK2
    {
        // update old solution buffer
        m_discr->update_oldSol(*U);

        // solve first step
        (*B) = 0.0;
        m_discr->solve_firstStep(m_solver, dt, U, B);

        // re-initialize
        m_discr->partial_reinit_after_firstStep(dt);
        m_incompNSOp = m_discr->get_incompNS_op();
        update_linear_solver();
        update_preconditioner(dt);
    }
    else
    {
        BlockVector Un(*U);
        m_discr->update_system(t, dt, v);

        (*B) = 0.0;
        m_discr->update_rhs(t, dt, U, B);
        m_solver->Mult(*B, *U);

        // update old solution buffer
        m_discr->update_oldSol(Un);
    }
}

// End of file
