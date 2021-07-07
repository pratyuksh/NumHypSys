#include "../../include/incompNS/pdiscretisation.hpp"
#include "../../include/mymfem/mypbilinearform.hpp"
#include "../../include/incompNS/psolver.hpp"
#include "../../include/stokes/assembly.hpp"

#include <fmt/format.h>


//----------------//
// Discretisation //
//----------------//

//! Constructors
IncompNSImexBdf2ParFEM
:: IncompNSImexBdf2ParFEM
(MPI_Comm& comm, const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase)
    : IncompNSParFEM(comm, config, testCase) {}

//! Initializes discretisation for first step
void IncompNSImexBdf2ParFEM
:: init_firstStep(const double dt)
{
    init_firstStep_assemble_system(dt);
    init_system_op();

    m_vnm1 = std::make_shared<ParGridFunction>
                (m_fespaces[0]);
}

//! Initial assembly of system for first step
//! Assembles mass, divergence, diffusion
//! Initializes convection
void IncompNSImexBdf2ParFEM
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
        m_block01Mat = ParAdd(m_divT, m_openBdry);
    }
    
    if (m_bool_viscous)
    {
        HypreParMatrix *tmp
                = new HypreParMatrix(*m_diffusion);
        double coeff = dt*m_gamma*m_viscosity;
        (*tmp) *= coeff;
        m_massDiffusion = ParAdd(m_mass, tmp);
        delete tmp;
    }
}

#if MFEM_VERSION == 40100
//! Solves first step using Imex RK2
void IncompNSImexBdf2ParFEM
:: solve_firstStep(Solver* solver,
                   double dt,
                   ParGridFunction *v,
                   BlockVector* U,
                   BlockVector* B)
{
    auto dimR = B->GetBlock(0).Size();
    Vector &B0 = B->GetBlock(0);
    m_buf.SetSize(dimR);

    // save Vn for next time steps
    update_oldSol(*U);
    update_convection(v);

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
    m_mass->Mult(1, Vn, 1, B0);
    m_convNflux->Mult(-dt*m_gamma, Vn, 1.0, B0);
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
    auto vs = std::make_shared<ParGridFunction>
            (m_fespaces[0]);
    vs->Distribute(&Vs);
    m_mass->Mult(1, Vn, 1, B0);
    m_convNflux->Mult(-dt*m_delta, Vn, 1.0, B0);

    update_convection(vs.get());
    m_convNflux->Mult(-dt*(1-m_delta), Vs, 1.0, B0);
    m_diffusion->Mult(-dt*(1-m_gamma)*m_viscosity, Vs,
                      1, B0);
    apply_BCs(*B);

    // solve stage 1
    solver->Mult(*B, *U);
}

#elif MFEM_VERSION == 40200
//! Solves first step using Imex RK2
void IncompNSImexBdf2ParFEM
:: solve_firstStep(Solver* solver,
                   double dt,
                   ParGridFunction *v,
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
    m_mass->Mult(1, Vn, 1, B0);
    m_buf = m_applyMFConvNFlux(v, v);
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
    auto vs = std::make_shared<ParGridFunction>
            (m_fespaces[0]);
    vs->Distribute(&Vs);
    m_mass->Mult(1, Vn, 1, B0);
    m_buf = m_applyMFConvNFlux(v, v);
    B0.Add(-dt*m_delta, m_buf);
    m_buf = m_applyMFConvNFlux(vs.get(), vs.get());
    B0.Add(-dt*(1-m_delta), m_buf);
    m_diffusion->Mult(-dt*(1-m_gamma)*m_viscosity, Vs,
                      1, B0);
    apply_BCs(*B);

    // solve stage 1
    solver->Mult(*B, *U);
}
#endif

//! Re-initializes system operator and block 00
//! at the end of first step
void IncompNSImexBdf2ParFEM
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
void IncompNSImexBdf2ParFEM :: init(const double dt)
{
    init_assemble_system(dt);
    init_system_op();
}

//! Initial assembly of system
//! Initializes massDiffusion
void IncompNSImexBdf2ParFEM
:: init_assemble_system(const double dt)
{
    // M + dt*m_cs*A
    if (m_bool_viscous)
    {
        if (m_massDiffusion) {delete m_massDiffusion;}
        HypreParMatrix *tmp
                = new HypreParMatrix(*m_diffusion);
        double coeff = dt*m_cs*m_viscosity;
        (*tmp) *= coeff;
        m_massDiffusion = ParAdd(m_mass, tmp);
        delete tmp;
    }
}

//! Updates system
void IncompNSImexBdf2ParFEM
:: update_system(const double t,
                 const double,
                 ParGridFunction *g)
{
#if MFEM_VERSION == 40100
    // update convection operator
    update_convection(g);
#endif
    // boundary conditions
    if (m_block00MatNoBCs) { update_BCs(t); }
}

//! Updates boundary conditions
void IncompNSImexBdf2ParFEM :: update_BCs(double t)
{
    update_vBdry(t);
    Vector vBdry;
    m_vBdry->GetTrueDofs(vBdry);

    *m_rhsDirichlet = 0.0;
    m_block00MatNoBCs->Mult(-1, vBdry,
                            0, m_rhsDirichlet->GetBlock(0));
    m_divNoBCs->Mult(-1, vBdry,
                     0, m_rhsDirichlet->GetBlock(1));
}

//! Initializes system operator
void IncompNSImexBdf2ParFEM:: init_system_op()
{
    if (!m_incompNSOp)
    {
        if (!m_block00Mat) {
            if (!m_bool_viscous) {
                m_block00Mat = new HypreParMatrix(*m_mass);
            } else {
                m_block00Mat = new HypreParMatrix
                        (*m_massDiffusion);
            }
            m_block00MatNoBCs = new HypreParMatrix
                    (*m_block00Mat);
        }
        apply_BCs(*m_block00Mat);

        m_incompNSOp
                = new BlockOperator(m_block_trueOffsets);
        m_incompNSOp->SetBlock(0,0, m_block00Mat);
        m_incompNSOp->SetBlock(0,1, m_block01Mat);
        m_incompNSOp->SetBlock(1,0, m_div);
    }
}

//! Updates system operator
void IncompNSImexBdf2ParFEM:: update_system_op() {}


#if MFEM_VERSION == 40100
//! Updates right-hand side
void IncompNSImexBdf2ParFEM
:: update_rhs(double tn, double dt, ParGridFunction* vn,
              BlockVector* Un, BlockVector* B)
{
    auto dimR = B->GetBlock(0).Size();
    Vector &B0 = B->GetBlock(0);
    m_buf.SetSize(dimR);

    if (m_bool_viscous) {
        assemble_bdry(tn, m_buf);    //t_{n}
        B0.Add(m_viscosity*m_cs*m_b0, m_buf);
        assemble_bdry(tn-dt, m_buf); //t_{n-1}
        B0.Add(m_viscosity*m_cs*m_b1, m_buf);
    }
    B0 *= dt;

    // a0*M*V_{n} + a1*M*V_{n-1}
    const Vector &Vn = Un->GetBlock(0);
    m_mass->Mult(m_a0, Vn, 1, B0);
    m_mass->Mult(m_a1, Vn, 1, B0);

    // - dt*b0*C*V_{n} - dt*b1*C*V_{n-1}
    m_convNflux->Mult(-dt*m_cs*m_b0, Vn, 1.0, B0);
    update_convection(m_vnm1.get());
    m_convNflux->Mult(-dt*m_cs*m_b1, m_Vnm1, 1.0, B0);
    apply_BCs(*B);
}

#elif MFEM_VERSION == 40200
//! Updates right-hand side
void IncompNSImexBdf2ParFEM
:: update_rhs(double tnp1, double dt,
              ParGridFunction* vn,
              BlockVector* Un,
              BlockVector* B)
{
    auto dimR = B->GetBlock(0).Size();
    Vector &B0 = B->GetBlock(0);
    m_buf.SetSize(dimR);

    assemble_source(tnp1, m_buf);
    B0.Add(m_cs, m_buf);
    if (m_bool_viscous) {
        assemble_bdry(tnp1, m_buf);
        B0.Add(m_viscosity*m_cs, m_buf);
    }
    B0 *= dt;

    // a0*M*V_{n} + a1*M*V_{n-1}
    const Vector &Vn = Un->GetBlock(0);
    m_mass->Mult(m_a0, Vn, 1, B0);
    m_mass->Mult(m_a1, Vn, 1, B0);

    // - dt*b0*C*V_{n} - dt*b1*C*V_{n-1}
    m_buf = m_applyMFConvNFlux(vn, vn);
    B0.Add(-dt*m_b0, m_buf);
    m_buf = m_applyMFConvNFlux(m_vnm1.get(), m_vnm1.get());
    B0.Add(-dt*m_b1, m_buf);

    apply_BCs(*B);
}

#endif

//--------//
// Solver //
//--------//

//! Constructor
IncompNSImexBdf2ParSolver
:: IncompNSImexBdf2ParSolver
(MPI_Comm comm, const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir, const int lx, const int Nt)
    : IncompNSParSolver(comm, config, testCase,
                        mesh_dir, lx, Nt)
{
    m_discr = std::make_shared<IncompNSImexBdf2ParFEM>
            (comm, config, testCase);
}

//! Initializes solver
void IncompNSImexBdf2ParSolver :: init ()
{
    const double dt = m_tEnd/m_Nt;

    std::string linear_solver_type
            = m_config["linear_solver_type"];

    // assemble system
    m_discr->set(m_pmesh);
    m_discr->init_firstStep(dt);

    // init linear solver and preconditioner
    init_linear_solver();
    init_preconditioner();

    // update linear solver and preconditioner
    m_incompNSOp = m_discr->get_incompNS_op();
    update_linear_solver();
    update_preconditioner(dt);

    m_block_offsets = m_discr->get_block_offsets();
    m_block_trueOffsets = m_discr->get_block_trueOffsets();

    // init observer
    m_observer->init(m_discr);
}

//! Solves one time step of IMEX Bdf2 solver
void IncompNSImexBdf2ParSolver
:: solve_one_step (const int step_num,
                   std::shared_ptr<ParGridFunction>& v,
                   BlockVector *U,
                   BlockVector *B) const
{
    const double dt = m_tEnd/m_Nt;
    const double t = step_num*dt;
    if (m_myrank == IamRoot) {
        std::cout << "\n"
                  << step_num << "\t"
                  << t << "\t"
                  << dt << std::endl;
    }

    if (step_num == 1) // first step is Imex RK2
    {
        // update old solution buffer
        m_discr->update_oldSol(*U);

        // solve first step
        (*B) = 0.0;
        m_discr->solve_firstStep(m_solver, dt,
                                 v.get(), U, B);
        if (!static_cast<IterativeSolver *>
                (m_solver)->GetConverged())
        {
            if (m_myrank == IamRoot) {
                std::cout << "Final norm: "
                          << static_cast<IterativeSolver *>
                             (m_solver)->GetFinalNorm()
                          << std::endl;
            }
            abort();
        }

        // re-initialize
        m_discr->partial_reinit_after_firstStep(dt);
        m_incompNSOp = m_discr->get_incompNS_op();
        update_linear_solver();
        update_preconditioner(dt);
    }
    else
    {
        BlockVector Un(*U);
        m_discr->update_system(t, dt, v.get());

        (*B) = 0.0;
        m_discr->update_rhs(t, dt, v.get(), U, B);
        m_solver->Mult(*B, *U);
        if (!static_cast<IterativeSolver *>
                (m_solver)->GetConverged())
        {
            if (m_myrank == IamRoot) {
                std::cout << "Final norm: "
                          << static_cast<IterativeSolver *>
                             (m_solver)->GetFinalNorm()
                          << std::endl;
            }
            abort();
        }
        // update old solution buffer
        m_discr->update_oldSol(Un);
    }
}

// End of file
