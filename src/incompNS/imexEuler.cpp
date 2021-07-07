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
IncompNSImexEulerFEM
:: IncompNSImexEulerFEM (const nlohmann::json& config)
    : IncompNSFEM(config) {}

IncompNSImexEulerFEM
:: IncompNSImexEulerFEM
(const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase)
    : IncompNSFEM(config, testCase) {}

//! Updates system
void IncompNSImexEulerFEM
:: update_system(const double t,
                 const double,
                 GridFunction *)
{
    // boundary conditions
    if (m_block00MatNoBCs) { update_BCs(t); }
}

//! Updates boundary conditions
void IncompNSImexEulerFEM :: update_BCs(double t)
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
void IncompNSImexEulerFEM:: init_system_op()
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
void IncompNSImexEulerFEM:: update_system_op() {}

//! Updates right-hand side
void IncompNSImexEulerFEM
:: update_rhs(double t,
              double dt,
              BlockVector* U,
              BlockVector* B)
{
    auto dimR = B->GetBlock(0).Size();
    Vector &B0 = B->GetBlock(0);
    assemble_source(t, B0);
    m_buf.SetSize(dimR);

    if (m_bool_viscous) {
        assemble_bdry(t, m_buf);
        B0.Add(m_viscosity, m_buf);
    }
    B0 *= dt;

    Vector &V = U->GetBlock(0);
    m_mass->AddMult(V, B0);
    m_buf = m_applyMFConvNFlux(V, V, m_fespaces[0]);
    B0.Add(-dt, m_buf);
    apply_BCs(*B);
}


//--------//
// Solver //
//--------//

//! Constructor
IncompNSImexEulerSolver
:: IncompNSImexEulerSolver
(const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir, const int lx, const int Nt)
    : IncompNSSolver(config, testCase, mesh_dir, lx, Nt)
{
    m_discr = std::make_shared<IncompNSImexEulerFEM>
            (config, testCase);
}

//! Initializes solver
void IncompNSImexEulerSolver :: init ()
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
}

//! Solves one time step of IMEX Euler solver
void IncompNSImexEulerSolver
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

    m_discr->update_system(t, dt, v);

    (*B) = 0.0;
    m_discr->update_rhs(t, dt, U, B);
    m_solver->Mult(*B, *U);
}

// End of file
