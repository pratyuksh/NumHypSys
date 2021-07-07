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
IncompNSBackwardEulerFEM
:: IncompNSBackwardEulerFEM (const nlohmann::json& config)
    : IncompNSFEM(config) {}

IncompNSBackwardEulerFEM
:: IncompNSBackwardEulerFEM
(const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 bool bool_embedded_preconditioner)
    : IncompNSFEM(config, testCase,
                  bool_embedded_preconditioner) {}

//! Initializes discretisation
void IncompNSBackwardEulerFEM
:: init(const double dt)
{
    init_assemble_system(dt);
    if (m_bool_embedded_preconditioner) {
        init_preconditioned_system_op();
    }
    else {
        init_system_op();
    }
}

//! Updates system
void IncompNSBackwardEulerFEM
:: update_system(const double t,
                 const double dt,
                 GridFunction *v)
{
    // update convection operator
    update_convection(v);

    // reset block00 matrix
    reset_block00Mat();

    // set block00 matrix
    if (!m_bool_viscous) {
        m_block00Mat = Add(1, *m_mass,
                           dt, *m_convNflux);
    } else {
        m_block00Mat = Add(1, *m_massDiffusion,
                           dt, *m_convNflux);
    }

    // boundary conditions
    m_appliedBCs = false;
    update_BCs(t);
    apply_BCs(*m_block00Mat);
    m_appliedBCs = true;
}

//! Updates the boundary conditions
void IncompNSBackwardEulerFEM :: update_BCs(double t)
{
    if (m_appliedBCs) {
        throw std::runtime_error(fmt::format(
            "Rows and Cols for Dirichlet Dofs have "
            "already been "
            "eliminated from m_block00Mat"));
    }

    update_vBdry(t);
    *m_rhsDirichlet = 0.0;
    m_block00Mat->AddMult(*m_vBdry,
                          m_rhsDirichlet->GetBlock(0), -1);
    m_divNoBCs->AddMult(*m_vBdry,
                        m_rhsDirichlet->GetBlock(1), -1);
}

//! Initializes system operator
void IncompNSBackwardEulerFEM:: init_system_op()
{
    if (!m_incompNSOp) {
        m_incompNSOp = new BlockOperator(m_block_offsets);
        m_incompNSOp->SetBlock(1,0, m_div);
        m_incompNSOp->SetBlock(0,1, m_block01Mat);
    }
}

//! Updates system operator
void IncompNSBackwardEulerFEM:: update_system_op()
{
    if (m_block00Mat) {
        m_incompNSOp->SetBlock(0,0, m_block00Mat);
    }
}

//! Initializes system operator with
//! embedded preconditioner
void IncompNSBackwardEulerFEM
:: init_preconditioned_system_op()
{
    if (!m_incompNSPr) {
        m_incompNSPr = new BlockDiagonalPreconditioner
                (m_block_offsets);
    }
    if (!m_invPrP1) {
        m_invPrP1 = new mymfem::InvPrP1Op
                (m_block_offsets);
    }
    if (!m_invMatS)
    {
        double absTol = 1E-8;
        double relTol = 1E-8;
        int maxIter = 1000;
        int verbose = 0;

        auto cg = new CGSolver ();
        cg->SetAbsTol(absTol);
        cg->SetRelTol(relTol);
        cg->SetMaxIter(maxIter);
        cg->SetPrintLevel(verbose);

        m_invMatS = cg;
        m_invMatS->iterative_mode = false;
    }
}

//! Updates system operator with
//! embedded preconditioner
void IncompNSBackwardEulerFEM
:: update_preconditioned_system_op()
{
    // release old memory
    if (m_incompNSPrOp) {
        delete m_incompNSPrOp;
        m_incompNSPrOp = nullptr;
    }
    if (m_prP2) {
        delete m_prP2;
        m_prP2 = nullptr;
    }
    if (m_invMatD) {
        delete m_invMatD;
        m_invMatD = nullptr;
    }
    if (m_matS) {
        delete m_matS;
        m_matS = nullptr;
    }

    // matS = B2 * diag(block00Mat) * B1^T
    SparseMatrix *B2MultInvDMultB1T
            = new SparseMatrix(*m_block01Mat);
    m_block00Diag.SetSize(m_block00Mat->Height());

    m_block00Mat->GetDiag(m_block00Diag);
    for (int i = 0; i < m_block00Diag.Size(); i++) {
        B2MultInvDMultB1T->
                ScaleRow(i, 1./m_block00Diag(i));
    }
    m_matS = Mult(*m_div, *B2MultInvDMultB1T);
    delete B2MultInvDMultB1T;

    // invMatS
    m_invMatS->SetOperator(*m_matS);

    // inverse diag(block00)
    m_invMatD = new DSmoother(*m_block00Mat);
    static_cast<Solver *>
            (m_invMatD)->iterative_mode = false;

    // update inverse P1 preconditioner
    m_invPrP1->update(m_block01Mat, m_div,
                      m_invMatS, m_invMatD);

    // set P2 preconditioner
    m_prP2 = new mymfem::PrP2Op (m_block00Mat, m_invMatD);

    // set preconditioned system operator
    m_incompNSPrOp = new mymfem::IncompNSPrOp
            (m_invPrP1, m_prP2);

    // set diagonal preconditioner
    auto blockDiagPr
            = static_cast<BlockDiagonalPreconditioner*>
            (m_incompNSPr);
    blockDiagPr->SetDiagonalBlock(0, m_invMatD);
    blockDiagPr->SetDiagonalBlock(1, m_invMatS);
}

//! Solves system operator with
//! embedded preconditioner
void IncompNSBackwardEulerFEM
:: preconditioned_solve (Solver *solver,
                         const BlockVector& B,
                         BlockVector& U) const
{
    // preconditioned rhs
    BlockVector buf(B);
    BlockVector buf2(B);
    BlockVector prB(B);
    m_incompNSPr->Mult(B, buf);
    m_invPrP1->Mult(buf, prB);

    // solve velocity
    solver->SetOperator(*m_incompNSPrOp);
    solver->Mult(prB.GetBlock(0), U.GetBlock(0));

    // evaluate pressure
    Vector tmp(m_prP2->NumRows());
    m_prP2->Mult(U.GetBlock(0), tmp);
    m_invPrP1->get_block10()->Mult(tmp, U.GetBlock(1));
    U.GetBlock(1) *= -1;
    U.GetBlock(1).Add(1, prB.GetBlock(1));
}

//! Updates right-hand side
void IncompNSBackwardEulerFEM
:: update_rhs(double t,
              double dt,
              BlockVector* U,
              BlockVector* B)
{
    auto dimR = B->GetBlock(0).Size();
    Vector &B0 = B->GetBlock(0);
    assemble_source(t, B0);
    if (m_bool_viscous) {
        Vector buf(dimR);
        assemble_bdry(t, buf);
        B0.Add(m_viscosity, buf);
    }
    B0 *= dt;

    const Vector &V = U->GetBlock(0);
    m_mass->AddMult(V, B0);
    apply_BCs(*B);
}


//--------//
// Solver //
//--------//

//! Constructor
IncompNSBackwardEulerSolver
:: IncompNSBackwardEulerSolver
(const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir, const int lx, const int Nt)
    : IncompNSSolver(config, testCase, mesh_dir, lx, Nt)
{
    m_discr = std::make_shared<IncompNSBackwardEulerFEM>
            (config, testCase,
             m_bool_embedded_preconditioner);
    m_dt = m_tEnd/m_Nt;
}

//! Initializes solver
void IncompNSBackwardEulerSolver :: init ()
{
    // init assemble system
    m_discr->set(m_mesh);
    m_discr->init(m_dt);
    m_block_offsets = m_discr->get_block_offsets();
    m_ess_bdr_marker = m_discr->get_ess_bdr_marker();

    // init linear solver and preconditioner
    init_linear_solver();
    init_preconditioner();

    // init observer
    m_observer->init(m_discr);
}

//! Solves one time step of Backward Euler solver
void IncompNSBackwardEulerSolver
:: solve_one_step (const int step_num,
                   GridFunction *v,
                   BlockVector *U,
                   BlockVector *B) const
{
    const double t = step_num*m_dt;
    if (step_num%10 == 0) {
        std::cout << "\n"
                  << step_num << "\t"
                  << t << "\t"
                  << m_dt << std::endl;
    }
    update(m_dt, t, v, U, B);
    solve(U, B);
}

//! Updates solver components: system, rhs and solver
void IncompNSBackwardEulerSolver
:: update (const double dt, const double t,
           GridFunction *v,
           BlockVector *U, BlockVector *B) const
{

    m_discr->update_system(t, dt, v);

    (*B) = 0.0;
    m_discr->update_rhs(t, dt, U, B);

    if (m_bool_embedded_preconditioner) {
        m_discr->update_preconditioned_system_op();
    }
    else {
        m_discr->update_system_op();
        m_incompNSOp = m_discr->get_incompNS_op();
        update_linear_solver();
        update_preconditioner(dt);
    }
}

//! Runs the appropriate linear system solvers
void IncompNSBackwardEulerSolver
:: solve (BlockVector *U, BlockVector *B) const
{
    if (m_bool_embedded_preconditioner) {
        m_discr->preconditioned_solve(m_solver, *B, *U);
    }
    else {
        m_solver->Mult(*B, *U);
    }

    if (!static_cast<IterativeSolver *>
            (m_solver)->GetConverged())
    {
        std::cout << "Final norm: "
                  << static_cast<IterativeSolver *>
                     (m_solver)->GetFinalNorm()
                  << std::endl;
        abort();
    }
}

// End of file
