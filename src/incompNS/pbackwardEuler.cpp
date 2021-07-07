#include "../../include/incompNS/pdiscretisation.hpp"
#include "../../include/mymfem/mypbilinearform.hpp"
#include "../../include/incompNS/psolver.hpp"
#include "../../include/stokes/assembly.hpp"

#include <fmt/format.h>


//----------------//
// Discretisation //
//----------------//

//! Constructor
IncompNSBackwardEulerParFEM
:: IncompNSBackwardEulerParFEM
(MPI_Comm& comm, const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 bool bool_embedded_preconditioner)
    : IncompNSParFEM(comm, config, testCase,
                     bool_embedded_preconditioner) {}

//! Initializes discretisation
void IncompNSBackwardEulerParFEM
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
void IncompNSBackwardEulerParFEM
:: update_system(const double t,
                 const double dt,
                 ParGridFunction *g)
{
    // update convection operator
    update_convection(g);

    // reset block00 matrix
    reset_block00Mat();

    // set block00 matrix
    (*m_convNflux) *= dt;
    if (!m_bool_viscous) {
        m_block00Mat = ParAdd(m_mass, m_convNflux);
    }
    else {
        m_block00Mat = ParAdd(m_massDiffusion, m_convNflux);
    }

    // boundary conditions
    m_appliedBCs = false;
    update_BCs(t);
    apply_BCs(*m_block00Mat);
    m_appliedBCs = true;
}

//! Updates boundary conditions
void IncompNSBackwardEulerParFEM :: update_BCs(double t)
{
    if (m_appliedBCs) {
        throw std::runtime_error(fmt::format(
            "Rows and Cols for Dirichlet Dofs have "
            "already been "
            "eliminated from m_block00Mat"));
    }

    update_vBdry(t);
    Vector vBdry;
    m_vBdry->GetTrueDofs(vBdry);

    *m_rhsDirichlet = 0.0;
    m_block00Mat->Mult(-1, vBdry,
                       0, m_rhsDirichlet->GetBlock(0));
    m_divNoBCs->Mult(-1, vBdry,
                     0, m_rhsDirichlet->GetBlock(1));
}

//! Initializes system operator
void IncompNSBackwardEulerParFEM:: init_system_op()
{
    if (!m_incompNSOp) {
        m_incompNSOp
                = new BlockOperator(m_block_trueOffsets);
        m_incompNSOp->SetBlock(0,1, m_block01Mat);
        m_incompNSOp->SetBlock(1,0, m_div);
    }
}

//! Updates system operator
void IncompNSBackwardEulerParFEM:: update_system_op()
{
    if (m_block00Mat) {
        m_incompNSOp->SetBlock(0,0, m_block00Mat);
    }
}

//! Initializes system operator with
//! embedded preconditioner
void IncompNSBackwardEulerParFEM
:: init_preconditioned_system_op()
{
    if (!m_incompNSPr)
    {
        m_incompNSPr = new BlockDiagonalPreconditioner
                (m_block_trueOffsets);
    }

    if (!m_invPrP1) {
        m_invPrP1 = new mymfem::InvPrP1Op
                (m_block_trueOffsets);
    }

    if (!m_invMatS)
    {
        // CG solver
        double tol = 1E-8;
        int maxIter = 1000;
        int verbose = 0;

        auto cg = new HyprePCG(m_mass->GetComm());
        cg->SetTol(tol);
        cg->SetMaxIter(maxIter);
        cg->SetPrintLevel(verbose);

        m_invMatS = cg;
        m_invMatS->iterative_mode = false;
    }
}

//! Updates system operator with
//! embedded preconditioner
void IncompNSBackwardEulerParFEM
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
    if (m_block00InvMultDivuT)
    {
        delete m_block00InvMultDivuT;
        m_block00InvMultDivuT = nullptr;
    }
    if (m_block00Diag) {
        delete m_block00Diag;
        m_block00Diag = nullptr;
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
    m_block00InvMultDivuT
            = new HypreParMatrix(*m_block01Mat);
    m_block00Diag
            = new HypreParVector
            (m_block00Mat->GetComm(),
             m_block00Mat->GetGlobalNumRows(),
             m_block00Mat->GetRowStarts());
    m_block00Mat->GetDiag(*m_block00Diag);
    m_block00InvMultDivuT->InvScaleRows(*m_block00Diag);
    m_matS = ParMult(m_div, m_block00InvMultDivuT);

    // invMatS
    m_invMatS->SetOperator(*m_matS);
    /*auto amg = new HypreBoomerAMG();
    amg->SetPrintLevel(0);
    amg->SetOperator(*m_matS);
    static_cast<HyprePCG *>(m_invMatS)
            ->SetPreconditioner(*amg);*/

    // inverse diag(block00)
    m_invMatD = new HypreDiagScale(*m_block00Mat);
    static_cast<HypreDiagScale *>
            (m_invMatD)->iterative_mode = false;

    // set inverse P1 preconditioner
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
void IncompNSBackwardEulerParFEM
:: preconditioned_solve (Solver *solver,
                         const BlockVector& B,
                         BlockVector& U) const
{
    solver->SetOperator(*m_incompNSPrOp);

    // preconditioned rhs
    BlockVector buf(B);
    BlockVector prB(B);
    m_incompNSPr->Mult(B, buf);
    m_invPrP1->Mult(buf, prB);

    // solve velocity
    solver->Mult(prB.GetBlock(0), U.GetBlock(0));

    // evaluate pressure
    Vector tmp(U.BlockSize(0));
    m_prP2->Mult(U.GetBlock(0), tmp);
    m_invPrP1->get_block10()->Mult(tmp, U.GetBlock(1));
    U.GetBlock(1) *= -1;
    U.GetBlock(1).Add(1, prB.GetBlock(1));
}

//! Updates right-hand side
void IncompNSBackwardEulerParFEM
:: update_rhs(double t, double dt,
              BlockVector* U, BlockVector* B)
{
    auto dimR = B->GetBlock(0).Size();
    Vector &B0 = B->GetBlock(0);
    assemble_source(t, B0);
    if (m_bool_viscous) {
        Vector buf(dimR);
        assemble_bdry(t, buf);
        B0.Add(m_viscosity, buf);
    }

    Vector &V = U->GetBlock(0);
    m_mass->Mult(1., V, dt, B0);
    apply_BCs(*B);
}


//--------//
// Solver //
//--------//

//! Constructor
IncompNSBackwardEulerParSolver
:: IncompNSBackwardEulerParSolver
(MPI_Comm comm, const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir, const int lx, const int Nt)
    : IncompNSParSolver(comm, config, testCase,
                        mesh_dir, lx, Nt)
{
    m_discr = std::make_shared<IncompNSBackwardEulerParFEM>
            (comm, config, testCase,
             m_bool_embedded_preconditioner);
}

//! Initializes solver
void IncompNSBackwardEulerParSolver :: init ()
{
    const double dt = m_tEnd/m_Nt;

    // assemble system
    m_discr->set(m_pmesh);
    m_discr->init(dt);
    m_block_offsets = m_discr->get_block_offsets();
    m_block_trueOffsets = m_discr->get_block_trueOffsets();

    // init linear solver and preconditioner
    init_linear_solver();
    init_preconditioner();

    // init observer
    m_observer->init(m_discr);

    // dump mesh
    //m_observer->dump_mesh(m_pmesh);
}

//! Solves one time step of Backward Euler solver
void IncompNSBackwardEulerParSolver
:: solve_one_step (const int step_num,
                   std::shared_ptr<ParGridFunction>& v,
                   BlockVector *U,
                   BlockVector *B) const
{
    const double dt = m_tEnd/m_Nt;
    const double t = step_num*dt;
    if (m_myrank == IamRoot) {
        if (step_num%10 == 0) {
            std::cout << "\n"
                      << step_num << "\t"
                      << t << "\t"
                      << dt << std::endl;
        }
    }
    update(dt, t, v, U, B);
    solve(U, B);
}

//! Updates solver components: system, rhs and solver
void IncompNSBackwardEulerParSolver
:: update (const double dt, const double t,
           std::shared_ptr<ParGridFunction>& v,
           BlockVector *U, BlockVector *B) const
{
    m_discr->update_system(t, dt, v.get());

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
void IncompNSBackwardEulerParSolver
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
        if (m_myrank == IamRoot) {
            std::cout << "Final norm: "
                      << static_cast<IterativeSolver *>
                         (m_solver)->GetFinalNorm()
                      << std::endl;
        }
        abort();
    }
}

// End of file
