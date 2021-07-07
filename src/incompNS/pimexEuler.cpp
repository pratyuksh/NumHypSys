#include "../../include/incompNS/pdiscretisation.hpp"
#include "../../include/mymfem/mypbilinearform.hpp"
#include "../../include/incompNS/psolver.hpp"
#include "../../include/stokes/assembly.hpp"

#include <fmt/format.h>


//----------------//
// Discretisation //
//----------------//

//! Constructors
IncompNSImexEulerParFEM
:: IncompNSImexEulerParFEM
(MPI_Comm& comm, const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase)
    : IncompNSParFEM(comm, config, testCase) {}

//! Updates system
void IncompNSImexEulerParFEM
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
void IncompNSImexEulerParFEM :: update_BCs(double t)
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
void IncompNSImexEulerParFEM:: init_system_op()
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
void IncompNSImexEulerParFEM:: update_system_op() {}

//! Initializes system matrix
void IncompNSImexEulerParFEM:: init_system_mat()
{
    if (!m_incompNSMat)
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

        m_incompNSBlocks.SetSize(2,2);
        m_incompNSBlocks(0,0) = m_block00Mat;
        m_incompNSBlocks(0,1) = m_block01Mat;
        m_incompNSBlocks(1,0) = m_div;
        m_incompNSBlocks(1,1) = nullptr;
        m_incompNSMat = mymfem::HypreParMatrixFromBlocks
                (m_incompNSBlocks);
    }
}

//! Updates system matrix
void IncompNSImexEulerParFEM:: update_system_mat() {}

//! Updates right-hand side
void IncompNSImexEulerParFEM
:: update_rhs(double t, double dt, ParGridFunction* v,
              BlockVector* U, BlockVector* B)
{
    auto dimR = B->GetBlock(0).Size();
    Vector &B0 = B->GetBlock(0);
    m_buf.SetSize(dimR);

    assemble_source(t, B0);
    if (m_bool_viscous) {
        assemble_bdry(t, m_buf);
        B0.Add(m_viscosity, m_buf);
    }

    const Vector &V = U->GetBlock(0);
    m_mass->Mult(1., V, dt, B0);
#if MFEM_VERSION == 40100
    m_convNflux->Mult(-dt, V, 1.0, B0);
#elif MFEM_VERSION == 40200
    m_buf = m_applyMFConvNFlux(v, v);
    B0.Add(-dt, m_buf);
#endif
    apply_BCs(*B);
}


//--------//
// Solver //
//--------//

//! Constructor
IncompNSImexEulerParSolver
:: IncompNSImexEulerParSolver
(MPI_Comm comm, const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir, const int lx, const int Nt)
    : IncompNSParSolver(comm, config, testCase,
                        mesh_dir, lx, Nt)
{
    m_discr = std::make_shared<IncompNSImexEulerParFEM>
            (comm, config, testCase);
}

//! Initializes solver
void IncompNSImexEulerParSolver :: init ()
{
    const double dt = m_tEnd/m_Nt;

    std::string linear_solver_type
            = m_config["linear_solver_type"];

    // assemble system
    m_discr->set(m_pmesh);
    if (linear_solver_type != "superlu") {
        m_discr->init(dt);
        m_incompNSOp = m_discr->get_incompNS_op();
    }
    else {
        m_discr->init_assemble_system(dt);
        m_discr->init_system_mat();
        m_incompNSMat = m_discr->get_incompNS_mat();
    }

    // init linear solver and preconditioner
    init_linear_solver();
    init_preconditioner();

    // update linear solver and preconditioner
    update_linear_solver();
    update_preconditioner(dt);

    m_block_offsets = m_discr->get_block_offsets();
    m_block_trueOffsets = m_discr->get_block_trueOffsets();

    // init observer
    m_observer->init(m_discr);

    // dump mesh
    //m_observer->dump_mesh(m_pmesh);
}

//! Solves one time step of IMEX Euler solver
void IncompNSImexEulerParSolver
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
}

// End of file
