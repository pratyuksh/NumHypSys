#include "../../include/incompNS/pdiscretisation.hpp"
#include "../../include/mymfem/mypbilinearform.hpp"
#include "../../include/incompNS/psolver.hpp"
#include "../../include/incompNS/passembly.hpp"
#include "../../include/stokes/assembly.hpp"

#include <fmt/format.h>
#include <chrono>

using namespace std::chrono;


//----------------//
// Discretisation //
//----------------//

//! Constructors
IncompNSBdf1Rk2ParFEM
:: IncompNSBdf1Rk2ParFEM
(MPI_Comm& comm, const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase)
    : IncompNSParFEM(comm, config, testCase) {}

//! Updates system
void IncompNSBdf1Rk2ParFEM
:: update_system(const double t, const double)
{
    // boundary conditions
    if (m_block00MatNoBCs) { update_BCs(t); }
}

//! Updates boundary conditions
void IncompNSBdf1Rk2ParFEM :: update_BCs(double t)
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
void IncompNSBdf1Rk2ParFEM:: init_system_op()
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
void IncompNSBdf1Rk2ParFEM:: update_system_op() {}

//! Updates right-hand side
void IncompNSBdf1Rk2ParFEM
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

    m_mass->Mult(1., V, dt, B0);
    apply_BCs(*B);
}


//--------//
// Solver //
//--------//

//! Constructor
IncompNSBdf1Rk2ParSolver
:: IncompNSBdf1Rk2ParSolver
(MPI_Comm comm, const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir, const int lx, const int Nt)
    : IncompNSParSolver(comm, config, testCase,
                        mesh_dir, lx, Nt)
{
    m_discr = std::make_shared<IncompNSBdf1Rk2ParFEM>
            (comm, config, testCase);
    m_dt = m_tEnd/m_Nt;
}

//! Initializes solver
void IncompNSBdf1Rk2ParSolver :: init ()
{
    std::string linear_solver_type
            = m_config["linear_solver_type"];

    // assemble system
    m_discr->set(m_pmesh);
    m_discr->init(m_dt);

    // init linear solver and preconditioner
    init_linear_solver();
    init_preconditioner();

    // update linear solver and preconditioner
    m_incompNSOp = m_discr->get_incompNS_op();
    update_linear_solver();
    update_preconditioner(m_dt);

    m_block_offsets = m_discr->get_block_offsets();
    m_block_trueOffsets = m_discr->get_block_trueOffsets();

    // init observer
    m_observer->init(m_discr);

    // sets mass inverse for propagation
    m_discr->assemble_invMass();
    m_invMass = m_discr->get_invMass_matrix();

    // sets auxiliary variables
    m_vs = std::make_shared<ParGridFunction>
            (m_discr->get_fespaces()[0]);
    m_Vs = std::make_unique<Vector>
            (m_discr->get_fespaces()[0]->GetTrueVSize());
    m_buf = std::make_unique<Vector>(m_Vs->Size());
    m_km = std::make_unique<Vector>(m_Vs->Size());
}

//! Solves one time step of BDF1-RK2 solver
void IncompNSBdf1Rk2ParSolver
:: solve_one_step (const int step_num,
                   std::shared_ptr<ParGridFunction>& v,
                   BlockVector *U,
                   BlockVector *B) const
{
    const double tn = (step_num-1)*m_dt;
    const double tnp1 = step_num*m_dt;
    if (m_myrank == IamRoot) {
        std::cout << "\n"
                  << step_num << "\t"
                  << tnp1 << "\t"
                  << m_dt << std::endl;
    }

    // auxiliary variable
    m_vn = v;

    // explicit propagation
    *m_Vs = U->GetBlock(0);
    propagate(tn, *m_Vs);

    // implicit diffusion
    m_discr->update_system(tnp1, m_dt);

    (*B) = 0.0;
    m_discr->update_rhs(tnp1, m_dt, *m_Vs, B);
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

    //Vs.Destroy();
}

//! Runs propagation
void IncompNSBdf1Rk2ParSolver
:: propagate(double tn, Vector& Vs) const
{
    double s = 0;
    while (s < m_dt)
    {
        // cfl time step
        m_vs->Distribute(&Vs);
        double ds_cfl = compute_time_step(m_vs);
        double ds = {ds_cfl > m_dt - s ? m_dt-s : ds_cfl};
        if (m_myrank == IamRoot) {
            std::cout << s << "\t"
                      << ds_cfl << "\t"
                      << ds << std::endl;
        }

        // one step
        propagate_one_step(s, ds, Vs);
        m_discr->update_vBdry(tn+s+ds);
        m_discr->apply_vBdry(Vs);
        s += ds;
    }
}

//! Runs one time step of propagation
//! Explicit RK2, mid-point method
//! Applies matrix-free convection operator
#if MFEM_VERSION == 40100
void IncompNSBdf1Rk2ParSolver
:: propagate_one_step(double s, double ds, Vector& V) const
{
    propagate_one_step_slow(s, ds, V);
}

#elif MFEM_VERSION == 40200

void IncompNSBdf1Rk2ParSolver
:: propagate_one_step(double, double ds, Vector& V) const
{
    mymfem::ApplyParMFConvNFluxOperator applyMFConvNFlux;

    // stage 0
    m_vs->Distribute(&(V));
    *m_buf = applyMFConvNFlux(m_vs.get(), m_vn.get());
    m_invMass->Mult(*m_buf, *m_km);

    // stage 1, mid-point rule
    (*m_km) *= -0.5*ds;
    m_km->Add(1, V);
    m_vs->Distribute(&(*m_km));
    *m_buf = applyMFConvNFlux(m_vs.get(), m_vn.get());
    m_invMass->Mult(*m_buf, *m_km);

    // update solution
    V.Add(-ds, *m_km);
}
#endif

//! Runs one time step of propagation
//! Explicit RK2, mid-point method
//! Slow version assembles the convection matrix
void IncompNSBdf1Rk2ParSolver
:: propagate_one_step_slow
(double, double ds, Vector& V) const
{
    m_discr->update_convection(m_vn.get());
    auto conv = m_discr->get_convec_matrix();

    // stage 0
    conv->Mult(V, *m_buf);
    m_invMass->Mult(*m_buf, *m_km);

    // stage 1, mid-point rule
    (*m_km) *= -0.5*ds;
    m_km->Add(1, V);
    conv->Mult(*m_km, *m_buf);
    m_invMass->Mult(*m_buf, *m_km);

    // update solution
    V.Add(-ds, *m_km);
}

//! Constant exptrapolation
void IncompNSBdf1Rk2ParSolver
:: extrapolate(double, Vector& Vs) const
{
    m_vn->GetTrueDofs(Vs);
}



// End of file
