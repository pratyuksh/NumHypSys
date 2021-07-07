#include "../../include/incompNS/solver.hpp"
#include "../../include/incompNS/utilities.hpp"

#include <iostream>
#include <chrono>

using namespace std::chrono;


//! Constructors
IncompNSSolver
:: IncompNSSolver (const nlohmann::json config)
    : m_config(config)
{
    m_deg = config["deg_x"];
    m_tEnd = config["end_time"];

    m_bool_read_mesh_from_file = false;
    if (config.contains("read_mesh_from_file")) {
        m_bool_read_mesh_from_file
                = config["read_mesh_from_file"];

        m_num_refinements = 0;
        if (config.contains("num_refinements")) {
            m_num_refinements = config["num_refinements"];
        }
    }

    m_bool_error = false;
    if (config.contains("eval_error")) {
        m_bool_error = config["eval_error"];
    }

    m_mesh_format = "mesh";
    if (config.contains("mesh_format")) {
        m_mesh_format = m_config["mesh_format"];
    }

    m_mesh_elem_type = "quad";
    if (config.contains("mesh_elem_type")) {
        m_mesh_elem_type = m_config["mesh_elem_type"];
    }

    m_bool_cleanDivg = false;
    if (config.contains("clean_divergence")) {
        m_bool_cleanDivg = config["clean_divergence"];
    }

    m_bool_embedded_preconditioner = false;
    if (config.contains("embedded_preconditioner")) {
        m_bool_embedded_preconditioner
                = config["embedded_preconditioner"];
    }

    m_bool_mean_free_pressure = true;
    if (config.contains("bool_mean_free_pressure")) {
        m_bool_mean_free_pressure
                = config["bool_mean_free_pressure"];
    }

    if (config.contains("cfl_number")) {
        m_cfl_number = config["cfl_number"];
    }
}

IncompNSSolver
:: IncompNSSolver
(const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase)
    : IncompNSSolver(config)
{
    m_testCase = testCase;
}

IncompNSSolver
:: IncompNSSolver
(const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir,
 const int lx, const int Nt)
    : IncompNSSolver(config, testCase)
{
    m_Nt = Nt;
    set(mesh_dir, lx);
}

//! Destructor
IncompNSSolver
:: ~IncompNSSolver ()
{
    clear();
}

//! Sets mesh and observer
void IncompNSSolver :: set(const std::string mesh_dir,
                           const int lx)
{
    std::cout << "\nRun Incompressible "
                 "Navier-Stokes solver ..."
              << std::endl;

    // test case
    if (!m_testCase) {
        m_testCase = make_incompNS_test_case(m_config);
    }

    // mesh
    m_mesh_dir = mesh_dir;
    if (m_bool_read_mesh_from_file) {
        const std::string mesh_file
                = mesh_dir+m_mesh_elem_type+"_mesh_l"
                +std::to_string(lx-m_num_refinements)
                +"."+m_mesh_format;

        std::cout << "  Mesh file: "
                  << mesh_file << std::endl;
        std::cout << "  Number of uniform "
                     "mesh refinements: "
                  << m_num_refinements << std::endl;

        m_mesh = std::make_shared<Mesh>(mesh_file.c_str());
        for (int k=0; k<m_num_refinements; k++) {
            m_mesh->UniformRefinement();
        }
    }
    else {
        const std::string mesh_file
                = mesh_dir+m_mesh_elem_type+"_mesh_l0."
                +m_mesh_format;

        std::cout << "  Base mesh file: "
                  << mesh_file << std::endl;
        std::cout << "  Number of uniform mesh refinements: "
                  << lx << std::endl;

        m_mesh = std::make_shared<Mesh>(mesh_file.c_str());
        for (int k=0; k<lx; k++) {
            m_mesh->UniformRefinement();
        }
    }
    std::tie(m_hMin, m_hMax) = eval_hMinMax();
    std::cout << "  Minimum cell size: "
              << m_hMin << std::endl;
    std::cout << "  Maximum cell size: "
              << m_hMax << std::endl;

    // observer
    m_observer = std::make_shared<IncompNSObserver>
            (m_config, m_testCase, lx);
}

//! Initializes and runs solver
void IncompNSSolver
:: operator()(std::unique_ptr<BlockVector>& U)
{
    init ();
    run (U);
}

//! Runs initialized solver
void IncompNSSolver
:: run (std::unique_ptr<BlockVector>& U) const
{
    U = std::make_unique<BlockVector>(m_block_offsets);

    const double dt = m_tEnd/m_Nt;
    auto fespaces = m_discr->get_fespaces();

    // MFEM rhs variable
    auto B = std::make_unique<BlockVector>(m_block_offsets);

    // auxiliary variables
    auto v = std::make_shared<GridFunction>();
    auto p = std::make_shared<GridFunction>();
    v->MakeRef(fespaces[0], U->GetBlock(0));
    p->MakeRef(fespaces[1], U->GetBlock(1));

    // initial conditions
    IncompNSInitialVelocityCoeff v0_coeff(m_testCase);
    IncompNSInitialPressureCoeff p0_coeff(m_testCase);
    v0_coeff.SetTime(0);
    p0_coeff.SetTime(0);
    v->ProjectCoefficient(v0_coeff);
    p->ProjectCoefficient(p0_coeff);

    clean_divergence(v);
    //visualize(v, p);

    // scale pressure dofs for convenience
    (*p) *= dt;

    // integrate from time 0 -> T
    for (int n=1; n<=m_Nt; n++) {
        // one step: t_{n} -> t_{n+1}
        solve_one_step(n, v.get(), U.get(), B.get());
    }
    (*p) *= (1./dt);
    //visualize(v, p);
}

//! Initializes and runs solver
//! Computes the solution error if needed
std::pair<double, Eigen::VectorXd> IncompNSSolver
:: operator() (void)
{
    Eigen::VectorXd errL2(2);
    errL2.setZero();

    const double dt = m_tEnd/m_Nt;

    init ();
    auto fespaces = m_discr->get_fespaces();

    // MFEM solution and rhs variables
    auto U = std::make_unique<BlockVector>
            (m_block_offsets);
    auto B = std::make_unique<BlockVector>
            (m_block_offsets);

    // auxiliary variables
    auto v = std::make_shared<GridFunction>();
    auto p = std::make_shared<GridFunction>();
    v->MakeRef(fespaces[0], U->GetBlock(0));
    p->MakeRef(fespaces[1], U->GetBlock(1));

    // initial conditions
    IncompNSInitialVelocityCoeff v0_coeff(m_testCase);
    IncompNSInitialPressureCoeff p0_coeff(m_testCase);
    v0_coeff.SetTime(0);
    p0_coeff.SetTime(0);
    v->ProjectCoefficient(v0_coeff);
    p->ProjectCoefficient(p0_coeff);

    clean_divergence(v);
    visualize(v, p);

    // routines for making pressure mean free
    MeanFreePressure *meanFreePressure = nullptr;
    if (m_bool_embedded_preconditioner &&
            m_bool_mean_free_pressure)
    {
        meanFreePressure
                = new MeanFreePressure (fespaces[1]);
    }

    // integrate from time 0 -> T
    for (int n=1; n<=m_Nt; n++)
    {
        // one step: t_{n} -> t_{n+1}
        (*p) *= dt;
        solve_one_step(n, v.get(), U.get(), B.get());

        // make pressure mean free
        if (m_bool_embedded_preconditioner &&
                m_bool_mean_free_pressure)
        {
            (*meanFreePressure)(*p);
            p->GetTrueDofs(U->GetBlock(1));
            double p_mean = meanFreePressure->get(*p);
            std::cout << "Mean pressure: "
                      << p_mean << std::endl;
        }

        (*p) /= dt;
        compute_error(dt*n, v, p);
    }
    visualize(v, p);
    errL2 = compute_error(m_tEnd, v, p);
    if (meanFreePressure) { delete meanFreePressure; }

    return {m_hMax, errL2};
}

//! Deletes memory
void IncompNSSolver
:: clear ()
{
    if (m_solver) { delete m_solver; }
    if (m_precond) { delete m_precond; }
}

//! Initializes linear system solver
void IncompNSSolver :: init_linear_solver() const
{
    std::string linear_solver_type
            = m_config["linear_solver_type"];

    if (linear_solver_type == "minres")
    {
        if (!m_solver)
        {
            double absTol = m_config["minres_absTol"];
            double relTol = m_config["minres_relTol"];
            int maxIter = m_config["minres_maxIter"];
            int verbose = m_config["minres_verbose"];

            MINRESSolver *minres = new MINRESSolver ();
            minres->SetAbsTol(absTol);
            minres->SetRelTol(relTol);
            minres->SetMaxIter(maxIter);
            minres->SetPrintLevel(verbose);
            m_solver = minres;
        }
    }
    else if (linear_solver_type == "bicgstab")
    {
        if (!m_solver)
        {
            double absTol = m_config["bicgstab_absTol"];
            double relTol = m_config["bicgstab_relTol"];
            int maxIter = m_config["bicgstab_maxIter"];
            int verbose = m_config["bicgstab_verbose"];

            BiCGSTABSolver *bicgstab
                    = new BiCGSTABSolver ();
            bicgstab->SetOperator(*m_incompNSOp);
            bicgstab->SetAbsTol(absTol);
            bicgstab->SetRelTol(relTol);
            bicgstab->SetMaxIter(maxIter);
            bicgstab->SetPrintLevel(verbose-1);
            m_solver = bicgstab;
        }
    }
    else if (linear_solver_type == "gmres")
    {
        if (!m_solver)
        {
            double absTol = m_config["gmres_absTol"];
            double relTol = m_config["gmres_relTol"];
            int kDim = m_config["gmres_kDim"];
            int maxIter = m_config["gmres_maxIter"];
            int verbose = m_config["gmres_verbose"];

            GMRESSolver *gmres = new GMRESSolver ();
            gmres->SetAbsTol(absTol);
            gmres->SetRelTol(relTol);
            gmres->SetMaxIter(maxIter);
            gmres->SetPrintLevel(verbose);
            gmres->SetKDim(kDim);
            m_solver = gmres;
        }
    }
    else { MFEM_ABORT("\nUnknown linear system solver!") }

    m_solver->iterative_mode = true;
    if (m_config.contains("iterative_mode")) {
        m_solver->iterative_mode
                = m_config["iterative_mode"];
    }
}

//! Updates linear system solver
void IncompNSSolver :: update_linear_solver() const
{
    if (m_incompNSOp) {
        m_solver->SetOperator(*m_incompNSOp);
    }
}

//! Initializes preconditioner
void IncompNSSolver
:: init_preconditioner () const
{
    std::string precond_type = m_config["precond_type"];
    m_discr->init_preconditioner(precond_type);
}

//! Updates preconditioner
void IncompNSSolver
:: update_preconditioner (const double dt) const
{
    std::string linear_solver_type
            = m_config["linear_solver_type"];
    std::string precond_type = m_config["precond_type"];

    m_discr->update_preconditioner(precond_type, dt);
    m_incompNSPr = m_discr->get_incompNS_pr();

    if (linear_solver_type == "minres" && m_incompNSPr)
    {
        auto minres
                = static_cast<MINRESSolver *>(m_solver);
        minres->SetPreconditioner(*m_incompNSPr);
    }
    else if (linear_solver_type == "bicgstab"
             && m_incompNSPr)
    {
        auto bicgstab
                = static_cast<BiCGSTABSolver*>(m_solver);
        bicgstab->SetPreconditioner(*m_incompNSPr);
    }
    else if (linear_solver_type == "gmres" && m_incompNSPr)
    {
        auto gmres = static_cast<GMRESSolver *>(m_solver);
        gmres->SetPreconditioner(*m_incompNSPr);
    }
}

//! Computes error for velocity and pressure
Eigen::VectorXd IncompNSSolver
:: compute_error (double t,
                  std::shared_ptr<GridFunction> v,
                  std::shared_ptr<GridFunction> p) const
{
    Eigen::VectorXd errL2(2);
    errL2.setZero();

    if (m_bool_error)
    {
        double errvL2, vEL2, errpL2, pEL2;
        std::tie (errvL2, vEL2, errpL2, pEL2)
                = m_observer->eval_error_L2(t, v, p);
        {
            std::cout << "velocity L2 error: "
                      << errvL2 << "\t"
                      << vEL2 << std::endl;
            std::cout << "pressure L2 error: "
                      << errpL2 << "\t"
                      << pEL2 << std::endl;
        }
        errL2(0) = errvL2/vEL2;
        errL2(1) = errpL2/pEL2;
    }

    return errL2;
}

//! Cleans divergence, if needed
void IncompNSSolver
:: clean_divergence (std::shared_ptr<GridFunction> v) const
{
    if (m_bool_cleanDivg)
    {
        SparseMatrix *divNoBCs
                = m_discr->get_divNoBCs_matrix();
        double div_old = measure_divergence(divNoBCs,
                                            v.get());
        std::cout << "Weak divergence before cleaning: "
                  << div_old << std::endl;

        // make divergence free
        DivergenceFreeVelocity* makeDivFree
                = new DivergenceFreeVelocity(m_config,
                                             m_mesh);
        (*makeDivFree) (v.get());
        delete makeDivFree;

        double div_new = measure_divergence(divNoBCs,
                                            v.get());
        std::cout << "Weak divergence after cleaning: "
                  << div_new << std::endl;
    }
}

//! Computes the time-step according to the cfl conditions.
//! Assumes that all mesh elements have same geometry
double IncompNSSolver
:: compute_time_step
(std::shared_ptr<GridFunction>& v) const
{
    double dt=1E+5;
    double coeff = m_cfl_number/((m_deg+1)*(m_deg+1));

    // assume all cells have the same type
    const FiniteElement& fe = *(v->FESpace()->GetFE(0));
    const IntegrationRule *ir
            = &IntRules.Get(fe.GetGeomType(),
                            fe.GetOrder());

    // evaluates time step
    Vector vc(m_mesh->Dimension());
    for (int i=0; i<m_mesh->GetNE(); i++)
    {
        double hMin = coeff*m_mesh->GetElementSize(i, 1);

        for (int j=0; j<ir->GetNPoints(); j++)
        {
            const IntegrationPoint &ip = ir->IntPoint(j);
            v->GetVectorValue(i, ip, vc);
            double vc_norm = vc.Norml2();

            dt = std::fmin(dt, hMin/vc_norm);
        }
    }

    return dt;
}

//! Visualizes solution
void IncompNSSolver
:: visualize (std::shared_ptr<GridFunction> v,
              std::shared_ptr<GridFunction> p) const
{
    (*m_observer)(v);
    (*m_observer)(p);
    //m_observer->visualize_velocities(v);
    //m_observer->visualize_vorticity(v);
}

// End of file
