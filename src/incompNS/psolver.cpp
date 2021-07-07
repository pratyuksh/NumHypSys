#include "../../include/incompNS/psolver.hpp"
#include "../../include/incompNS/putilities.hpp"

#include <iostream>


//! Constructors
IncompNSParSolver
:: IncompNSParSolver (MPI_Comm comm,
                      const nlohmann::json config)
    : m_comm (comm), m_config(config)
{
    MPI_Comm_rank(comm, &m_myrank);

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

IncompNSParSolver
:: IncompNSParSolver
(MPI_Comm comm, const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase)
    : IncompNSParSolver(comm, config)
{
    m_testCase = testCase;
}

IncompNSParSolver
:: IncompNSParSolver
(MPI_Comm comm, const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir, const int lx, const int Nt)
    : IncompNSParSolver(comm, config, testCase)
{
    m_Nt = Nt;
    set(mesh_dir, lx);
}

//! Destructor
IncompNSParSolver
:: ~IncompNSParSolver ()
{
    clear();
}

//! Sets mesh and observer
void IncompNSParSolver :: set(const std::string mesh_dir,
                           const int lx)
{
    if(m_myrank == 0) {
        std::cout << "\nRun Incompressible Navier-Stokes "
                     "solver ..."
                  << std::endl;
    }

    // test case
    if (!m_testCase) {
        m_testCase = make_incompNS_test_case(m_config);
    }

    // parallel mesh
    if (m_bool_read_mesh_from_file) {
        const std::string mesh_file
                = mesh_dir+m_mesh_elem_type+"_mesh_l"
                +std::to_string(lx-m_num_refinements)
                +"."+m_mesh_format;

        if(m_myrank == 0) {
            std::cout << "  Mesh file: "
                      << mesh_file << std::endl;
            std::cout << "  Number of uniform "
                         "mesh refinements: "
                      << m_num_refinements << std::endl;
        }

        Mesh *mesh = new Mesh(mesh_file.c_str());
        for (int k=0; k<m_num_refinements; k++) {
            mesh->UniformRefinement();
        }
        m_pmesh = std::make_shared<ParMesh>(m_comm, *mesh);
        delete mesh;
    }
    else {
        m_mesh_dir = mesh_dir;
        const std::string mesh_file
                = mesh_dir+m_mesh_elem_type+"_mesh_l0."
                +m_mesh_format;

        if(m_myrank == 0) {
            std::cout << "  Base mesh file: "
                      << mesh_file << std::endl;
            std::cout << "  Number of uniform "
                         "mesh refinements: "
                      << lx << std::endl;
        }

        Mesh *mesh = new Mesh(mesh_file.c_str());
        for (int k=0; k<lx; k++) {
            mesh->UniformRefinement();
        }
        m_pmesh = std::make_shared<ParMesh>(m_comm, *mesh);
        delete mesh;
    }

    std::tie(m_hMin, m_hMax) = eval_hMinMax();
    if (m_myrank == 0) {
        std::cout << "  Minimum cell size: "
                  << m_hMin << std::endl;
        std::cout << "  Maximum cell size: "
                  << m_hMax << std::endl;
    }

    // observer
    m_observer = std::make_shared<IncompNSParObserver>
            (m_comm, m_config, m_testCase, lx);
}

//! Initializes and runs solver
void IncompNSParSolver
:: operator()(std::unique_ptr<BlockVector>& U)
{
    init ();
    run (U);
}

//! Projects initial conditions
void IncompNSParSolver
:: project_init_conditions
(std::unique_ptr<BlockVector>& U,
 std::shared_ptr<ParGridFunction>& v,
 std::shared_ptr<ParGridFunction>& p) const
{
    IncompNSInitialVelocityCoeff v0_coeff(m_testCase);

    // new velocity projection routine
    auto fespaces = m_discr->get_fespaces();
    auto mass = m_discr->get_mass_matrix();

    // rhs \int v0_coeff*test_fn d\Omega
    Vector v0(fespaces[0]->GetTrueVSize());
    ParLinearForm v0_form(fespaces[0]);
    v0_form.AddDomainIntegrator
            (new VectorFEDomainLFIntegrator(v0_coeff));
    v0_form.Assemble();
    v0_form.ParallelAssemble(v0);

    // CG solver + AMG preconditioner
    double tol = 1E-12;
    int maxIter = 10000;

    auto amg = new HypreBoomerAMG();
    amg->SetPrintLevel(0);
    amg->SetOperator(*mass);

    auto cgsolver = new HyprePCG(*mass);
    cgsolver->SetPreconditioner(*amg);
    cgsolver->SetTol(tol);
    cgsolver->SetMaxIter(maxIter);
    cgsolver->SetPrintLevel(0);

    // solve projection
    U->GetBlock(0) = 0.;
    cgsolver->Mult(v0, U->GetBlock(0));

    // release memory
    v0.Destroy();
    delete cgsolver;
    delete amg;

    // set grid function
    v->Distribute(&(U->GetBlock(0)));

    // projection routine provided by mfem
    //v->ProjectCoefficient(v0_coeff);
    //v->GetTrueDofs(U->GetBlock(0));

    // velocity divergence cleaning, if needed
    clean_divergence(v);

    // pressure initial conditions
    IncompNSInitialPressureCoeff p0_coeff(m_testCase);
    p0_coeff.SetTime(0);
    p->ProjectCoefficient(p0_coeff);
    p->GetTrueDofs(U->GetBlock(1));
}

//! Runs initialized solver
void IncompNSParSolver
:: run (std::unique_ptr<BlockVector>& U,
        int sampleId) const
{
    const double dt = m_tEnd/m_Nt;
    auto fespaces = m_discr->get_fespaces();

    // MFEM solution and rhs variables
    U = std::make_unique<BlockVector>
            (m_block_trueOffsets);
    auto B = std::make_unique<BlockVector>
            (m_block_trueOffsets);

    // auxiliary variables
    auto U_buf
            = std::make_unique<BlockVector>(m_block_offsets);
    auto v = std::make_shared<ParGridFunction>
            (fespaces[0],U_buf->GetData());
    auto p = std::make_shared<ParGridFunction>
            (fespaces[1], U_buf->GetData()
            + m_block_offsets[1]);

    // initial conditions
    project_init_conditions(U, v, p);
    m_observer->dump_sol(v, p, sampleId, 0);
    //visualize(v, p);

    // scale pressure dofs for convenience
    U->GetBlock(1) *= dt;

    // integrate from time 0 -> T
    for (int n=1; n<=m_Nt; n++)
    {
        /// one step: t_{n} -> t_{n+1}
        solve_one_step(n, v, U.get(), B.get());
        v->Distribute(&(U->GetBlock(0)));

        /// write solution
       p->Distribute(&(U->GetBlock(1)));
       (*p) *= (1./dt);
       m_observer->dump_sol(v, p, sampleId, n);
    }
    U->GetBlock(1) *= (1./dt);

    /*p->Distribute(&(U->GetBlock(1)));
    visualize(v, p);*/
}

//! Initializes and runs solver
//! Computes the solution error if needed
std::pair<double, Eigen::VectorXd> IncompNSParSolver
:: operator() (void)
{
    Eigen::VectorXd errL2(2);
    errL2.setZero();

    const double dt = m_tEnd/m_Nt;

    init ();
    auto fespaces = m_discr->get_fespaces();

    // MFEM solution and rhs variables
    auto U = std::make_unique<BlockVector>
            (m_block_trueOffsets);
    auto B = std::make_unique<BlockVector>
            (m_block_trueOffsets);

    // auxiliary variables
    auto U_buf
            = std::make_unique<BlockVector>(m_block_offsets);
    auto v = std::make_shared<ParGridFunction>
            (fespaces[0], U_buf->GetData());
    auto p = std::make_shared<ParGridFunction>
            (fespaces[1], U_buf->GetData()
            + m_block_offsets[1]);

    // initial conditions
    project_init_conditions(U, v, p);
    visualize(v, p);

    // routine for making pressure mean free
    ParMeanFreePressure *meanFreePressure = nullptr;
    if (m_bool_embedded_preconditioner &&
            m_bool_mean_free_pressure)
    {
        meanFreePressure
                = new ParMeanFreePressure (fespaces[1]);
    }

    // scale pressure dofs for convenience
    U->GetBlock(1) *= dt;

    // dump initial sol
    m_observer->dump_sol(v, p, 0);

    // integrate from time 0 -> T
    for (int n=1; n<=m_Nt; n++)
    {
        /// one step: t_{n} -> t_{n+1}
        solve_one_step(n, v, U.get(), B.get());
        v->Distribute(&(U->GetBlock(0)));
        p->Distribute(&(U->GetBlock(1)));

        /// make pressure mean free
        if (m_bool_embedded_preconditioner &&
                m_bool_mean_free_pressure)
        {
            (*meanFreePressure)(*p);
            p->GetTrueDofs(U->GetBlock(1));
            double p_mean = meanFreePressure->get(*p);
            if (m_myrank == 0) {
                std::cout << "Mean pressure: "
                          << p_mean << std::endl;
            }
        }

        (*p) /= dt;
        compute_error(dt*n, v, p);

        /// dump solution
        m_observer->dump_sol(v, p, n);
    }
    visualize(v, p);
    errL2 = compute_error(m_tEnd, v, p);
    if (meanFreePressure) { delete meanFreePressure; }

    return {m_hMax, errL2};
}

//! Deletes memory
void IncompNSParSolver
:: clear ()
{
    if (m_solver) { delete m_solver; }
    if (m_precond) { delete m_precond; }
}

//! Initializes linear system solver
void IncompNSParSolver :: init_linear_solver() const
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

            MINRESSolver *minres
                    = new MINRESSolver (m_comm);
            minres->SetAbsTol(absTol);
            minres->SetRelTol(relTol);
            minres->SetMaxIter(maxIter);
            minres->SetPrintLevel(verbose);
            m_solver = minres;
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

            GMRESSolver *gmres = new GMRESSolver (m_comm);
            gmres->SetAbsTol(absTol);
            gmres->SetRelTol(relTol);
            gmres->SetMaxIter(maxIter);
            gmres->SetPrintLevel(verbose);
            gmres->SetKDim(kDim);
            m_solver = gmres;
        }
    }
    else { MFEM_ABORT("\nUnknown linear system solver!") }

    {
        m_solver->iterative_mode = true;
        if (m_config.contains("iterative_mode")) {
            m_solver->iterative_mode
                    = m_config["iterative_mode"];
        }
    }
}

//! Updates linear system solver
void IncompNSParSolver :: update_linear_solver() const
{
    std::string linear_solver_type
            = m_config["linear_solver_type"];

    {
        if (m_incompNSOp) {
            m_solver->SetOperator(*m_incompNSOp);
        }
    }
}

//! Initializes preconditioner
void IncompNSParSolver
:: init_preconditioner () const
{
    std::string precond_type = m_config["precond_type"];
    m_discr->init_preconditioner(precond_type);
}

//! Updates preconditioner
void IncompNSParSolver
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
    else if (linear_solver_type == "gmres" && m_incompNSPr)
    {
        auto gmres = static_cast<GMRESSolver *>(m_solver);
        gmres->SetPreconditioner(*m_incompNSPr);
    }
}

//! Computes error for velocity and pressure
Eigen::VectorXd IncompNSParSolver
:: compute_error (double t,
                  std::shared_ptr<ParGridFunction> v,
                  std::shared_ptr<ParGridFunction> p) const
{
    Eigen::VectorXd errL2(2);
    errL2.setZero();

    if (m_bool_error)
    {
        double errvL2, vEL2, errpL2, pEL2;
        std::tie (errvL2, vEL2, errpL2, pEL2)
                = m_observer->eval_error_L2
                (t, v, p);
        if (m_myrank == 0)
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
void IncompNSParSolver
:: clean_divergence
(std::shared_ptr<ParGridFunction> v) const
{
    if (m_bool_cleanDivg)
    {
        HypreParMatrix *div
                = m_discr->get_divNoBCs_matrix();
        double div_old = measure_divergence(div, v.get());
        if (m_myrank == 0) {
            std::cout << "Weak divergence before cleaning: "
                      << div_old << std::endl;
        }

        // make divergence free
        ParDivergenceFreeVelocity* makeDivFree
                = new ParDivergenceFreeVelocity(m_config,
                                                m_pmesh);
        (*makeDivFree) (v.get());
        delete makeDivFree;

        double div_new = measure_divergence(div, v.get());
        if (m_myrank == 0) {
            std::cout << "Weak divergence after cleaning: "
                      << div_new << std::endl;
        }
    }
}

//! Computes the time-step according to the cfl conditions.
//! Assumes that all mesh elements have same geometry
double IncompNSParSolver
:: compute_time_step
(std::shared_ptr<ParGridFunction>& v) const
{
    double dt;
    double mydt=1E+5;
    double coeff = m_cfl_number/((m_deg+1)*(m_deg+1));

    // assume all cells have the same type
    const FiniteElement& fe = *(v->FESpace()->GetFE(0));
    const IntegrationRule *ir
            = &IntRules.Get(fe.GetGeomType(),
                            fe.GetOrder());

    // evaluates minimum time step for processor
    Vector vc(m_pmesh->Dimension());
    for (int i=0; i<m_pmesh->GetNE(); i++)
    {
        double hMin = coeff*m_pmesh->GetElementSize(i, 1);

        for (int j=0; j<ir->GetNPoints(); j++)
        {
            const IntegrationPoint &ip = ir->IntPoint(j);
            v->GetVectorValue(i, ip, vc);
            double vc_norm = vc.Norml2();

            mydt = std::fmin(mydt, hMin/vc_norm);
        }
    }

    // global minimum time step
    MPI_Allreduce(&mydt, &dt, 1,
                  MPI_DOUBLE, MPI_MIN, m_comm);

    return dt;
}

//! Visualizes solution
void IncompNSParSolver
:: visualize (std::shared_ptr<ParGridFunction> v,
              std::shared_ptr<ParGridFunction> p) const
{
    //(*m_observer)(v);
    //(*m_observer)(p);
    m_observer->visualize_velocities(v);
    //m_observer->visualize_vorticity(v);
}


// End of file
