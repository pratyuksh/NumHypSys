#include "../../include/stokes/solver.hpp"

#include <iostream>
using namespace std;


StokesSolver :: StokesSolver (const nlohmann::json& config,
                              std::string mesh_dir, 
                              const int lx)
    : m_config(config)
{
    m_bool_error = config["eval_error"];

    m_mesh_format = "mesh";
    if (config.contains("mesh_format")) {
        m_mesh_format = m_config["mesh_format"];
    }

    m_mesh_elem_type = "quad";
    if (config.contains("mesh_elem_type")) {
        m_mesh_elem_type = m_config["mesh_elem_type"];
    }
    
    const std::string mesh_file
            = mesh_dir+m_mesh_elem_type+"_mesh_l0."
            +m_mesh_format;

    cout << "\nRun Stokes solver ..." << endl;
    cout << "  Base mesh file: " << mesh_file << endl;
    cout << "  Number of uniform mesh refinements: " << lx << endl;

    // mesh
    m_mesh = std::make_shared<Mesh>(mesh_file.c_str());
    for (int k=0; k<lx; k++) {
        m_mesh->UniformRefinement();
    }

    // discretisation
    m_discr = std::make_unique<StokesFEM>(config);

    // observer
    m_observer = std::make_unique<StokesObserver> (config, lx);

    if (config["problem_type"] == "square_test3") {
        m_bool_mean_free_pressure = false;
    }
}

StokesSolver :: ~ StokesSolver ()
{
    if (m_solver) { delete m_solver; }
}


std::pair<double, Eigen::VectorXd> StokesSolver :: operator() (void)
{
    Eigen::VectorXd errL2(2);
    errL2.setZero();
    
    init ();

    std::shared_ptr<StokesTestCases> testCase = m_discr->get_test_case();
    Array<FiniteElementSpace*> fespaces = m_discr->get_fespaces();

    // MFEM solution variables
    std::unique_ptr <BlockVector> U
            = std::make_unique<BlockVector>(m_block_offsets);
    
    std::shared_ptr <GridFunction> v
            = std::make_shared<GridFunction>();
    std::shared_ptr <GridFunction> p
            = std::make_shared<GridFunction>();

    v->MakeRef(fespaces[0], U->GetBlock(0));
    p->MakeRef(fespaces[1], U->GetBlock(1));

    // routines for making pressure mean free
    MeanFreePressure mean_free_pressure(fespaces[1]);

    // MFEM rhs variables
    std::unique_ptr <BlockVector> B
            = std::make_unique<BlockVector>(m_block_offsets);

    // solve
    solve (U.get(), B.get());
    if (m_bool_mean_free_pressure) { mean_free_pressure(*p); }
    (*m_observer)(v);
    (*m_observer)(p);

    if (m_bool_error)
    {
        double errvL2, vEL2, errpL2, pEL2;
        std::tie (errvL2, vEL2, errpL2, pEL2)
                = m_observer->eval_error_L2(testCase, v, p);
        {
            cout << "velocity L2 error: " << errvL2
                 << "\t" << vEL2 << endl;
            cout << "pressure L2 error: " << errpL2
                 << "\t" << pEL2 << endl;
        }
        errL2(0) = errvL2/vEL2;
        errL2(1) = errpL2/pEL2;
    }

    double h_min, h_max, kappa_min, kappa_max;
    m_mesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

    return {h_max, errL2};
}


void StokesSolver :: init ()
{
    /// assemble system
    m_discr->set(m_mesh);
    m_block_offsets = m_discr->get_block_offsets();
    m_discr->assemble_system();

    m_discr->build_system_op();
    m_stokesOp = m_discr->get_stokes_op();

    set_linear_solver();
    set_preconditioner();
}


void StokesSolver :: solve (BlockVector *U, 
                            BlockVector *B) const
{
    (*B) = 0.0;
    m_discr->assemble_rhs(B);

    (*U) = 0.0;
    m_solver->Mult(*B, *U);
}


// Set the linear system solver
void StokesSolver :: set_linear_solver() const
{
    std::string linear_solver_type =
            m_config["linear_solver_type"];

    if (m_solver) {
        delete m_solver;
        m_solver = nullptr;
    }

    if (linear_solver_type == "minres")
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
    else if (linear_solver_type == "gmres")
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
    else { MFEM_ABORT("\nUnknown linear system solver!") }

    m_solver->SetOperator(*m_stokesOp);
    m_solver->iterative_mode = true;
}


// Set the preconditioner
void StokesSolver :: set_preconditioner () const
{
    m_discr->assemble_preconditioner();
    m_stokesPr = m_discr->get_stokes_pr();

    std::string linear_solver_type =
            m_config["linear_solver_type"];

    if (linear_solver_type == "minres" && m_stokesPr)
    {
        auto minres = static_cast<MINRESSolver *>(m_solver);
        minres->SetPreconditioner(*m_stokesPr);
    }
    else if (linear_solver_type == "gmres" && m_stokesPr)
    {
        auto gmres = static_cast<GMRESSolver *>(m_solver);
        gmres->SetPreconditioner(*m_stokesPr);
    }
}

// End of file
