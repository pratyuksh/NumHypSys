#include "../../include/stokes/discretisation.hpp"
#include "../../include/mymfem/utilities.hpp"

#include <fstream>


// Constructor
StokesFEM :: StokesFEM (const nlohmann::json& config)
    : m_config (config)
{
    m_deg = config["deg_x"];
    m_testCase = set_stokes_test_case(config);

    std::cout << "\tFor Stokes system run "
              << config["problem_type"] << std::endl;

    m_penalty = config["penalty"];
}

// Destructor
StokesFEM :: ~StokesFEM()
{
    for (int i=0; i<m_fecs.Size(); i++) 
    {
        if (m_fecs[i]) { delete m_fecs[i]; }
        
        if (m_fespaces[i]) { delete m_fespaces[i]; }
    }

    if (m_stokesOp) { delete m_stokesOp; }

    if (m_stokesPr) { delete m_stokesPr; }

    if (m_stokesMat) { delete m_stokesMat; }

    if (m_vBdry) { delete m_vBdry; }

    if (m_rhsDirichlet) { delete m_rhsDirichlet; }
}

// Set-up the finite element spaces and boundary conditions
void StokesFEM :: set(std::shared_ptr<Mesh>& mesh)
{
    m_ndim = mesh->Dimension();

    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(m_deg, m_ndim);
    FiniteElementSpace *R_space
            = new FiniteElementSpace(mesh.get(), hdiv_coll);

    FiniteElementCollection *l2_coll
            = new L2_FECollection(m_deg, m_ndim);
    FiniteElementSpace *W_space
            = new FiniteElementSpace(mesh.get(), l2_coll);

    HYPRE_Int dimR = R_space->GetTrueVSize();
    HYPRE_Int dimW = W_space->GetTrueVSize();
    std::cout << "\nNumber of degrees of freedom in R_space: "
         << dimR << std::endl;
    std::cout << "Number of degrees of freedom in W_space: "
         << dimW << std::endl;
    std::cout << "Total number of degrees of freedom: "
         << dimR+dimW << std::endl;

    m_fecs.Append(hdiv_coll);
    m_fecs.Append(l2_coll);
    
    m_fespaces.Append(R_space);
    m_fespaces.Append(W_space);

    /// Define the two BlockStructure of the problem.
    m_block_offsets.SetSize(3);
    m_block_offsets[0] = 0;
    m_block_offsets[1] = R_space->GetVSize();
    m_block_offsets[2] = W_space->GetVSize();
    m_block_offsets.PartialSum();

    std::cout << "Block offsets:" << "\t"
              << m_block_offsets[0] << "\t"
              << m_block_offsets[1] << "\t"
              << m_block_offsets[2] << std::endl;
    
    // Mark boundary dofs for R_space
    if (mesh->bdr_attributes.Size())
    {
        m_ess_bdr_marker.SetSize(mesh->bdr_attributes.Max());
        m_testCase->set_bdry_dirichlet(m_ess_bdr_marker);

        m_nat_bdr_marker.SetSize(mesh->bdr_attributes.Max());
        m_nat_bdr_marker = 1;
        for (int i=0; i<m_nat_bdr_marker.Size(); i++) {
            if (m_ess_bdr_marker[i] == 1) {
                m_nat_bdr_marker[i] = 0;
            }
        }

        R_space->GetEssentialTrueDofs(m_ess_bdr_marker,
                                      m_ess_tdof_list);
    }

    // initialize rhs Dirichlet
    m_vBdry = new GridFunction(m_fespaces[0]);
    StokesBdryVelocityCoeff vBdry_coeff(m_testCase);
    (*m_vBdry) = 0.0;
    m_vBdry->ProjectBdrCoefficientNormal(vBdry_coeff,
                                         m_ess_bdr_marker);

    m_rhsDirichlet = new BlockVector(m_block_offsets);
    *m_rhsDirichlet = 0.0;
}

/// Assemble _diffusion_ and _divergence_ matrices
/*void StokesFEM :: assemble_system()
{
    Vector *temp1 = nullptr, *temp2 = nullptr; // dummy
    GridFunction *gdum = nullptr;

    // Divergence form: - \int_{\Omega} div(u_h) q_h d_{\Omega}
    MixedBilinearForm *div_form
            = new MixedBilinearForm(m_fespaces[0],
                                    m_fespaces[1]);
    ConstantCoefficient one(-1.0);
    div_form->AddDomainIntegrator
            (new VectorFEDivergenceIntegrator(one));
    div_form->Assemble();
    div_form->Finalize();
    div_form->EliminateTrialDofs(m_ess_bdr_marker,
                                 *temp1, *temp2);
    m_div = div_form->LoseMat();
    delete div_form;

    m_divT = Transpose(*m_div);

    // Diffusion form
    mymfem::MyBilinearForm *diffusion_form
            = new mymfem::MyBilinearForm(m_fespaces[0]);
    diffusion_form->MyAddDomainIntegrator
            (new mymfem::DiffusionIntegrator);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionConsistencyIntegrator);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionSymmetryIntegrator);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionPenaltyIntegrator(m_penalty));
    diffusion_form->MyAssemble(gdum);
    diffusion_form->Finalize();
    diffusion_form->EliminateEssentialBC(m_ess_bdr_marker);
    m_diffusion = diffusion_form->LoseMat();
    delete diffusion_form;
    //applyBCs(*m_diffusion);
}*/

/// Assemble _diffusion_ and _divergence_ matrices
void StokesFEM :: assemble_system()
{
    GridFunction *gdum = nullptr; // dummy

    // Divergence form: - \int_{\Omega} div(u_h) q_h d_{\Omega}
    MixedBilinearForm *div_form
            = new MixedBilinearForm(m_fespaces[0],
                                    m_fespaces[1]);
    ConstantCoefficient one(-1.0);
    div_form->AddDomainIntegrator
            (new VectorFEDivergenceIntegrator(one));
    div_form->Assemble();
    div_form->Finalize();
    div_form->EliminateTrialDofs(m_ess_bdr_marker,
                                 *m_vBdry, m_rhsDirichlet->GetBlock(1));
    m_div = div_form->LoseMat();
    delete div_form;

    m_divT = Transpose(*m_div);

    // Diffusion form
    mymfem::MyBilinearForm *diffusion_form
            = new mymfem::MyBilinearForm(m_fespaces[0]);
    diffusion_form->MyAddDomainIntegrator
            (new mymfem::DiffusionIntegrator);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionConsistencyIntegrator,
             m_ess_bdr_marker);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionSymmetryIntegrator,
             m_ess_bdr_marker);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionPenaltyIntegrator(m_penalty),
             m_ess_bdr_marker);
    diffusion_form->MyAssemble(gdum);
    diffusion_form->Finalize();
    diffusion_form->EliminateEssentialBC(m_ess_bdr_marker,
                                         *m_vBdry, m_rhsDirichlet->GetBlock(0));
    m_diffusion = diffusion_form->LoseMat();
    delete diffusion_form;

    // Open boundary form: \int_{\partial \Omega} (v_h \cdot n) p_h dS
    /*mymfem::MyMixedBilinearForm *openBdry_form
            = new mymfem::MyMixedBilinearForm(m_fespaces[1],
                                              m_fespaces[0]);
    openBdry_form->MyAddBoundaryFaceIntegrator
            (new mymfem::OpenBdryIntegrator,
             m_nat_bdr_marker);
    openBdry_form->MyAssemble();
    openBdry_form->Finalize();
    m_openBdry = openBdry_form->LoseMat();
    openBdry_form->MyClear();
    delete openBdry_form;*/
}

// Build stokesOp
void StokesFEM :: build_system_op()
{
    if (m_stokesOp) {
        delete m_stokesOp;
        m_stokesOp = nullptr;
    }

    m_stokesOp = new BlockOperator(m_block_offsets);
    m_stokesOp->SetBlock(0,0, m_diffusion);
    m_stokesOp->SetBlock(0,1, m_divT);
    //m_stokesOp->SetBlock(0,1, Add(*m_divT, *m_openBdry));
    m_stokesOp->SetBlock(1,0, m_div);
}

// Build system matrix from stokesOp in CSR format
void StokesFEM :: build_system_matrix()
{
    BlockMatrix *stokesBlockMat
                = new BlockMatrix(m_block_offsets);
    stokesBlockMat->SetBlock(0,0, m_diffusion);
    stokesBlockMat->SetBlock(0,1, m_divT);
    stokesBlockMat->SetBlock(1,0, m_div);

    if (m_stokesMat) {
        delete m_stokesMat;
        m_stokesMat = nullptr;
    }
    m_stokesMat = stokesBlockMat->CreateMonolithic();
    delete stokesBlockMat;
}

// Assemble _rhs_
void StokesFEM :: assemble_rhs(BlockVector* B)
{
    auto dimR = B->GetBlock(0).Size();
    Vector &B0 = B->GetBlock(0);
    assemble_source(B0);
    
    Vector buf(dimR);
    assemble_bdry(buf);
    B0.Add(1, buf);

    //applyBCs(B0);
    apply_BCs(B);

    //std::string f1 = "test_B0.txt";
    //std::ofstream os1(f1.c_str(), std::ofstream::out);
    //B0.Print(os1);
}

// Assemble source
void StokesFEM :: assemble_source(Vector& b)
{
    LinearForm source_form(m_fespaces[0]);
    StokesSourceCoeff f(m_testCase);
    source_form.AddDomainIntegrator
            (new VectorFEDomainLFIntegrator(f));
    source_form.Assemble();
    b = source_form.GetData();
}

// Assemble boundary
void StokesFEM :: assemble_bdry(Vector& b)
{
    LinearForm bdryDiffusion_form(m_fespaces[0]);
    StokesExactVelocityCoeff v(m_testCase);
    bdryDiffusion_form.AddBdrFaceIntegrator
            (new mymfem::BdryDiffusionConsistencyIntegrator(v),
             m_ess_bdr_marker);
    bdryDiffusion_form.AddBdrFaceIntegrator
            (new mymfem::BdryDiffusionPenaltyIntegrator(v, m_penalty),
             m_ess_bdr_marker);
    bdryDiffusion_form.Assemble();
    b = bdryDiffusion_form.GetData();
}

// Assemble preconditioner
void StokesFEM :: assemble_preconditioner ()
{
    SparseMatrix *tmp = Transpose(*m_diffusion);
    SparseMatrix *buf_mat = Transpose(*tmp);

    Solver *invM;
    invM = new GSSmoother(*buf_mat);
    invM->iterative_mode = false;

    m_stokesPr = new BlockDiagonalPreconditioner
            (m_block_offsets);
    m_stokesPr->SetDiagonalBlock(0, invM);
}

// Apply boundary conditions to SparseMatrix
void StokesFEM :: apply_BCs(SparseMatrix& A) const
{
    for (int k=0; k<m_ess_tdof_list.Size(); k++) {
        A.EliminateRowCol(m_ess_tdof_list[k]);
    }
}

// Apply boundary conditions to Vector
void StokesFEM :: apply_BCs(Vector& z) const
{
    z.SetSubVector(m_ess_tdof_list, 0.0);
}

// Apply boundary conditions to BlockVector
void StokesFEM :: apply_BCs(BlockVector* B) const
{
    Vector &B0 = B->GetBlock(0);
    Vector &rhsDirichletBlock0 = m_rhsDirichlet->GetBlock(0);
    B0 += rhsDirichletBlock0;
    for (int i=0; i<m_ess_tdof_list.Size(); i++) {
        B0(m_ess_tdof_list[i]) = rhsDirichletBlock0(m_ess_tdof_list[i]);
    }

    Vector &B1 = B->GetBlock(1);
    Vector &rhsDirichletBlock1 = m_rhsDirichlet->GetBlock(1);
    B1 += rhsDirichletBlock1;
}

// End of file
