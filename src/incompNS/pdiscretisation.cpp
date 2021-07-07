#include "../../include/incompNS/pdiscretisation.hpp"

#include <fstream>
#include <fmt/format.h>

#include "../../include/stokes/assembly.hpp"
#include "../../include/mymfem/mypbilinearform.hpp"
#include "../../include/mymfem/mypmixedbilinearform.hpp"
#include "../../include/mymfem/mypoperators.hpp"
#include "../../include/mymfem/putilities.hpp"


//! Constructors
IncompNSParFEM
:: IncompNSParFEM (MPI_Comm& comm,
                   const nlohmann::json& config)
    : m_config (config)
{
    MPI_Comm_rank(comm, &m_myrank);
    MPI_Comm_size(comm, &m_numprocs);

    m_deg = config["deg_x"];
    m_nfluxType = "upwind";
    m_bool_viscous = false;
    m_bool_mean_free_pressure = true;

    if (config.contains("numerical_flux")) {
        m_nfluxType = config["numerical_flux"];
    }

    if (config.contains("bool_viscous")) {
        m_bool_viscous = config["bool_viscous"];
        if (m_bool_viscous) {
            m_viscosity = config["viscosity"];
            m_penalty = config["penalty"];
        }
    }

    if (config.contains("bool_mean_free_pressure")) {
        m_bool_mean_free_pressure
                = config["bool_mean_free_pressure"];
    }

    if (m_myrank == 0) {
        std::cout << "\tFor Incompressible Navier-Stokes "
                     "system run "
                  << config["problem_type"] << std::endl;
    }
}

IncompNSParFEM
:: IncompNSParFEM
(MPI_Comm& comm, const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 bool bool_embedded_preconditioner)
    : IncompNSParFEM(comm, config)
{
    m_testCase = testCase;
    m_bool_embedded_preconditioner
            = bool_embedded_preconditioner;
}

//! Destructor
IncompNSParFEM :: ~IncompNSParFEM()
{
    // delete FE collections and spaces
    for (int i=0; i<m_fecs.Size(); i++)
    {
        if (m_fecs[i]) { delete m_fecs[i]; }
        if (m_fespaces[i]) { delete m_fespaces[i]; }
    }

    // delete variables used to apply BCs
    if (m_vBdry) { delete m_vBdry; }
    if (m_vBdryCoeff) { delete m_vBdryCoeff; }
    if (m_rhsDirichlet) { delete m_rhsDirichlet; }

    // delete convection and numerical flux integrators
    if (m_convInt) { delete m_convInt; }
    if (m_numFluxInt) { delete m_numFluxInt; }

    // delete assembles matrices
    if (m_mass) { delete m_mass; }
    if (m_divNoBCs) { delete m_divNoBCs; }
    if (m_diffusion) { delete m_diffusion; }
    if (m_massDiffusion) { delete m_massDiffusion; }
    if (m_convNflux) { delete m_convNflux; }
    if (!m_bool_viscous) {
        if (m_openBdry) { delete m_openBdry; }
        if (m_divT) { delete m_divT; }
    }
    if (m_invMass) { delete m_invMass; }

    // delete incompNS operator blocks
    if (m_block00Mat) { delete m_block00Mat; }
    if (m_block01Mat) { delete m_block01Mat; }
    if (m_div) { delete m_div; }
    if (m_incompNSOp) {
        // does not own blocks
        m_incompNSOp->owns_blocks = 0;
        delete m_incompNSOp;
    }

    // delete incompNS matrix
    if (m_incompNSMat) { delete m_incompNSMat; }

    // delete incompNS preconditioner
    if (m_incompNSPr && !m_bool_embedded_preconditioner)
    {
        auto blockDiagPr
                = dynamic_cast<BlockDiagonalPreconditioner*>
                (m_incompNSPr);
        auto blockTriPr
                = dynamic_cast
                <mymfem::IncompNSParBlockTriPr*>
                (m_incompNSPr);

        // delete block diagonal preconditioner
        if (blockDiagPr) {
            // owns blocks
            blockDiagPr->owns_blocks = 1;
            delete blockDiagPr;

            if (m_invMatS && m_bool_mean_free_pressure)
                { delete m_invMatS; }
        }

        // delete block tri-diagonal preconditioner
        if (blockTriPr) {
            if (m_invMatS) { delete m_invMatS; }
            if (m_blockPr00) { delete m_blockPr00; }
            if (m_blockPr11) { delete m_blockPr11; }
            delete blockTriPr;
        }
    }
    if (m_matS) { delete m_matS; }
    if (m_block00Diag) { delete m_block00Diag; }
    if (m_block00InvMultDivuT)
    { delete m_block00InvMultDivuT; }

    // deallocate memory of embedded preconditioner
    if (m_bool_embedded_preconditioner)
    {
        if (m_incompNSPr)
        {
            auto blockDiagPr
                    = dynamic_cast
                    <BlockDiagonalPreconditioner*>
                    (m_incompNSPr);
            if (blockDiagPr) {
                blockDiagPr->owns_blocks = 1;
                delete blockDiagPr;
            }
            if (m_invPrP1) { delete m_invPrP1; }
            if (m_prP2) { delete m_prP2; }
            if (m_incompNSPrOp) { delete m_incompNSPrOp; }
        }
    }
}

//! Sets FE spaces and BCs
void IncompNSParFEM :: set(std::shared_ptr<ParMesh>& pmesh)
{
    m_ndim = pmesh->Dimension();

    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(m_deg, m_ndim);
    ParFiniteElementSpace *R_space
            = new ParFiniteElementSpace(pmesh.get(),
                                        hdiv_coll);

    FiniteElementCollection *l2_coll
            = new L2_FECollection(m_deg, m_ndim);
    ParFiniteElementSpace *W_space
            = new ParFiniteElementSpace(pmesh.get(),
                                        l2_coll);

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimW = W_space->GlobalTrueVSize();
    if (m_myrank == 0)
    {
        std::cout << "\nNumber of degrees of freedom "
                     "in R_space: "
                  << dimR << std::endl;
        std::cout << "Number of degrees of freedom "
                     "in W_space: "
                  << dimW << std::endl;
        std::cout << "Total number of degrees of freedom: "
                  << dimR+dimW << std::endl;
    }

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
              << m_myrank << "\t"
              << m_block_offsets[0] << "\t"
              << m_block_offsets[1] << "\t"
              << m_block_offsets[2] << std::endl;

    m_block_trueOffsets.SetSize(3);
    m_block_trueOffsets[0] = 0;
    m_block_trueOffsets[1] = R_space->TrueVSize();
    m_block_trueOffsets[2] = W_space->TrueVSize();
    m_block_trueOffsets.PartialSum();

    std::cout << "Block true offsets:" << "\t"
              << m_myrank << "\t"
              << m_block_trueOffsets[0] << "\t"
              << m_block_trueOffsets[1] << "\t"
              << m_block_trueOffsets[2] << std::endl;

    // Mark boundary dofs for R_space
    if (pmesh->bdr_attributes.Size())
    {
        m_ess_bdr_marker.SetSize
                (pmesh->bdr_attributes.Max());
        m_testCase->set_bdry_dirichlet(m_ess_bdr_marker);

        m_nat_bdr_marker.SetSize
                (pmesh->bdr_attributes.Max());
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
    m_vBdryCoeff
            = new IncompNSBdryVelocityCoeff (m_testCase);
    m_vBdry = new ParGridFunction(m_fespaces[0]);
    m_rhsDirichlet = new BlockVector(m_block_trueOffsets);
}

//! Initializes discretisation
void IncompNSParFEM :: init(const double dt)
{
    init_assemble_system(dt);
    init_system_op();
}

//! Initial assembly of system
//! Assembles mass, divergence, diffusion
//! Initializes convection
void IncompNSParFEM
:: init_assemble_system(const double dt)
{
    assemble_mass();
    assemble_divergence();
    init_convection();

    // diffusion SIP form
    if (m_bool_viscous)
    {
        assemble_diffusion();
        auto *tmp = new HypreParMatrix(*m_diffusion);
        (*tmp) *= dt*m_viscosity;
        m_massDiffusion = ParAdd(m_mass, tmp);
        delete tmp;
    }

    // matrix block 01
    if (m_bool_viscous) {
        m_block01Mat = m_divT;
    }
    else {
        assemble_openBdry();
        m_block01Mat = ParAdd(m_divT, m_openBdry);
    }
}

//! Assembles mass operator
//! Mass form: \int_{\Omega} u_h v_h d_{\Omega}
void IncompNSParFEM :: assemble_mass()
{
    ParBilinearForm *mass_form
            = new ParBilinearForm(m_fespaces[0]);
    mass_form->AddDomainIntegrator
            (new VectorFEMassIntegrator);
    mass_form->Assemble();
    mass_form->Finalize();
    m_mass = mass_form->ParallelAssemble();
    delete mass_form;
}

//! Assembles divergence operator
//! Divergence form:
//! \int_{\Omega} div(u_h) q_h d_{\Omega}
void IncompNSParFEM :: assemble_divergence()
{
    Vector *temp1 = nullptr, *temp2 = nullptr; // dummy

    ParMixedBilinearForm *div_form
            = new ParMixedBilinearForm(m_fespaces[0],
                                       m_fespaces[1]);
    ConstantCoefficient one(-1.0);
    div_form->AddDomainIntegrator
            (new VectorFEDivergenceIntegrator(one));
    div_form->Assemble();
    div_form->Finalize();
    div_form->EliminateTrialDofs(m_ess_bdr_marker,
                                 *temp1, *temp2);
    m_div = div_form->ParallelAssemble();
    delete div_form;

    m_divT = m_div->Transpose();

    // stupid test
    div_form = new ParMixedBilinearForm(m_fespaces[0],
                                        m_fespaces[1]);
    div_form->AddDomainIntegrator
            (new VectorFEDivergenceIntegrator(one));
    div_form->Assemble();
    div_form->Finalize();
    m_divNoBCs = div_form->ParallelAssemble();
    delete div_form;
}

//! Assembles SIP diffusion operator
void IncompNSParFEM :: assemble_diffusion()
{
    ParGridFunction *gdum = nullptr; // dummy

    mymfem::MyParBilinearForm *diffusion_form
            = new mymfem::MyParBilinearForm
            (m_fespaces[0]);
    diffusion_form->MyAddDomainIntegrator
            (new mymfem::DiffusionIntegrator);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionConsistencyIntegrator,
             m_ess_bdr_marker);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionSymmetryIntegrator,
             m_ess_bdr_marker);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionPenaltyIntegrator
             (m_penalty),
             m_ess_bdr_marker);
    diffusion_form->MyAssemble(gdum);
    diffusion_form->Finalize();
    m_diffusion = diffusion_form->MyParallelAssemble();
    diffusion_form->MyClear();
    delete diffusion_form;
}

//! Initializes convection operator
void IncompNSParFEM
:: init_convection()
{
    // convection and numerical flux integrators
    m_convInt = new mymfem::ConvectionIntegrator;
    if (m_nfluxType == "central") {
        m_numFluxInt
                = new mymfem::CentralNumFluxIntegrator;
    }
    else if (m_nfluxType == "upwind") {
        m_numFluxInt
                = new mymfem::UpwindNumFluxIntegrator;
    }
}

//! Assembles/updates convection operator
void IncompNSParFEM
:: update_convection(ParGridFunction *v)
{
    reset_convection();

    // set integration rules
    // assumes that all elements have the same geometry
    auto elGeomType = m_fespaces[0]
            ->GetFE(0)->GetGeomType();
    auto fGeomType = m_fespaces[0]
            ->GetFaceElement(0)->GetGeomType();

    IntegrationRules rule1{};
    int order = 3*(m_deg+1)-1;
    const IntegrationRule& convIntRule
            = rule1.Get(elGeomType, order);
    m_convInt->SetIntRule(&convIntRule);

    IntegrationRules rule2{};
    order = 3*(m_deg+1);
    const IntegrationRule &numFluxIntRule
            = rule2.Get(fGeomType, order);
    m_numFluxInt->SetIntRule(&numFluxIntRule);

    // assemble
    mymfem::MyParBilinearForm *convNFlux_form
            = new mymfem::MyParBilinearForm(m_fespaces[0]);
    convNFlux_form->MyAddDomainIntegrator(m_convInt);
    convNFlux_form->MyAddInteriorFaceIntegrator
            (m_numFluxInt);
    convNFlux_form->MyAssemble(v);
    convNFlux_form->Finalize();
    m_convNflux = convNFlux_form->MyParallelAssemble();
    delete convNFlux_form;
}

//! Assembles open boundary operator
//! Open boundary form:
//! \int_{\partial \Omega} (v_h \cdot n) p_h dS
void IncompNSParFEM :: assemble_openBdry()
{
    mymfem::MyParMixedBilinearForm *openBdry_form
            = new mymfem::MyParMixedBilinearForm
            (m_fespaces[1], m_fespaces[0]);
    openBdry_form->MyAddBoundaryFaceIntegrator
            (new mymfem::OpenBdryIntegrator,
             m_nat_bdr_marker);
    openBdry_form->MyAssemble();
    openBdry_form->Finalize();
    m_openBdry = openBdry_form->ParallelAssemble();
    openBdry_form->MyClear();
    delete openBdry_form;
}

//! Assembles inverse mass operator
void IncompNSParFEM :: assemble_invMass()
{
    auto invMassBf
            = new ParBilinearForm(m_fespaces[0]);
    invMassBf->AddDomainIntegrator
            (new InverseIntegrator
             (new VectorFEMassIntegrator));
    invMassBf->Assemble();
    invMassBf->Finalize();
    m_invMass = invMassBf->ParallelAssemble();
    delete invMassBf;
}

//! Assembles source
void IncompNSParFEM
:: assemble_source(const double t, Vector& b) const
{
    ParLinearForm source_form(m_fespaces[0]);
    IncompNSSourceCoeff f(m_testCase);
    f.SetTime(t);
    source_form.AddDomainIntegrator
            (new VectorFEDomainLFIntegrator(f));
    source_form.Assemble();
    source_form.ParallelAssemble(b);
}

//! Assembles boundary
void IncompNSParFEM :: assemble_bdry(const double t,
                                     Vector& b)
{
    ParLinearForm bdryDiffusion_form(m_fespaces[0]);
    m_vBdryCoeff->SetTime(t);
    bdryDiffusion_form.AddBdrFaceIntegrator
            (new mymfem::BdryDiffusionConsistencyIntegrator
             (*m_vBdryCoeff), m_ess_bdr_marker);
    bdryDiffusion_form.AddBdrFaceIntegrator
            (new mymfem::BdryDiffusionPenaltyIntegrator
             (*m_vBdryCoeff, m_penalty), m_ess_bdr_marker);
    bdryDiffusion_form.Assemble();
    bdryDiffusion_form.ParallelAssemble(b);
}

//! Updates boundary velocity
void IncompNSParFEM :: update_vBdry(double t)
{
    m_vBdryCoeff->SetTime(t);
    (*m_vBdry) = 0.0;
    m_vBdry->ProjectBdrCoefficientNormal(*m_vBdryCoeff,
                                         m_ess_bdr_marker);
}

//! Applies boundary conditions to Vector
void IncompNSParFEM :: apply_BCs(Vector& z) const
{
    z.SetSubVector(m_ess_tdof_list, 0.0);
}

//! Applies boundary conditions to BlockVector
void IncompNSParFEM :: apply_BCs(BlockVector& B) const
{
    B.Add(1, *m_rhsDirichlet);
    apply_vBdry(B.GetBlock(0));

    /*Vector vBdry;
    m_vBdry->GetTrueDofs(vBdry);
    B.Add(1, *m_rhsDirichlet);
    for (int i=0; i<m_ess_tdof_list.Size(); i++) {
        B(m_ess_tdof_list[i]) = vBdry(m_ess_tdof_list[i]);
    }*/
}

//! Applies essential velocity boundary conditions
void IncompNSParFEM :: apply_vBdry(Vector &B0) const
{
    Vector vBdry;
    m_vBdry->GetTrueDofs(vBdry);
    for (int i=0; i<m_ess_tdof_list.Size(); i++) {
        B0(m_ess_tdof_list[i])
                = vBdry(m_ess_tdof_list[i]);
    }
}

//! Applies boundary conditions to HypreParMatrix
void IncompNSParFEM :: apply_BCs(HypreParMatrix& A) const
{
    HypreParMatrix *Ae
            = A.EliminateRowsCols(m_ess_tdof_list);
    delete Ae;
}

// End of file
