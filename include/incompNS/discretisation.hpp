#ifndef INCOMPNS_DISCRETISATION_HPP
#define INCOMPNS_DISCRETISATION_HPP

#include "mfem.hpp"
using namespace mfem;

#include <iostream>

#include "../core/config.hpp"
#include "../mymfem/mybilinearform.hpp"
#include "../mymfem/mymixedbilinearform.hpp"

#include "test_cases.hpp"
#include "coefficients.hpp"
#include "assembly.hpp"
#include "operators.hpp"


//! H(div) FEM discretisation
//! for Incompressible Navier-Stokes
class IncompNSFEM
{
public:
    //! Constructors
    IncompNSFEM (const nlohmann::json&);

    IncompNSFEM (const nlohmann::json&,
                 std::shared_ptr<IncompNSTestCases>&,
                 bool bool_embedded_preconditioner=false);

    //! Destructor
    virtual ~IncompNSFEM ();
    
    //! Sets FE spaces and BCs
    void set(std::shared_ptr<Mesh>&);

    //! Initializes discretisation
    virtual void init(const double);

    //! Initial assembly of system
    virtual void init_assemble_system(const double);

    //! Assembles mass operator
    void assemble_mass();

    //! Assembles divergence operator
    void assemble_divergence();

    //! Assembles SIP diffusion operator
    void assemble_diffusion();

    //! Initializes convection operator
    void init_convection();

    //! Assembles/updates convection operator
    void update_convection(GridFunction *);
    void update_convection(Vector&);

    //! Assembles open boundary operator
    void assemble_openBdry();

    //! Assembles inverse mass operator
    void assemble_invMass();

    //! Updates system
    virtual void update_system(const double,
                                 const double,
                                 GridFunction *)
    {
        std::cout << "Assemble system not implemented "
                     "for the class IncompNSFEM"
                  << std::endl;
        abort();
    }

    //! Initializes system operator
    virtual void init_system_op()
    {
        std::cout << "Init build system not implemented "
                     "for the class IncompNSFEM"
                  << std::endl;
        abort();
    }

    //! Updates system operator
    virtual void update_system_op()
    {
        std::cout << "Build system op not implemented "
                     "for the class IncompNSFEM"
                  << std::endl;
        abort();
    }

    //! Initializes system operator with
    //! embedded preconditioner
    virtual void init_preconditioned_system_op()
    {
        std::cout << "Init preconditioned system Op "
                     "not implemented "
                     "for the class IncompNSParFEM"
                  << std::endl;
        abort();
    }

    //! Updates system operator with
    //! embedded preconditioner
    virtual void update_preconditioned_system_op()
    {
        std::cout << "Update preconditioned system Op "
                     "not implemented "
                     "for the class IncompNSParFEM"
                  << std::endl;
        abort();
    }

    //! Solves system operator with
    //! embedded preconditioner
    virtual void preconditioned_solve
    (Solver *, const BlockVector&, BlockVector&) const
    {
        std::cout << "Preconditioned system solve"
                     "not implemented "
                     "for the class IncompNSParFEM"
                  << std::endl;
        abort();
    }

    //! Initializes preconditioner
    void init_preconditioner(const std::string);

    //! Updates preconditioner
    void update_preconditioner(const std::string,
                               const double);

    //! Updates right-hand side
    virtual void update_rhs
    (double, double,
     BlockVector*, BlockVector*)
    {
        std::cout << "Assemble rhs not implemented "
                     "for the class IncompNSFEM"
                  << std::endl;
        abort();
    }

    virtual void update_rhs
    (double, double,
     Vector&, BlockVector*)
    {
        std::cout << "Assemble rhs not implemented "
                     "for the class IncompNSFEM"
                  << std::endl;
        abort();
    }

    //! Updates boundary velocity
    void update_vBdry(double t=0);

    //! Applies essential velocity boundary conditions
    void apply_vBdry(Vector&) const;

    //! Applies boundary conditions to SparseMatrix
    void apply_BCs(SparseMatrix&) const;

    //! Used in Imex BDF2
    //! Initializes discretisation of first step
    virtual void init_firstStep(const double) {}

    //! Used in Imex BDF2
    //! Solves first-step
    virtual void solve_firstStep(Solver *,
                                 double,
                                 BlockVector*,
                                 BlockVector*) {}

    //! Re-initializes system operator and block 00
    //! at the end of first step
    virtual void partial_reinit_after_firstStep(double) {}

    //! Sets old solution
    virtual void update_oldSol(BlockVector&) {}

protected:
    //! Updates boundary conditions
    virtual void update_BCs(double t=0)
    {
        std::cout << "Set BCs not implemented "
                     "for the class IncompNSFEM"
                  << std::endl;
        abort();
    }

    //! Assembles source
    void assemble_source(const double, Vector&);

    //! Assembles boundary
    void assemble_bdry(const double, Vector&);

    //! Applies boundary conditions to Vector
    void apply_BCs(Vector&) const;

    //! Applies boundary conditions to BlockVector
    void apply_BCs(BlockVector&) const;
    
    //! Resets matrices that are updated at each time step
    virtual void reset_matrices ()
    {
        std::cout << "Reset matrices not implemented "
                     "for the class IncompNSFEM"
                  << std::endl;
        abort();
    }

    //! Resets convection matrix
    void reset_convection()
    {
        if (m_convNflux) {
            delete m_convNflux;
            m_convNflux = nullptr;
        }
    }

    //! Resets system matrix block(0,0)
    void reset_block00Mat()
    {
        if (m_block00Mat) {
            delete m_block00Mat;
            m_block00Mat = nullptr;
        }
    }

    //! Eliminates rows corresponding to essential boundary
    //! dofs from SparseMatrix
    void eliminate_rows(SparseMatrix& A) const {
        for (int i=0; i<m_ess_tdof_list.Size(); i++) {
            A.EliminateRow(m_ess_tdof_list[i],
                           Matrix::DIAG_ZERO);
        }
    }

public:
    //! Returns test case
    inline std::shared_ptr<IncompNSTestCases>
    get_test_case() const {
        return m_testCase;
    }
    
    //! Returns mesh
    inline Mesh* get_mesh() const {
        return m_fespaces[0]->GetMesh(); 
    }

    //! Returns FE collections
    inline Array<FiniteElementCollection*> get_fecs()
    const {
        return m_fecs;
    }
    
    //! Returns FE spaces
    inline Array<FiniteElementSpace*> get_fespaces()
    const {
        return m_fespaces;
    }

    //! Returns mass matrix
    inline SparseMatrix* get_mass_matrix() const {
        return m_mass;
    }
    
    //! Returns divergence matrix
    inline SparseMatrix* get_div_matrix() const {
        return m_div;
    }

    //! Returns divergence matrix
    //! when no BCs are applied
    inline SparseMatrix* get_divNoBCs_matrix() const {
        return m_divNoBCs;
    }

    //! Returns convection matrix
    inline SparseMatrix* get_convec_matrix() const {
        return m_convNflux;
    }

    //! Returns inverse mass matrix
    inline SparseMatrix* get_invMass_matrix() const {
        return m_invMass;
    }
    
    //! Returns system block operator
    inline BlockOperator* get_incompNS_op() const {
        return m_incompNSOp;
    }

    //! Returns preconditioner
    inline Solver* get_incompNS_pr()
    const {
        return m_incompNSPr;
    }
    
    //! Returns essential dofs marker
    inline Array<int> get_ess_bdr_marker() const {
        return m_ess_bdr_marker;
    }

    //! Returns block offsets
    inline Array<int> get_block_offsets() const {
        return m_block_offsets;
    }

    //! Prints information
    inline void print() const {
        std::cout << "Degree in x: "
                  << m_deg << std::endl;
    }

protected:
    const nlohmann::json& m_config;
    
    int m_ndim, m_deg;
    std::shared_ptr<IncompNSTestCases> m_testCase;
    
    Array<FiniteElementCollection*> m_fecs;
    Array<FiniteElementSpace*> m_fespaces;
    Array<int> m_block_offsets;

    Array<int> m_ess_bdr_marker;
    Array<int> m_nat_bdr_marker;
    Array<int> m_ess_tdof_list;
    
    bool m_bool_viscous = false;
    double m_viscosity, m_penalty;
    std::string m_nfluxType;

    bool m_bool_mean_free_pressure = true;

    MyBilinearFormIntegrator *m_convInt = nullptr;
    MyBilinearFormIntegrator *m_numFluxInt = nullptr;

    SparseMatrix *m_mass = nullptr;
    SparseMatrix *m_div = nullptr;
    SparseMatrix *m_divT = nullptr;
    SparseMatrix *m_convNflux = nullptr; 
    SparseMatrix *m_diffusion = nullptr;
    SparseMatrix *m_massDiffusion = nullptr;
    SparseMatrix *m_invMass = nullptr;

    bool m_appliedBCs = false;
    SparseMatrix *m_divNoBCs = nullptr;
    SparseMatrix *m_openBdry = nullptr;

    GridFunction *m_vBdry = nullptr;
    IncompNSBdryVelocityCoeff *m_vBdryCoeff = nullptr;
    BlockVector *m_rhsDirichlet = nullptr;

    BlockOperator *m_incompNSOp = nullptr;
    SparseMatrix *m_block00Mat = nullptr;
    SparseMatrix *m_block01Mat = nullptr;

    Solver *m_incompNSPr = nullptr;
    Operator *m_blockPr00 = nullptr;
    Operator *m_blockPr11 = nullptr;
    SparseMatrix *m_matS = nullptr;
    Solver *m_invMatS = nullptr;
    Vector m_block00Diag;

    // for the embedded preconditioner
    bool m_bool_embedded_preconditioner;
    Operator *m_incompNSPrOp = nullptr;
    mymfem::InvPrP1Op *m_invPrP1 = nullptr;
    mymfem::PrP2Op *m_prP2 = nullptr;
    Operator *m_invMatD = nullptr;
};


//! Backward Euler + FEM
class IncompNSBackwardEulerFEM : public IncompNSFEM
{
public:
    //! Constructors
    IncompNSBackwardEulerFEM (const nlohmann::json&);

    IncompNSBackwardEulerFEM
    (const nlohmann::json&,
     std::shared_ptr<IncompNSTestCases>&,
     bool bool_embedded_preconditioner=false);

    //! Destructor
    ~IncompNSBackwardEulerFEM () override {}

    //! Initializes discretisation
    void init(const double) override;

    //! Updates system
    void update_system(const double,
                       const double,
                       GridFunction *) override;

    //! Initializes system operator
    void init_system_op() override;

    //! Updates system operator
    void update_system_op() override;

    //! Initializes system operator with
    //! embedded preconditioner
    void init_preconditioned_system_op() override;

    //! Updates system operator with
    //! embedded preconditioner
    void update_preconditioned_system_op() override;

    //! Solves system operator with
    //! embedded preconditioner
    void preconditioned_solve
    (Solver *, const BlockVector&, BlockVector&)
    const override;

    //! Updates right-hand side
    void update_rhs
    (double, double,
     BlockVector*, BlockVector*) override;

protected:
    //! Updates boundary conditions
    void update_BCs(double t=0) override;

    //! Resets convection and system matrix block(0,0)
    void reset_matrices () override
    {
        reset_convection();
        reset_block00Mat();
    }
};


//! Semi-implicit/explicit Euler + FEM
class IncompNSImexEulerFEM : public IncompNSFEM
{
public:
    //! Constructors
    IncompNSImexEulerFEM (const nlohmann::json&);

    IncompNSImexEulerFEM
    (const nlohmann::json&,
     std::shared_ptr<IncompNSTestCases>&);

    //! Destructor
    ~IncompNSImexEulerFEM () override
    {
        if (m_block00MatNoBCs) { delete m_block00MatNoBCs; }
    }

    //! Updates system
    void update_system(const double,
                       const double,
                       GridFunction *) override;

    //! Initializes system operator
    void init_system_op() override;

    //! Updates system operator
    void update_system_op() override;

    //! Updates right-hand side
    void update_rhs
    (double, double,
     BlockVector*, BlockVector*) override;

protected:
    //! Updates boundary conditions
    void update_BCs(double t=0) override;

    //! Resets convection matrix
    void reset_matrices () override
    {
        reset_convection();
    }

private:
    SparseMatrix *m_block00MatNoBCs = nullptr;

    Vector m_buf;
    mymfem::ApplyMFConvNFluxOperator m_applyMFConvNFlux;
};


//! Semi-implicit/explicit BDF2 + FEM
class IncompNSImexBdf2FEM : public IncompNSFEM
{
public:
    //! Constructors
    IncompNSImexBdf2FEM (const nlohmann::json&);

    IncompNSImexBdf2FEM
    (const nlohmann::json&,
     std::shared_ptr<IncompNSTestCases>&);

    //! Destructor
    ~IncompNSImexBdf2FEM () override
    {
        if (m_block00MatNoBCs) { delete m_block00MatNoBCs; }
    }

    //! Initializes discretisation for first step
    void init_firstStep(const double) override;

    //! Solves first-step using Imex RK2
    void solve_firstStep(Solver *,
                         double,
                         BlockVector*,
                         BlockVector*) override;

    //! Re-initializes system operator and block 00
    //! at the end of first step
    void partial_reinit_after_firstStep(double) override;

    //! Initializes discretisation
    void init(const double) override;

    //! Initial assembly of system
    void init_assemble_system(const double) override;

    //! Updates system
    void update_system(const double,
                       const double,
                       GridFunction *) override;

    //! Initializes system operator
    void init_system_op() override;

    //! Updates system operator
    void update_system_op() override;

    //! Updates right-hand side
    void update_rhs
    (double, double,
     BlockVector*, BlockVector*) override;

    //! Update old solution
    void update_oldSol(BlockVector& U) override {
        m_Vnm1.SetSize(U.GetBlock(0).Size());
        m_Vnm1 = U.GetBlock(0);
    }

protected:
    //! Updates boundary conditions
    void update_BCs(double t=0) override;

    //! Resets convection matrix
    void reset_matrices () override
    {
        reset_convection();
    }

private:
    //! Initial assembly of system for first step
    void init_firstStep_assemble_system (const double);

private:
    SparseMatrix *m_block00MatNoBCs = nullptr;

    Vector m_Vnm1, m_buf;
    mymfem::ApplyMFConvNFluxOperator m_applyMFConvNFlux;

    // SBDF2 coefficients
    double m_cs = 2./3;
    double m_a0 = 4./3;
    double m_a1 = -1./3;
    double m_b0 = 2;
    double m_b1 = -1;

    // Imex RK2 coefficients
    double m_gamma = 0.5*(2 - std::sqrt(2));
    double m_delta = -2*std::sqrt(2)/3;
};


//! BDF1/RK2 + FEM
class IncompNSBdf1Rk2FEM : public IncompNSFEM
{
public:
    //! Constructors
    IncompNSBdf1Rk2FEM (const nlohmann::json&);

    IncompNSBdf1Rk2FEM
    (const nlohmann::json&,
     std::shared_ptr<IncompNSTestCases>&);

    //! Destructor
    ~IncompNSBdf1Rk2FEM () override
    {
        if (m_block00MatNoBCs) { delete m_block00MatNoBCs; }
    }

    //! Updates system
    void update_system(const double,
                       const double,
                       GridFunction *) override;

    //! Initializes system operator
    void init_system_op() override;

    //! Updates system operator
    void update_system_op() override;

    //! Updates right-hand side
    void update_rhs
    (double, double,
     Vector&, BlockVector*) override;

protected:
    //! Updates boundary conditions
    void update_BCs(double t=0) override;

    //! Resets convection matrix
    void reset_matrices () override
    {
        reset_convection();
    }

private:
    SparseMatrix *m_block00MatNoBCs = nullptr;
};

#endif /// INCOMPNS_DISCRETISATION_HPP
