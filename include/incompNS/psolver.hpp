#ifndef INCOMPNS_PSOLVER_HPP
#define INCOMPNS_PSOLVER_HPP

#include "mfem.hpp"
using namespace mfem;

#include <Eigen/Core>

#include "../core/config.hpp"

#include "../mymfem/utilities.hpp"

#include "test_cases_factory.hpp"
#include "coefficients.hpp"
#include "pdiscretisation.hpp"
#include "pobserver.hpp"


class IncompNSParSolver
{
public:
    //! Constructors
    explicit IncompNSParSolver
    (MPI_Comm comm,
     const nlohmann::json config);

    explicit IncompNSParSolver
    (MPI_Comm comm,
     const nlohmann::json config,
     std::shared_ptr<IncompNSTestCases>& testCase);

    explicit IncompNSParSolver
    (MPI_Comm comm,
     const nlohmann::json config,
     std::shared_ptr<IncompNSTestCases>& testCase,
     const std::string mesh_dir,
     const int lx, const int Nt);

    //! Destructor
    virtual ~IncompNSParSolver ();

    //! Sets mesh and observer
    void set(const std::string mesh_dir, const int lx);

    //! Initializes and runs solver
    void operator() (std::unique_ptr<BlockVector>& U);

    //! Initializes and runs solver
    //! Computes the solution error if needed
    std::pair<double, Eigen::VectorXd> operator() (void);
    
    //! Initializes solver
    virtual void init ()
    {
        std::cout << "Init not implemented "
                     "for the class IncompNSParSolver"
                  << std::endl;
        abort();
    }

    //! Projects initial conditions
    void project_init_conditions
    (std::unique_ptr<BlockVector>& U,
     std::shared_ptr<ParGridFunction>& v,
     std::shared_ptr<ParGridFunction>& p) const;

    //! Runs initialized solver
    void run (std::unique_ptr<BlockVector>&,
              int sampleId=0) const;

    //! Deletes memory
    void clear ();
    
    //! Initializes linear system solver
    void init_linear_solver() const;

    //! Updates linear system solver
    void update_linear_solver() const;

    //! Initializes preconditioner
    void init_preconditioner () const;

    //! Updates preconditioner
    void update_preconditioner (const double dt) const;

    //! Solves one time step
    virtual void solve_one_step
    (const int, std::shared_ptr<ParGridFunction>&,
     BlockVector *, BlockVector *) const
    {
        std::cout << "Solve one step not implemented "
                     "for the class IncompNSParSolver"
                  << std::endl;
        abort();
    }

    //! Computes error for velocity and pressure
    Eigen::VectorXd compute_error
    (double t,
     std::shared_ptr<ParGridFunction> v,
     std::shared_ptr<ParGridFunction> p) const;

    //! Cleans divergence, if needed
    void clean_divergence
    (std::shared_ptr<ParGridFunction> v) const;

    //! Computes the time-step according to the
    //! cfl conditions.
    //! Assumes that all mesh elements have same geometry
    double compute_time_step
    (std::shared_ptr<ParGridFunction>& v) const;

    //! Visualizes solution
    void visualize
    (std::shared_ptr<ParGridFunction> v,
     std::shared_ptr<ParGridFunction> p) const;

    //! Returns test case
    inline std::shared_ptr<IncompNSTestCases>
    get_test_case() const {
        return m_testCase;
    }

    //! Returns mesh
    inline std::shared_ptr<ParMesh>
    get_mesh() const {
        return m_pmesh;
    }

    //! Returns discretisation
    inline std::shared_ptr<IncompNSParFEM>
    get_discr() const {
        return m_discr;
    }

    //! Returns observer
    inline std::shared_ptr<IncompNSParObserver>
    get_observer() const {
        return m_observer;
    }

    //! Evluates minimum and maximum mesh size
    inline std::pair<double, double>
    eval_hMinMax() const
    {
        double h_min, h_max, kappa_min, kappa_max;
        m_pmesh->GetCharacteristics(h_min, h_max,
                                    kappa_min, kappa_max);
        return {std::move(h_min), std::move(h_max)};
    }

    //! Returns minimum mesh size
    inline double
    get_hMin() const {
        return m_hMin;
    }

    //! Returns maximum mesh size
    inline double
    get_hMax() const {
        return m_hMax;
    }

protected:
    int m_myrank;
    MPI_Comm m_comm;

    const nlohmann::json m_config;
    int m_deg;
    
    int m_Nt;
    double m_tEnd;

    bool m_bool_read_mesh_from_file;
    int m_num_refinements;

    bool m_bool_error;
    bool m_bool_cleanDivg;

    std::string m_mesh_format;
    std::string m_mesh_elem_type;
    std::string m_mesh_dir;

    Array<int> m_block_offsets;
    Array<int> m_block_trueOffsets;

    std::shared_ptr<IncompNSTestCases> m_testCase;
    std::shared_ptr<ParMesh> m_pmesh;
    std::shared_ptr<IncompNSParFEM> m_discr;
    std::shared_ptr<IncompNSParObserver> m_observer;
    
    mutable HypreParMatrix *m_incompNSMat = nullptr;
    mutable BlockOperator *m_incompNSOp = nullptr;
    mutable Solver *m_incompNSPr = nullptr;

    mutable Solver *m_precond = nullptr;
    mutable Solver *m_solver = nullptr;

    bool m_bool_embedded_preconditioner;
    bool m_bool_mean_free_pressure;

    double m_hMin, m_hMax;
    double m_cfl_number = 0.3;
};


//! Backward Euler + FEM
class IncompNSBackwardEulerParSolver
        : public IncompNSParSolver
{
public:
    //! Constructor
    explicit IncompNSBackwardEulerParSolver
    (MPI_Comm comm,
     const nlohmann::json config,
     std::shared_ptr<IncompNSTestCases>& testCase,
     const std::string mesh_dir,
     const int lx, const int Nt);

    //! Destructor
    ~IncompNSBackwardEulerParSolver () override {}

    //! Initializes solver
    void init () override;

    //! Solves one time step
    void solve_one_step
    (const int, std::shared_ptr<ParGridFunction>&,
     BlockVector *, BlockVector *) const override;

private:
    //! Updates solver components
    void update (const double, const double,
                 std::shared_ptr<ParGridFunction>&,
                 BlockVector *, BlockVector *) const;

    //! Runs the appropriate linear system solvers
    void solve (BlockVector *, BlockVector *) const;
};


//! Semi-implicit/explicit Euler + FEM
class IncompNSImexEulerParSolver
        : public IncompNSParSolver
{
public:
    //! Constructor
    explicit IncompNSImexEulerParSolver
    (MPI_Comm comm,
     const nlohmann::json config,
     std::shared_ptr<IncompNSTestCases>& testCase,
     const std::string mesh_dir,
     const int lx, const int Nt);

    //! Destructor
    ~IncompNSImexEulerParSolver () override {}

    //! Initializes solver
    void init () override;

    //! Solves one time step
    void solve_one_step
    (const int, std::shared_ptr<ParGridFunction>&,
     BlockVector *, BlockVector *) const override;
};


//! Semi-implicit/explicit BDF2 + FEM
class IncompNSImexBdf2ParSolver
        : public IncompNSParSolver
{
public:
    //! Constructor
    explicit IncompNSImexBdf2ParSolver
    (MPI_Comm comm,
     const nlohmann::json config,
     std::shared_ptr<IncompNSTestCases>& testCase,
     const std::string mesh_dir,
     const int lx, const int Nt);

    //! Destructor
    ~IncompNSImexBdf2ParSolver () override {}

    //! Initializes solver
    void init () override;

    //! Solves one time step
    void solve_one_step
    (const int, std::shared_ptr<ParGridFunction>&,
     BlockVector *, BlockVector *) const override;
};


//! BDF1/RK2 + FEM
class IncompNSBdf1Rk2ParSolver
        : public IncompNSParSolver
{
public:
    //! Constructor
    explicit IncompNSBdf1Rk2ParSolver
    (MPI_Comm comm,
     const nlohmann::json config,
     std::shared_ptr<IncompNSTestCases>& testCase,
     const std::string mesh_dir,
     const int lx, const int Nt);

    //! Destructor
    ~IncompNSBdf1Rk2ParSolver () override {}

    //! Initializes solver
    void init () override;

    //! Solves one time step
    void solve_one_step
    (const int, std::shared_ptr<ParGridFunction>&,
     BlockVector *, BlockVector *) const override;

private:
    //! Runs propagation
    void propagate(double tn, Vector& Vs) const;

    //! Runs one time step of propagation
    //! Explicit RK2, mid-point method
    //! Applies matrix-free convection operator
    void propagate_one_step(double s,
                            double ds,
                            Vector& V) const;

    //! Runs one time step of propagation
    //! Explicit RK2, mid-point method
    //! Slow version assembles the convection matrix
    void propagate_one_step_slow(double s,
                                 double ds,
                                 Vector& V) const;

    //! Extrapolates solution from previous time steps
    void extrapolate(double ds, Vector& Vs) const;

private:
    double m_dt;
    HypreParMatrix *m_invMass = nullptr;

    mutable std::shared_ptr<ParGridFunction> m_vn;
    mutable std::shared_ptr<ParGridFunction> m_vs;

    mutable std::unique_ptr<Vector> m_Vs;
    mutable std::unique_ptr<Vector> m_buf;
    mutable std::unique_ptr<Vector> m_km;
};


#endif /// INCOMPNS_PSOLVER_HPP
