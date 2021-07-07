#ifndef INCOMPNS_SOLVER_HPP
#define INCOMPNS_SOLVER_HPP

#include "mfem.hpp"
using namespace mfem;

#include <Eigen/Core>

#include "../core/config.hpp"
#include "../mymfem/utilities.hpp"

#include "test_cases_factory.hpp"
#include "coefficients.hpp"
#include "discretisation.hpp"
#include "observer.hpp"


class IncompNSSolver
{
public:
    //! Constructors
    IncompNSSolver (const nlohmann::json config);

    IncompNSSolver
    (const nlohmann::json config,
     std::shared_ptr<IncompNSTestCases>& testCase);

    IncompNSSolver
    (const nlohmann::json config,
     std::shared_ptr<IncompNSTestCases>& testCase,
     const std::string mesh_dir,
     const int lx, const int Nt);

    //! Destructor
    virtual ~IncompNSSolver ();

    //! Sets mesh and observer
    void set(const std::string mesh_dir, const int lx);

    //! Initializes and runs solver
    void operator()(std::unique_ptr<BlockVector>& U);

    //! Initializes and runs solver
    //! Computes the solution error if needed
    std::pair<double, Eigen::VectorXd> operator()(void);
    
    //! Initializes solver
    virtual void init ()
    {
        std::cout << "Init not implemented "
                     "for the class IncompNSSolver"
                  << std::endl;
        abort();
    }

    //! Runs initialized solver
    void run (std::unique_ptr<BlockVector>& U) const;

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
    virtual void solve_one_step (const int step_num,
                                 GridFunction *u,
                                 BlockVector *U,
                                 BlockVector *B) const
    {
        std::cout << "Solve one step not implemented "
                     "for the class IncompNSSolver"
                  << std::endl;
        abort();
    }

    //! Computes error for velocity and pressure
    Eigen::VectorXd compute_error
    (double t,
     std::shared_ptr<GridFunction> v,
     std::shared_ptr<GridFunction> p) const;

    //! Cleans divergence, if needed
    void clean_divergence
    (std::shared_ptr<GridFunction> v) const;

    //! Computes the time-step according to the
    //! cfl conditions.
    //! Assumes that all mesh elements have same geometry
    double compute_time_step
    (std::shared_ptr<GridFunction>& v) const;

    //! Visualizes solution
    void visualize(std::shared_ptr<GridFunction> v,
                   std::shared_ptr<GridFunction> p) const;

    //! Returns test case
    inline std::shared_ptr<IncompNSTestCases>
    get_test_case() const {
        return m_testCase;
    }

    //! Returns mesh
    inline std::shared_ptr<Mesh>
    get_mesh() const {
        return m_mesh;
    }

    //! Returns discretisation
    inline std::shared_ptr<IncompNSFEM>
    get_discr() const {
        return m_discr;
    }

    //! Returns observer
    inline std::shared_ptr<IncompNSObserver>
    get_observer() const {
        return m_observer;
    }

    //! Evluates minimum and maximum mesh size
    inline std::pair<double, double>
    eval_hMinMax() const
    {
        double h_min, h_max, kappa_min, kappa_max;
        m_mesh->GetCharacteristics(h_min, h_max,
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
    Array<int> m_ess_bdr_marker;

    std::shared_ptr<IncompNSTestCases> m_testCase;
    std::shared_ptr<Mesh> m_mesh;
    std::shared_ptr<IncompNSFEM> m_discr;
    std::shared_ptr<IncompNSObserver> m_observer;

    mutable BlockOperator *m_incompNSOp = nullptr;
    mutable Solver *m_incompNSPr = nullptr;

    mutable Solver *m_solver = nullptr;
    mutable Solver *m_precond = nullptr;

    bool m_bool_embedded_preconditioner;
    bool m_bool_mean_free_pressure;

    double m_hMin, m_hMax;
    double m_cfl_number = 0.3;
};


//! Backward Euler + FEM
class IncompNSBackwardEulerSolver : public IncompNSSolver
{
public:
    //! Constructor
    IncompNSBackwardEulerSolver
    (const nlohmann::json config,
     std::shared_ptr<IncompNSTestCases>& testCase,
     const std::string mesh_dir,
     const int lx, const int Nt);

    //! Destructor
    ~IncompNSBackwardEulerSolver () override {}

    //! Initializes solver
    void init () override;

    //! Solves one time step
    void solve_one_step (const int step_num,
                         GridFunction *v,
                         BlockVector *U,
                         BlockVector *B) const override;

private:
    //! Updates solver components
    void update (const double, const double,
                 GridFunction *,
                 BlockVector *, BlockVector *) const;

    //! Runs the appropriate linear system solvers
    void solve (BlockVector *, BlockVector *) const;

    double m_dt;
};


//! Semi-implicit/explicit Euler + FEM
class IncompNSImexEulerSolver : public IncompNSSolver
{
public:
    //! Constructor
    IncompNSImexEulerSolver
    (const nlohmann::json config,
     std::shared_ptr<IncompNSTestCases>& testCase,
     const std::string mesh_dir,
     const int lx, const int Nt);

    //! Destructor
    ~IncompNSImexEulerSolver () override {}

    //! Initializes solver
    void init () override;

    //! Solves one time step
    void solve_one_step (const int step_num,
                         GridFunction *u,
                         BlockVector *U,
                         BlockVector *B) const override;
};

//! Semi-implicit/explicit BDF2 + FEM
class IncompNSImexBdf2Solver : public IncompNSSolver
{
public:
    //! Constructor
    IncompNSImexBdf2Solver
    (const nlohmann::json config,
     std::shared_ptr<IncompNSTestCases>& testCase,
     const std::string mesh_dir,
     const int lx, const int Nt);

    //! Destructor
    ~IncompNSImexBdf2Solver () override {}

    //! Initializes solver
    void init () override;

    //! Solves one time step
    void solve_one_step (const int step_num,
                         GridFunction *u,
                         BlockVector *U,
                         BlockVector *B) const override;
};


//! BDF1/RK2 + FEM
class IncompNSBdf1Rk2Solver : public IncompNSSolver
{
public:
    //! Constructor
    IncompNSBdf1Rk2Solver
    (const nlohmann::json config,
     std::shared_ptr<IncompNSTestCases>& testCase,
     const std::string mesh_dir,
     const int lx, const int Nt);

    //! Destructor
    ~IncompNSBdf1Rk2Solver () override {}

    //! Initializes solver
    void init () override;

    //! Solves one time step
    void solve_one_step (const int step_num,
                         GridFunction *u,
                         BlockVector *U,
                         BlockVector *B) const override;

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

    mutable Vector m_bufVn;

    BilinearForm* m_invMassBf = nullptr;
    SparseMatrix *m_invMass = nullptr;
};


#endif /// INCOMPNS_SOLVER_HPP
