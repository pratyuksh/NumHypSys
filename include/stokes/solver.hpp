#ifndef STOKES_SOLVER_HPP
#define STOKES_SOLVER_HPP

#include "mfem.hpp"
using namespace mfem;

#include <Eigen/Core>

#include "../core/config.hpp"
#include "../mymfem/utilities.hpp"

#include "test_cases.hpp"
#include "coefficients.hpp"
#include "discretisation.hpp"
#include "observer.hpp"


class StokesSolver
{
public:
    StokesSolver (const nlohmann::json& config,
                  std::string mesh_dir,
                  const int lx);

    ~ StokesSolver ();

    std::pair<double, Eigen::VectorXd> operator()(void);
    
    void init ();
    
    void solve (BlockVector *U, BlockVector* B) const;
    
    void set_linear_solver() const;
    
    void set_preconditioner () const;

private:
    const nlohmann::json& m_config;
    
    std::string m_mesh_format;
    std::string m_mesh_elem_type;

    std::shared_ptr<Mesh> m_mesh;
    std::unique_ptr<StokesFEM> m_discr;
    std::unique_ptr<StokesObserver> m_observer;

    bool m_bool_mean_free_pressure = true;
    bool m_bool_error;
    
    Array<int> m_block_offsets;
    mutable Solver *m_solver = nullptr;

    mutable BlockOperator *m_stokesOp = nullptr;
    mutable BlockDiagonalPreconditioner *m_stokesPr = nullptr;
    mutable SparseMatrix *m_stokesMat = nullptr;
};


#endif /// STOKES_SOLVER_HPP
