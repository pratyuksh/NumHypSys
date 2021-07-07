#ifndef INCOMPNS_UTILITIES_HPP
#define INCOMPNS_UTILITIES_HPP

#include "../mymfem/utilities.hpp"
#include "../core/config.hpp"
#include <memory>

#include "mfem.hpp"
using namespace mfem;


// Divergence measurement and cleaning routines

//! Measures divergence
double measure_divergence(SparseMatrix* div,
                          GridFunction* v);

double measure_divergence(SparseMatrix* div,
                          Vector& V);


//! Weak Divergence Integrator
class WeakDivergenceLfIntegrator
        : public LinearFormIntegrator
{
public:
    //! Constructor
    WeakDivergenceLfIntegrator (GridFunction* gf)
    {m_gf = gf;}

    //! Assembles the linear form for given element
    void AssembleRHSElementVect (const FiniteElement &el,
                                 ElementTransformation &Tr,
                                 Vector &elvect) override;

private:
#ifndef MFEM_THREAD_SAFE
    mutable Vector shape;
#endif
    GridFunction* m_gf = nullptr;
};


//! Velocity divergence cleaner
class DivergenceFreeVelocity
{
public:
    //! Constructor
    DivergenceFreeVelocity (const nlohmann::json& config,
                            const std::shared_ptr<Mesh>& mesh);

    //! Destructor
    ~DivergenceFreeVelocity ();

    //! Sets components
    void set() const;

    //! Cleans divergence
    void operator()(GridFunction *) const;

private:
    int m_deg;

    FiniteElementSpace *m_h1fes = nullptr;
    FiniteElementSpace *m_rtfes = nullptr;

    CGSolver* m_cgsolver_stiff = nullptr;
    CGSolver* m_cgsolver_mass = nullptr;
    Solver* m_cgpr_stiff = nullptr;

    Array<int> m_ess_bdr_marker_h1fes;
    Array<int> m_ess_tdof_list_h1fes;

    Array<int> m_ess_bdr_marker_rtfes;
    Array<int> m_ess_tdof_list_rtfes;

    mutable SparseMatrix *m_mass_rt = nullptr;
    mutable SparseMatrix *m_stiff_h1 = nullptr;
    mutable SparseMatrix *m_grad = nullptr;

    mutable Vector *m_rhs = nullptr;
};

#endif
