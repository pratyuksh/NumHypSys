#ifndef INCOMPNS_PUTILITIES_HPP
#define INCOMPNS_PUTILITIES_HPP

#include "../mymfem/putilities.hpp"
#include "../core/config.hpp"
#include <memory>

#include "mfem.hpp"
using namespace mfem;


// Divergence measurement and cleaning routines

//! Measures divergence
double measure_divergence(HypreParMatrix* div,
                          ParGridFunction* v);


//! Weak Divergence Integrator
class ParWeakDivergenceLfIntegrator
        : public LinearFormIntegrator
{
public:
    //! Constructor
    ParWeakDivergenceLfIntegrator (ParGridFunction* gf)
    {m_gf = gf;}

    //! Assembles the linear form for given element
    void AssembleRHSElementVect (const FiniteElement &el,
                                 ElementTransformation &Tr,
                                 Vector &elvect) override;

private:
#ifndef MFEM_THREAD_SAFE
    mutable Vector shape;
#endif
    ParGridFunction* m_gf = nullptr;
};


//! Velocity divergence cleaner
class ParDivergenceFreeVelocity
{
public:
    //! Constructor
    ParDivergenceFreeVelocity
    (const nlohmann::json& config,
     const std::shared_ptr<ParMesh>& mesh);

    //! Destructor
    ~ParDivergenceFreeVelocity ();

    //! Sets components
    void set() const;

    //! Cleans divergence
    void operator()(ParGridFunction *) const;

private:
    int m_deg;

    ParFiniteElementSpace *m_h1fes = nullptr;
    ParFiniteElementSpace *m_rtfes = nullptr;

    HyprePCG* m_cgsolver_stiff = nullptr;
    HyprePCG* m_cgsolver_mass = nullptr;
    HypreSolver* m_cgpr_stiff = nullptr;

    Array<int> m_ess_bdr_marker_h1fes;
    Array<int> m_ess_tdof_list_h1fes;

    Array<int> m_ess_bdr_marker_rtfes;
    Array<int> m_ess_tdof_list_rtfes;

    mutable HypreParMatrix *m_mass_rt = nullptr;
    mutable HypreParMatrix *m_stiff_h1 = nullptr;
    mutable HypreParMatrix *m_grad = nullptr;

    mutable Vector *m_rhs = nullptr;
};

#endif
