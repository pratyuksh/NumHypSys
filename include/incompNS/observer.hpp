#ifndef INCOMPNS_OBSERVER_HPP
#define INCOMPNS_OBSERVER_HPP

#include "mfem.hpp"
using namespace mfem;

#include "../core/config.hpp"
#include "../core/base_observer.hpp"

#include "../mymfem/utilities.hpp"

#include "test_cases.hpp"
#include "coefficients.hpp"
#include "discretisation.hpp"
#include "utilities.hpp"


//! Observer for Incompressible Navier-Stokes
class IncompNSObserver : public BaseObserver
{
public:
    //! Constructors
    IncompNSObserver () : BaseObserver () {}

    IncompNSObserver (const nlohmann::json&, int);

    IncompNSObserver
    (const nlohmann::json&,
     std::shared_ptr<IncompNSTestCases>&, int);
    
    //! Computes L2 error for velocity and pressure
    std::tuple <double, double, double, double> 
    eval_error_L2 (const double t,
                   std::shared_ptr<GridFunction>&,
                   std::shared_ptr<GridFunction>&) const;

    //! Initializes vorticity and velocity component functions
    void init(std::shared_ptr<IncompNSFEM>&);

    //! Visualizes velocity components
    void visualize_velocities(std::shared_ptr<GridFunction>&);

    //! Visualizes vorticity
    void visualize_vorticity(std::shared_ptr<GridFunction>&);

    //! Dumps solution
    void dump_sol (std::shared_ptr<GridFunction>&,
                   std::shared_ptr<GridFunction>&) const;
    
private:
    bool m_bool_error;
    std::string m_solName_prefix;
    std::string m_solName_suffix;

    std::shared_ptr<IncompNSTestCases> m_testCase;
    std::shared_ptr<IncompNSFEM> m_discr;

    FiniteElementSpace *m_sfes = nullptr;
    FiniteElementSpace *m_vfes = nullptr;

    std::unique_ptr<VelocityFunction> m_vel;
    std::unique_ptr<VorticityFunction> m_vort;
};


#endif /// INCOMPNS_OBSERVER_HPP
