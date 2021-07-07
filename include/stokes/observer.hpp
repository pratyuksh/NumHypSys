#ifndef STOKES_OBSERVER_HPP
#define STOKES_OBSERVER_HPP

#include "mfem.hpp"
using namespace mfem;

#include "../core/config.hpp"
#include "../core/base_observer.hpp"

#include "../mymfem/utilities.hpp"

#include "test_cases.hpp"
#include "coefficients.hpp"
#include "discretisation.hpp"


// Observer for Stokes system
class StokesObserver : public BaseObserver
{
public:
    StokesObserver () : BaseObserver () {}

    StokesObserver (const nlohmann::json&, int);
    
    void dump_sol (std::shared_ptr<GridFunction>&,
                   std::shared_ptr<GridFunction>&) const;
    
    std::tuple <double, double, double, double> 
    eval_error_L2 (std::shared_ptr<StokesTestCases>&,
                   std::shared_ptr<GridFunction>&,
                   std::shared_ptr<GridFunction>&) const;
    
private:
    bool m_bool_error;
    std::string m_solName_prefix;
    std::string m_solName_suffix;
};


#endif /// STOKES_OBSERVER_HPP
