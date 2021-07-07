#ifndef INCOMPNS_POBSERVER_HPP
#define INCOMPNS_POBSERVER_HPP

#include "mfem.hpp"
using namespace mfem;

#include "../core/config.hpp"
#include "../core/perror.hpp"
#include "../core/base_pobserver.hpp"

#include "../mymfem/utilities.hpp"
#include "../mymfem/putilities.hpp"

#include "test_cases.hpp"
#include "coefficients.hpp"
#include "pdiscretisation.hpp"
#include "putilities.hpp"


//! Observer for Incompressible Navier-Stokes
class IncompNSParObserver : public BaseParObserver
{
public:
    //! Constructors
    IncompNSParObserver
    (MPI_Comm, const nlohmann::json&, int);

    IncompNSParObserver
    (MPI_Comm, const nlohmann::json&,
     std::shared_ptr<IncompNSTestCases>&, int);

    //! Computes L2 error for velocity and pressure
    std::tuple <double, double, double, double>
    eval_error_L2 (const double t,
                   std::shared_ptr<ParGridFunction>&,
                   std::shared_ptr<ParGridFunction>&) const;

    //! Initializes vorticity and velocity component functions
    void init(std::shared_ptr<IncompNSParFEM>&);

    //! Visualizes velocity components
    void visualize_velocities
    (std::shared_ptr<ParGridFunction>&);

    //! Visualizes vorticity
    void visualize_vorticity
    (std::shared_ptr<ParGridFunction>&);

    //! Dumps solution
    void dump_sol (std::shared_ptr<ParGridFunction>&, 
                   std::shared_ptr<ParGridFunction>&) const;
    
    //! Dumps solution at given time ids
    void dump_sol (std::shared_ptr<ParGridFunction>&,
                   std::shared_ptr<ParGridFunction>&,
                   int time_step_id) const;

    //! Dumps solution for a given sample at given time ids
    void dump_sol (std::shared_ptr<ParGridFunction>&,
                   std::shared_ptr<ParGridFunction>&,
                   int sampleId, int timeStepId) const;

private:
    bool m_bool_error;
    std::string m_solName_prefix;
    std::string m_solName_suffix;

    std::shared_ptr<IncompNSTestCases> m_testCase;
    std::shared_ptr<IncompNSParFEM> m_discr;

    ParFiniteElementSpace *m_sfes = nullptr;
    ParFiniteElementSpace *m_vfes = nullptr;

    std::unique_ptr<ParVelocityFunction> m_vel;
    std::unique_ptr<ParVorticityFunction> m_vort;

    int m_dump_out_nsteps;
};


#endif /// INCOMPNS_OBSERVER_HPP
