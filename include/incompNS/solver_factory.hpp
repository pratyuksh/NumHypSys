#ifndef INCOMPNS_SOLVER_FACTORY_HPP
#define INCOMPNS_SOLVER_FACTORY_HPP

#include "solver.hpp"


//! Makes solver for different time integrators
IncompNSSolver* make_solver
(const nlohmann::json config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir,
 int lx, int Nt);


#endif /// INCOMPNS_SOLVER_FACTORY_HPP
