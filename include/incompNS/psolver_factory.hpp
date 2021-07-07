#ifndef INCOMPNS_PSOLVER_FACTORY_HPP
#define INCOMPNS_PSOLVER_FACTORY_HPP

#include "psolver.hpp"


//! Makes solver for different time integrators
IncompNSParSolver* make_solver
(MPI_Comm& comm, const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir,
 int lx, int Nt);
 
 
#endif /// INCOMPNS_SOLVER_FACTORY_HPP
