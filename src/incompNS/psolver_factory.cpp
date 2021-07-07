#include "../../include/incompNS/psolver_factory.hpp"
#include <fmt/format.h>


//! Makes solver for different time integrators
IncompNSParSolver* make_solver
(MPI_Comm& comm, const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase,
 const std::string mesh_dir,
 int lx, int Nt)
{
    std::string time_integrator = "backward_euler";
    if (config.contains("time_integrator")) {
        time_integrator = config["time_integrator"];
    }

    if (time_integrator == "backward_euler") {
        return new IncompNSBackwardEulerParSolver
                (comm, config, testCase, mesh_dir, lx, Nt);
    }
    else if (time_integrator == "imex_euler") {
        return new IncompNSImexEulerParSolver
                (comm, config, testCase, mesh_dir, lx, Nt);
    }
    else if (time_integrator == "imex_bdf2") {
        return new IncompNSImexBdf2ParSolver
                (comm, config, testCase, mesh_dir, lx, Nt);
    }
    else if (time_integrator == "bdf1_rk2") {
        return new IncompNSBdf1Rk2ParSolver
                (comm, config, testCase, mesh_dir, lx, Nt);
    }

    throw std::runtime_error(fmt::format(
        "Unknown time-integrator for "
        "incompressible Navier-Stokes equations. "
        "[{}]", time_integrator));
}

// End of file
