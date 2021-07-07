#include "../../include/incompNS/test_cases_factory.hpp"
#include <fmt/format.h>


//! Makes different test cases
std::shared_ptr<IncompNSTestCases>
make_incompNS_test_case(const nlohmann::json& config)
{
    const std::string problem_type
            = config["problem_type"];

    if (problem_type == "tgv_test1") {
        return std::make_shared
                <IncompNSTestCase
                <TaylorGreenVortex_Test1>> (config);
    }
    else if (problem_type == "tgv_test2") {
        return std::make_shared
                <IncompNSTestCase
                <TaylorGreenVortex_Test2>> (config);
    }
    else if (problem_type == "tgv_test3") {
        return std::make_shared
                <IncompNSTestCase
                <TaylorGreenVortex_Test3>> (config);
    }
    else if (problem_type == "tgv_test4") {
        return std::make_shared
                <IncompNSTestCase
                <TaylorGreenVortex_Test4>> (config);
    }
    else if (problem_type == "dsl") {
        return std::make_shared
                <IncompNSTestCase
                <DoubleShearLayer>> (config);
    }
    else if (problem_type == "ldc") {
        return std::make_shared
                <IncompNSTestCase
                <LidDrivenCavity>> (config);
    }
    else if (problem_type == "kf") {
        return std::make_shared
                <IncompNSTestCase
                <KovasznayFlow>> (config);
    }
    else if (problem_type == "cf") {
        return std::make_shared
                <IncompNSTestCase
                <ChannelFlow>> (config);
    }
    else if (problem_type == "uq_test") {
        return std::make_shared
                <IncompNSTestCase
                <UqTestModel>> (config);
    }
    else if (problem_type == "uq_svs") {
        return std::make_shared
                <IncompNSTestCase
                <UQSmoothVortexSheet>> (config);
    }
    else if (problem_type == "uq_ldc") {
        return std::make_shared
                <IncompNSTestCase
                <UQLidDrivenCavity>> (config);
    }
    else if (problem_type == "uq_cf") {
        return std::make_shared
                <IncompNSTestCase
                <UQChannelFlow>> (config);
    }

    throw std::runtime_error(fmt::format(
        "Unknown problem type for "
        "incompressible Navier-Stokes equations. "
        "[{}]", problem_type));
}
