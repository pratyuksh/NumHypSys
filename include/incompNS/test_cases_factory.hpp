#ifndef INCOMPNS_TEST_CASES_FACTORY_HPP
#define INCOMPNS_TEST_CASES_FACTORY_HPP

#include "test_cases.hpp"


//! Makes different test cases
std::shared_ptr<IncompNSTestCases>
make_incompNS_test_case(const nlohmann::json& config);


#endif /// INCOMPNS_TEST_CASES_FACTORY_HPP
