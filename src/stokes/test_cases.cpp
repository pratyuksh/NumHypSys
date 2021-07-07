#include "../../include/stokes/test_cases.hpp"

#include <fmt/format.h>


std::shared_ptr<StokesTestCases>
set_stokes_test_case(const nlohmann::json& config)
{
    const std::string problem_type
            = config["problem_type"];

    if (problem_type == "square_test1") {
        return std::make_shared
                <StokesTestCase<Square_Test1>> (config);
    }
    else if (problem_type == "square_test2") {
        return std::make_shared
                <StokesTestCase<Square_Test2>> (config);
    }
    else if (problem_type == "square_test3") {
        return std::make_shared
                <StokesTestCase<Square_Test3>> (config);
    }

    throw std::runtime_error(fmt::format(
        "Unknown problem type for Stokes equation. "
        "[{}]", problem_type));
}


// Case: Square_Test1
// Homogeneous Dirichlet boundary conditions
double StokesTestCase <Square_Test1>
:: pressure_sol(const Vector& x) const
{
    double pressure = 2*cos(x(0))*cos(x(1));
    return pressure;
}

Vector StokesTestCase <Square_Test1>
:: velocity_sol (const Vector& x) const
{
    Vector v(m_dim);
    v(0) = +sin(x(0))*cos(x(1));
    v(1) = -cos(x(0))*sin(x(1));
    return v;
}

Vector StokesTestCase <Square_Test1>
:: source (const Vector& x) const
{
    Vector f(m_dim);
    f(0) = 0;
    f(1) = -4*cos(x(0))*sin(x(1));
    return f;
}

void StokesTestCase <Square_Test1>
:: set_bdry_dirichlet(Array<int>& bdr_marker) const
{
    bdr_marker = 1;
}


// Case: Square_Test2
// Non-homogeneous Dirichlet boundary conditions
double StokesTestCase <Square_Test2>
:: pressure_sol(const Vector& x) const
{
    double pressure = -2*sin(x(0))*sin(x(1));
    return pressure;
}

Vector StokesTestCase <Square_Test2>
:: velocity_sol (const Vector& x) const
{
    Vector v(m_dim);
    v(0) = -cos(x(0))*sin(x(1));
    v(1) = +sin(x(0))*cos(x(1));
    return v;
}

Vector StokesTestCase <Square_Test2>
:: source (const Vector& x) const
{
    Vector f(m_dim);
    f(0) = -4*cos(x(0))*sin(x(1));
    f(1) = 0;
    return f;
}

void StokesTestCase <Square_Test2>
:: set_bdry_dirichlet(Array<int>& bdr_marker) const
{
    bdr_marker = 1;
}


// Case: Square_Test3
// Inflow Outflow
double StokesTestCase <Square_Test3>
:: pressure_sol(const Vector& x) const
{
    double pressure = -2*x(0)+1;
    return pressure;
}

Vector StokesTestCase <Square_Test3>
:: velocity_sol (const Vector& x) const
{
    Vector v(m_dim);
    v(0) = x(1)*(1 - x(1));
    v(1) = 0;
    return v;
}

Vector StokesTestCase <Square_Test3>
:: source (const Vector& x) const
{
    Vector f(m_dim);
    f = 0.0;
    return f;
}

void StokesTestCase <Square_Test3>
:: set_bdry_dirichlet(Array<int>& bdr_marker) const
{
    bdr_marker = 1;
    bdr_marker[1] = 0;
}

// End of file
