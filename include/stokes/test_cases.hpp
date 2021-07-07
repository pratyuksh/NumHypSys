#ifndef STOKES_TEST_CASES_HPP
#define STOKES_TEST_CASES_HPP

#include "../core/config.hpp"
#include "mfem.hpp"

using namespace mfem;

// Cases
enum {Square_Test1,
      Square_Test2,
      Square_Test3};


// Abstract Base class for the Stokes test cases
class StokesTestCases
{
public:

    virtual ~StokesTestCases() = default;
    
    virtual double pressure_sol(const Vector&) const = 0;

    virtual Vector velocity_sol(const Vector&) const = 0;
    
    virtual Vector bdry_velocity(const Vector&) const = 0;

    virtual Vector source(const Vector&) const = 0;
    
    virtual void set_bdry_dirichlet(Array<int>&) const = 0;
    
    virtual int get_dim() const = 0;
};


std::shared_ptr<StokesTestCases>
set_stokes_test_case(const nlohmann::json&);


// class StokesTestCase, base template
template<int ProblemType>
class StokesTestCase;


// Case: Square_Test1
template<>
class StokesTestCase <Square_Test1>
        : public StokesTestCases
{
public:
    explicit StokesTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2) {
    }

    double pressure_sol(const Vector&) const override;

    Vector velocity_sol(const Vector&) const override;

    Vector source(const Vector&) const override;

    void set_bdry_dirichlet(Array<int>&) const override;

    Vector bdry_velocity(const Vector& x) const override {
        return velocity_sol(x);
    }

    inline int get_dim() const override {
        return m_dim;
    }

private:
    const nlohmann::json& m_config;
    int m_dim;
};


// Case: Square_Test2
template<>
class StokesTestCase <Square_Test2>
        : public StokesTestCases
{
public:
    explicit StokesTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2) {
    }

    double pressure_sol(const Vector&) const override;

    Vector velocity_sol(const Vector&) const override;

    Vector source(const Vector&) const override;

    void set_bdry_dirichlet(Array<int>&) const override;

    Vector bdry_velocity(const Vector& x) const override {
        return velocity_sol(x);
    }

    inline int get_dim() const override {
        return m_dim;
    }

private:
    const nlohmann::json& m_config;
    int m_dim;
};


// Case: Square_Test3
template<>
class StokesTestCase <Square_Test3>
        : public StokesTestCases
{
public:
    explicit StokesTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2) {
    }

    double pressure_sol(const Vector&) const override;

    Vector velocity_sol(const Vector&) const override;

    Vector source(const Vector&) const override;

    void set_bdry_dirichlet(Array<int>&) const override;

    Vector bdry_velocity(const Vector& x) const override {
        return velocity_sol(x);
    }

    inline int get_dim() const override {
        return m_dim;
    }

private:
    const nlohmann::json& m_config;
    int m_dim;
};

#endif /// STOKES_TEST_CASES_HPP
