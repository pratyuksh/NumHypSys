#ifndef INCOMPNS_TEST_CASES_HPP
#define INCOMPNS_TEST_CASES_HPP

#include "../core/config.hpp"
#include "mfem.hpp"

using namespace mfem;

// Cases
enum {TaylorGreenVortex_Test1,
      TaylorGreenVortex_Test2,
      TaylorGreenVortex_Test3,
      TaylorGreenVortex_Test4,
      DoubleShearLayer,
      LidDrivenCavity,
      KovasznayFlow,
      ChannelFlow,
      UqTestModel,
      UQSmoothVortexSheet,
      UQLidDrivenCavity,
      UQChannelFlow
     };


//! Abstract Base class for
//! Incompressible Navier-Stokes test cases
class IncompNSTestCases
{
public:

    virtual ~IncompNSTestCases() = default;
    
    virtual double pressure_sol(const Vector&,
                                const double) const = 0;

    virtual Vector velocity_sol(const Vector&,
                                const double) const = 0;
    
    virtual double init_pressure(const Vector&) const = 0;

    virtual Vector init_velocity(const Vector&) const = 0;

    virtual Vector bdry_velocity(const Vector&,
                                 const double) const = 0;

    virtual Vector source(const Vector&,
                          const double) const = 0;

    virtual void set_bdry_dirichlet(Array<int>&) const = 0;
    
    virtual int get_dim() const = 0;

    virtual void set_perturbations(const Vector&) const = 0;
};


//! Class IncompNSTestCase, base template
template<int ProblemType>
class IncompNSTestCase;


//! Case: inviscid, Taylor Green Vortex
//! Homogeneous Dirichlet boundary conditions
template<>
class IncompNSTestCase <TaylorGreenVortex_Test1>
        : public IncompNSTestCases
{
public:
    explicit IncompNSTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2) {
        m_viscosity = config["viscosity"];
    }
    
    double pressure_sol(const Vector&,
                        const double) const override;

    Vector velocity_sol(const Vector&,
                        const double) const override;

    Vector source(const Vector&,
                  const double) const override;
    
    void set_bdry_dirichlet(Array<int>&) const override;

    double init_pressure(const Vector& x) const override {
        return pressure_sol(x, 0);
    }

    Vector init_velocity(const Vector& x) const override {
        return velocity_sol(x, 0);
    }

    Vector bdry_velocity(const Vector& x,
                         const double t) const override {
        return velocity_sol(x, t);
    }

    inline int get_dim() const override {
        return m_dim;
    }

    void set_perturbations(const Vector&) const override {}

private:
    const nlohmann::json& m_config;
    int m_dim;
    
    double m_viscosity = 1E-2;
};


//! Case: viscous, Taylor Green Vortex
//! Homogeneous Dirichlet boundary conditions
template<>
class IncompNSTestCase <TaylorGreenVortex_Test2>
        : public IncompNSTestCases
{
public:
    explicit IncompNSTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2) {
        m_viscosity = config["viscosity"];
    }
    
    double pressure_sol(const Vector&,
                        const double) const override;

    Vector velocity_sol(const Vector&,
                        const double) const override;

    Vector source(const Vector&,
                  const double) const override;
    
    void set_bdry_dirichlet(Array<int>&) const override;

    double init_pressure(const Vector& x) const override {
        return pressure_sol(x, 0);
    }

    Vector init_velocity(const Vector& x) const override {
        return velocity_sol(x, 0);
    }

    Vector bdry_velocity(const Vector& x,
                         const double t) const override {
        return velocity_sol(x, t);
    }

    inline int get_dim() const override {
        return m_dim;
    }

    void set_perturbations(const Vector&) const override {}

private:
    const nlohmann::json& m_config;
    int m_dim;
    
    double m_viscosity = 1E-2;
};


//! Case: inviscid, Taylor Green Vortex
//! Non-homogeneous Dirichlet boundary conditions
template<>
class IncompNSTestCase <TaylorGreenVortex_Test3>
        : public IncompNSTestCases
{
public:
    explicit IncompNSTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2) {
        m_viscosity = config["viscosity"];
    }

    double pressure_sol(const Vector&,
                        const double) const override;

    Vector velocity_sol(const Vector&,
                        const double) const override;

    Vector source(const Vector&,
                  const double) const override;

    void set_bdry_dirichlet(Array<int>&) const override;

    double init_pressure(const Vector& x) const override {
        return pressure_sol(x, 0);
    }

    Vector init_velocity(const Vector& x) const override {
        return velocity_sol(x, 0);
    }

    Vector bdry_velocity(const Vector& x,
                         const double t) const override {
        return velocity_sol(x, t);
    }

    inline int get_dim() const override {
        return m_dim;
    }

    void set_perturbations(const Vector&) const override {}

private:
    const nlohmann::json& m_config;
    int m_dim;

    double m_viscosity = 1E-2;
};


//! Case: viscous, Taylor Green Vortex
//! Non-homogeneous Dirichlet boundary conditions
template<>
class IncompNSTestCase <TaylorGreenVortex_Test4>
        : public IncompNSTestCases
{
public:
    explicit IncompNSTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2) {
        m_viscosity = config["viscosity"];
    }

    double pressure_sol(const Vector&,
                        const double) const override;

    Vector velocity_sol(const Vector&,
                        const double) const override;

    Vector source(const Vector&,
                  const double) const override;

    void set_bdry_dirichlet(Array<int>&) const override;

    double init_pressure(const Vector& x) const override {
        return pressure_sol(x, 0);
    }

    Vector init_velocity(const Vector& x) const override {
        return velocity_sol(x, 0);
    }

    Vector bdry_velocity(const Vector& x,
                         const double t) const override {
        return velocity_sol(x, t);
    }

    inline int get_dim() const override {
        return m_dim;
    }

    void set_perturbations(const Vector&) const override {}

private:
    const nlohmann::json& m_config;
    int m_dim;

    double m_viscosity = 1E-2;
};


//! Case: Double Shear Layer
//! Periodic boundary conditions
template<>
class IncompNSTestCase <DoubleShearLayer>
        : public IncompNSTestCases
{
public:
    explicit IncompNSTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2) {}

    double init_pressure(const Vector&) const override;

    Vector init_velocity(const Vector&) const override;

    Vector source(const Vector&,
                  const double) const override;

    double pressure_sol(const Vector&,
                        const double) const override {
        return 0;
    }

    Vector velocity_sol(const Vector&,
                        const double) const override {
        Vector v(m_dim);
        return v;
    }

    Vector bdry_velocity(const Vector& x,
                         const double t) const override {
        return velocity_sol(x, t);
    }

    void set_bdry_dirichlet(Array<int>&) const override {}

    inline int get_dim() const override {
        return m_dim;
    }

    void set_perturbations(const Vector&) const override {}

private:
    const nlohmann::json& m_config;
    int m_dim;

    double m_epsilon = 5e-02;
    double m_kappa = M_PI/15.;
};


//! Case: Lid Driven Cavity
//! Non-homogeneous Dirichlet boundary conditions
template<>
class IncompNSTestCase <LidDrivenCavity>
        : public IncompNSTestCases
{
public:
    explicit IncompNSTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2) {}

    double init_pressure(const Vector&) const override;

    Vector init_velocity(const Vector&) const override;

    Vector bdry_velocity(const Vector& x,
                         const double t) const override;

    Vector source(const Vector&,
                  const double) const override;

    void set_bdry_dirichlet(Array<int>&) const override;

    double pressure_sol(const Vector&,
                        const double) const override {
        return 0;
    }

    Vector velocity_sol(const Vector&,
                        const double) const override {
        Vector v(m_dim);
        return v;
    }

    inline int get_dim() const override {
        return m_dim;
    }

    void set_perturbations(const Vector&) const override {}

private:
    const nlohmann::json& m_config;
    int m_dim;
};


//! Case: Kovasznay Flow
//! Inflow, outflow in x
//! Periodic walls in y
template<>
class IncompNSTestCase <KovasznayFlow>
        : public IncompNSTestCases
{
public:
    explicit IncompNSTestCase (const nlohmann::json& config);

    double pressure_sol(const Vector&,
                        const double) const override;

    Vector velocity_sol(const Vector&,
                        const double) const override;

    Vector source(const Vector&,
                  const double) const override;

    void set_bdry_dirichlet(Array<int>&) const override;

    double init_pressure(const Vector& x) const override {
        return pressure_sol(x, 0);
    }

    Vector init_velocity(const Vector& x) const override {
        return velocity_sol(x, 0);
    }

    Vector bdry_velocity(const Vector& x,
                         const double t) const override {
        return velocity_sol(x, t);
    }

    inline int get_dim() const override {
        return m_dim;
    }

    void set_perturbations(const Vector&) const override {}

private:
    const nlohmann::json& m_config;
    int m_dim;

    double m_viscosity;
    double m_lambda;
};


//! Case: Channel Flow
//! Inflow, outflow in x
//! Fixed walls in y
template<>
class IncompNSTestCase <ChannelFlow>
        : public IncompNSTestCases
{
public:
    explicit IncompNSTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2) {}

    double init_pressure(const Vector&) const override;

    Vector init_velocity(const Vector&) const override;

    Vector bdry_velocity(const Vector& x,
                         const double t) const override;

    Vector source(const Vector&,
                  const double) const override;

    void set_bdry_dirichlet(Array<int>&) const override;

    double pressure_sol(const Vector&,
                        const double) const override {
        return 0;
    }

    Vector velocity_sol(const Vector&,
                        const double) const override {
        Vector v(m_dim);
        return v;
    }

    inline int get_dim() const override {
        return m_dim;
    }

    void set_perturbations(const Vector&) const override {}

private:
    const nlohmann::json& m_config;
    int m_dim;

    double um = 1.5;
    double H = 0.5;
    double oneDivH2 = 1./(H*H);
    double fourUmDivH2 = 4*um*oneDivH2;
};


//! Case: UQ test model
template<>
class IncompNSTestCase <UqTestModel>
        : public IncompNSTestCases
{
public:
    explicit IncompNSTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2)
    {}

    double init_pressure(const Vector&) const override;

    Vector init_velocity(const Vector&) const override;

    Vector source(const Vector&,
                  const double) const override;

    double pressure_sol(const Vector&,
                        const double) const override {
        return 0;
    }

    Vector velocity_sol(const Vector&,
                        const double) const override {
        Vector v(m_dim);
        return v;
    }

    Vector bdry_velocity(const Vector& x,
                         const double t) const override {
        return velocity_sol(x, t);
    }

    void set_bdry_dirichlet(Array<int>&) const override {}

    inline int get_dim() const override {
        return m_dim;
    }

    void set_perturbations(const Vector& w) const override {
        m_rvars = w;
    }

private:

    Vector perturb_init_velocity(const Vector&) const;

private:
    const nlohmann::json& m_config;
    int m_dim;

    mutable Vector m_rvars;
};


//! Case: UQ Smooth Vortex Sheet
//! Periodic boundary conditions
template<>
class IncompNSTestCase <UQSmoothVortexSheet>
        : public IncompNSTestCases
{
public:
    explicit IncompNSTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2)
    {
        if (config.contains("svs_epsilon")) {
            m_epsilon = config["svs_epsilon"];
        }
        if (config.contains("svs_rho")) {
            m_rho = config["svs_rho"];
        }
        if (config.contains("svs_gamma")) {
            m_gamma = config["svs_gamma"];
        }
    }

    double init_pressure(const Vector&) const override;

    Vector init_velocity(const Vector&) const override;

    Vector source(const Vector&,
                  const double) const override;

    double pressure_sol(const Vector&,
                        const double) const override {
        return 0;
    }

    Vector velocity_sol(const Vector&,
                        const double) const override {
        Vector v(m_dim);
        return v;
    }

    Vector bdry_velocity(const Vector& x,
                         const double t) const override {
        return velocity_sol(x, t);
    }

    void set_bdry_dirichlet(Array<int>&) const override {}

    inline int get_dim() const override {
        return m_dim;
    }

    void set_perturbations(const Vector& w) const override {
        m_rvars = w;
    }

private:
    Vector perturb_coords(const Vector& x) const;

private:
    const nlohmann::json& m_config;
    int m_dim;

    double m_epsilon = 0;
    double m_rho = 0.1;

    double m_gamma = 0.025;
    mutable Vector m_rvars;
};


//! Case: UQ Lid Driven Cavity
//! Non-homogeneous Dirichlet boundary conditions
template<>
class IncompNSTestCase <UQLidDrivenCavity>
        : public IncompNSTestCases
{
public:
    explicit IncompNSTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2)
    {
        if (config.contains("ldc_gamma1")) {
            m_gamma1 = config["ldc_gamma1"];
        }
        if (config.contains("ldc_gamma2")) {
            m_gamma2 = config["ldc_gamma2"];
        }
    }

    double init_pressure(const Vector&) const override;

    Vector init_velocity(const Vector&) const override;

    Vector bdry_velocity(const Vector& x,
                         const double t) const override;

    Vector source(const Vector&,
                  const double) const override;

    void set_bdry_dirichlet(Array<int>&) const override;

    double pressure_sol(const Vector&,
                        const double) const override {
        return 0;
    }

    Vector velocity_sol(const Vector&,
                        const double) const override {
        Vector v(m_dim);
        return v;
    }

    inline int get_dim() const override {
        return m_dim;
    }

    void set_perturbations(const Vector& w) const override {
        m_rvars = w;
    }

private:
    Vector perturb_coords(const Vector&) const;
    Vector perturb_bdry_velocity(const Vector&) const;

private:
    const nlohmann::json& m_config;
    int m_dim;

    double m_gamma1 = 0.025;
    double m_gamma2 = 0.1;
    mutable Vector m_rvars;
};


//! Case: UQChannel flow
//! Inflow, outflow in x
//! Fixed walls in y
template<>
class IncompNSTestCase <UQChannelFlow>
        : public IncompNSTestCases
{
public:
    explicit IncompNSTestCase (const nlohmann::json& config)
        : m_config(config), m_dim(2)
    {
        if (config.contains("cf_gamma_vx")) {
            m_gamma_vx = config["cf_gamma_vx"];
        }

        if (config.contains("cf_gamma_vy")) {
            m_gamma_vy = config["cf_gamma_vy"];
        }
    }

    double init_pressure(const Vector&) const override;

    Vector init_velocity(const Vector&) const override;

    Vector bdry_velocity(const Vector& x,
                         const double t) const override;

    Vector source(const Vector&,
                  const double) const override;

    void set_bdry_dirichlet(Array<int>&) const override;

    double pressure_sol(const Vector&,
                        const double) const override {
        return 0;
    }

    Vector velocity_sol(const Vector&,
                        const double) const override {
        Vector v(m_dim);
        return v;
    }

    inline int get_dim() const override {
        return m_dim;
    }

    void set_perturbations(const Vector& w) const override {
        m_rvars = w;
    }

private:
    Vector perturb_velocity(const Vector&,
                            const Vector&) const;

private:
    const nlohmann::json& m_config;
    int m_dim;

    double um = 1.5;
    double H = 0.5;
    double oneDivH2 = 1./(H*H);
    double fourUmDivH2 = 4*um*oneDivH2;

    double m_gamma_vx = 0.01;
    double m_gamma_vy = 0.01;
    mutable Vector m_rvars;
};

#endif /// INCOMPNS_TEST_CASES_HPP
