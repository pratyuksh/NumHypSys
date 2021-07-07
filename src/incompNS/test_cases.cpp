#include "../../include/incompNS/test_cases.hpp"


//! Case: inviscid, Taylor Green Vortex
//! Homogeneous Dirichlet boundary conditions
double IncompNSTestCase <TaylorGreenVortex_Test1>
:: pressure_sol(const Vector& x, const double t) const
{
    double pressure
            = (1./4.)*(cos(2*x(0)) + cos(2*x(1)))
            *exp(-4*m_viscosity*t);
    return pressure;
}

Vector IncompNSTestCase <TaylorGreenVortex_Test1>
:: velocity_sol (const Vector& x, const double t) const
{
    Vector v(m_dim);
    v(0) = +sin(x(0))*cos(x(1))*exp(-2*m_viscosity*t);
    v(1) = -cos(x(0))*sin(x(1))*exp(-2*m_viscosity*t);
    return v;
}

Vector IncompNSTestCase <TaylorGreenVortex_Test1>
:: source (const Vector& x, const double t) const
{
    Vector f(m_dim);
    double coeff = -2.*m_viscosity;
    f(0) = +coeff*sin(x(0))*cos(x(1))*exp(-2*m_viscosity*t);
    f(1) = -coeff*cos(x(0))*sin(x(1))*exp(-2*m_viscosity*t);
    return f;
}

void IncompNSTestCase <TaylorGreenVortex_Test1>
:: set_bdry_dirichlet(Array<int>& bdr_marker) const
{
    bdr_marker = 1;
}


//! Case: viscous, Taylor Green Vortex
//! Homogeneous Dirichlet boundary conditions
double IncompNSTestCase <TaylorGreenVortex_Test2>
:: pressure_sol(const Vector& x, const double t) const
{
    double pressure
            = (1./4.)*(cos(2*x(0)) + cos(2*x(1)))
            *exp(-4*m_viscosity*t);
    return pressure;
}

Vector IncompNSTestCase <TaylorGreenVortex_Test2>
:: velocity_sol (const Vector& x, const double t) const
{
    Vector v(m_dim);
    v(0) = +sin(x(0))*cos(x(1))*exp(-2*m_viscosity*t);
    v(1) = -cos(x(0))*sin(x(1))*exp(-2*m_viscosity*t);
    return v;
}

Vector IncompNSTestCase <TaylorGreenVortex_Test2>
:: source (const Vector&, const double) const
{
    Vector f(m_dim);
    f = 0.0;
    return f;
}

void IncompNSTestCase <TaylorGreenVortex_Test2>
:: set_bdry_dirichlet(Array<int>& bdr_marker) const
{
    bdr_marker = 1;
}


//! Case: inviscid, Taylor Green Vortex
//! Non-homogeneous Dirichlet boundary conditions
double IncompNSTestCase <TaylorGreenVortex_Test3>
:: pressure_sol(const Vector& x, const double t) const
{
    double pressure
            = -(1./4.)*(cos(2*x(0)) - cos(2*x(1)))
            *exp(-4*m_viscosity*t);
    return pressure;
}

Vector IncompNSTestCase <TaylorGreenVortex_Test3>
:: velocity_sol (const Vector& x, const double t) const
{
    Vector v(m_dim);
    v(0) = cos(x(0))*cos(x(1))*exp(-2*m_viscosity*t);
    v(1) = sin(x(0))*sin(x(1))*exp(-2*m_viscosity*t);
    return v;
}

Vector IncompNSTestCase <TaylorGreenVortex_Test3>
:: source (const Vector& x, const double t) const
{
    Vector f(m_dim);
    double coeff = -2.*m_viscosity;
    f(0) = coeff*cos(x(0))*cos(x(1))*exp(-2*m_viscosity*t);
    f(1) = coeff*sin(x(0))*sin(x(1))*exp(-2*m_viscosity*t);
    return f;
}

void IncompNSTestCase <TaylorGreenVortex_Test3>
:: set_bdry_dirichlet(Array<int>& bdr_marker) const
{
    bdr_marker = 1;
}


//! Case: viscous, Taylor Green Vortex
//! Non-homogeneous Dirichlet boundary conditions
double IncompNSTestCase <TaylorGreenVortex_Test4>
:: pressure_sol(const Vector& x, const double t) const
{
    double pressure
            = -(1./4.)*(cos(2*x(0)) - cos(2*x(1)))
            *exp(-4*m_viscosity*t);
    return pressure;
}

Vector IncompNSTestCase <TaylorGreenVortex_Test4>
:: velocity_sol (const Vector& x, const double t) const
{
    Vector v(m_dim);
    v(0) = cos(x(0))*cos(x(1))*exp(-2*m_viscosity*t);
    v(1) = sin(x(0))*sin(x(1))*exp(-2*m_viscosity*t);
    return v;
}

Vector IncompNSTestCase <TaylorGreenVortex_Test4>
:: source (const Vector&, const double) const
{
    Vector f(m_dim);
    f = 0.0;
    return f;
}

void IncompNSTestCase <TaylorGreenVortex_Test4>
:: set_bdry_dirichlet(Array<int>& bdr_marker) const
{
    bdr_marker = 1;
}


//! Case: Double Shear Layer
//! Periodic boundary conditions
double IncompNSTestCase <DoubleShearLayer>
:: init_pressure(const Vector&) const
{
    return 0;
}

Vector IncompNSTestCase <DoubleShearLayer>
:: init_velocity (const Vector& x) const
{
    Vector v(m_dim);

    // x-component
    if (x(1) <= M_PI)
        v(0) = tanh((x(1)-M_PI/2)/m_kappa);
    else
        v(0) = tanh((3*M_PI/2-x(1))/m_kappa);

    // y-component
    v(1) = m_epsilon*sin(x(0));

    return v;
}

Vector IncompNSTestCase <DoubleShearLayer>
:: source (const Vector&, const double) const
{
    Vector f(m_dim);
    f = 0.0;
    return f;
}


//! Case: Lid Driven Cavity
//! Non-homogeneous Dirichlet boundary conditions
double IncompNSTestCase <LidDrivenCavity>
:: init_pressure(const Vector&) const
{
    return 0;
}

Vector IncompNSTestCase <LidDrivenCavity>
:: init_velocity (const Vector& x) const
{
    Vector v(m_dim);
    v(0) = x(1) - 0.5;
    v(1) = -(x(0) - 0.5);
    return v;
}

Vector IncompNSTestCase <LidDrivenCavity>
:: bdry_velocity(const Vector& x, const double) const
{
    Vector v(m_dim);
    v = 0.0;

    double tol = 1E-12;
    if (std::abs(x[1]-1) <= tol) { v(0) = 1; }
    return v;
}

Vector IncompNSTestCase <LidDrivenCavity>
:: source (const Vector&, const double) const
{
    Vector f(m_dim);
    f = 0.0;
    return f;
}

void IncompNSTestCase <LidDrivenCavity>
:: set_bdry_dirichlet(Array<int>& bdr_marker) const
{
    bdr_marker = 1;
}


//! Case: Kovasznay Flow
//! Inflow, outflow in x
//! Periodic walls in y
IncompNSTestCase <KovasznayFlow>
:: IncompNSTestCase (const nlohmann::json& config)
    : m_config(config), m_dim(2)
{
    m_viscosity = 1./40;
    if (config.contains("viscosity")) {
        m_viscosity = config["viscosity"];
    }

    double tmp1 = 0.25/(m_viscosity*m_viscosity);
    double tmp2 = 4*M_PI*M_PI;
    m_lambda = 0.5/m_viscosity - std::sqrt(tmp1 + tmp2);
}

double IncompNSTestCase <KovasznayFlow>
:: pressure_sol(const Vector& x, const double) const
{
    double p = 0.5*(1 - exp(2*m_lambda*x(0)));
    return p;
}

Vector IncompNSTestCase <KovasznayFlow>
:: velocity_sol (const Vector& x, const double) const
{
    Vector v(m_dim);
    v(0) = 1 - exp(m_lambda*x(0))*cos(2*M_PI*x(1));
    v(1) = m_lambda*exp(m_lambda*x(0))*sin(2*M_PI*x(1))
            /(2*M_PI);

    return v;
}

Vector IncompNSTestCase <KovasznayFlow>
:: source (const Vector&, const double) const
{
    Vector f(m_dim);
    f = 0.0;
    return f;
}

void IncompNSTestCase <KovasznayFlow>
:: set_bdry_dirichlet(Array<int>& bdr_marker) const
{
    bdr_marker = 1;
}


//! Case: Channel Flow
//! Inflow, outflow in x
//! Fixed walls in y
double IncompNSTestCase <ChannelFlow>
:: init_pressure(const Vector&) const
{
    double p = 0;
    return p;
}

Vector IncompNSTestCase <ChannelFlow>
:: init_velocity (const Vector& x) const
{
    Vector v(m_dim);
    v = 0.0;
    v(0) = 0.8*fourUmDivH2*x(1)*(H - x(1));
    //v(0) = fourUmDivH2*x(1)*(H - x(1));
    return v;
}

Vector IncompNSTestCase <ChannelFlow>
:: bdry_velocity(const Vector& x, const double) const
{
    Vector v(m_dim);
    v = 0.0;

    double tol = 1E-12;
    if (std::abs(x[0]) <= tol) {
        v(0) = fourUmDivH2*x(1)*(H - x(1));
    }

    return v;
}

Vector IncompNSTestCase <ChannelFlow>
:: source (const Vector&, const double) const
{
    Vector f(m_dim);
    f = 0.0;
    return f;
}

void IncompNSTestCase <ChannelFlow>
:: set_bdry_dirichlet(Array<int>& bdr_marker) const
{
    bdr_marker = 1;
    bdr_marker[2] = 0; // outflow
}


//! Case: UQ test model
double IncompNSTestCase <UqTestModel>
:: init_pressure(const Vector&) const
{
    return 0;
}

Vector IncompNSTestCase <UqTestModel>
:: init_velocity (const Vector& x) const
{
    assert(m_rvars.Size() == 1);
    Vector v(m_dim);
    v = 0.0;
    v(0) = 4*x(1)*(1 - x(1));

    return perturb_init_velocity(v);
}

Vector IncompNSTestCase <UqTestModel>
:: source (const Vector&, const double) const
{
    Vector f(m_dim);
    f = 0.0;
    return f;
}

Vector IncompNSTestCase <UqTestModel>
:: perturb_init_velocity (const Vector& v) const
{
    Vector vp(m_dim);
    vp = v;
    vp(0) += sin(2*M_PI*m_rvars(0));
    return vp;
}


//! Case: UQ Smooth Vortex Sheet
//! Periodic boundary conditions
double IncompNSTestCase <UQSmoothVortexSheet>
:: init_pressure(const Vector&) const
{
    return 0;
}

Vector IncompNSTestCase <UQSmoothVortexSheet>
:: init_velocity (const Vector& x) const
{
    assert(m_rvars.Size()%2 == 0);
    Vector xp = perturb_coords(x);
    Vector v(m_dim);

    // x-component
    if (xp(1) <= 0.5)
        v(0) = tanh((xp(1)-0.25)/m_rho);
    else
        v(0) = tanh((0.75-xp(1))/m_rho);

    // y-component
    //v(1) = 0;
    v(1) = m_epsilon*sin(2*M_PI*x(0));

    return v;
}

Vector IncompNSTestCase <UQSmoothVortexSheet>
:: source (const Vector&, const double) const
{
    Vector f(m_dim);
    f = 0.0;
    return f;
}

Vector IncompNSTestCase <UQSmoothVortexSheet>
:: perturb_coords (const Vector& x) const
{
    Vector xp(m_dim);
    xp = x;

    // perturb the y-coordinate
    double perturbVal = 0;
    for (int k=0; k<m_rvars.Size()/2; k++) {
        perturbVal += m_rvars(2*k)
                *sin(2*M_PI*k*(x(0) + m_rvars(2*k+1)));
    }
    xp(1) += m_gamma*perturbVal;

    return xp;
}


//! Case: UQ Lid Driven Cavity
//! Non-homogeneous Dirichlet boundary conditions
double IncompNSTestCase <UQLidDrivenCavity>
:: init_pressure(const Vector&) const
{
    return 0;
}

Vector IncompNSTestCase <UQLidDrivenCavity>
:: init_velocity (const Vector& x) const
{
    assert((m_rvars.Size()-1)%2 == 0);
    Vector xp = perturb_coords(x);

    Vector v(m_dim);
    v(0) = xp(1) - 0.5;
    v(1) = -(xp(0) - 0.5);
    return v;
}

Vector IncompNSTestCase <UQLidDrivenCavity>
:: bdry_velocity(const Vector& x, const double) const
{
    Vector v(m_dim);
    v = 0.0;

    double tol = 1E-12;
    if (std::abs(x[1]-1) <= tol) {
        v(0) = 1;
        v = perturb_bdry_velocity(v);
    }

    return v;
}

Vector IncompNSTestCase <UQLidDrivenCavity>
:: source (const Vector&, const double) const
{
    Vector f(m_dim);
    f = 0.0;
    return f;
}

void IncompNSTestCase <UQLidDrivenCavity>
:: set_bdry_dirichlet(Array<int>& bdr_marker) const
{
    bdr_marker = 1;
}

Vector IncompNSTestCase <UQLidDrivenCavity>
:: perturb_coords(const Vector& x) const
{
    Vector xp(m_dim);
    xp = x;

    double perturbValx = 0;
    double perturbValy = 0;
    for (int k=0; k<m_rvars.Size()/2; k++) {
        perturbValx += m_rvars(2*k)
                *sin(2*M_PI*k*(x(0)-0.5 + m_rvars(2*k+1)));
        perturbValy += m_rvars(2*k+1)
                *sin(2*M_PI*k*(x(1)-0.5 + m_rvars(2*k)));
    }
    xp(0) += m_gamma1*perturbValx;
    xp(1) += m_gamma1*perturbValy;

    return xp;
}

Vector IncompNSTestCase <UQLidDrivenCavity>
:: perturb_bdry_velocity(const Vector& v) const
{
    Vector vp(v);
    double perturbVal
            = sin(2*M_PI*m_rvars(m_rvars.Size()-1));
    vp(0) += m_gamma2*perturbVal;
    return vp;
}


//! Case: UQChannel flow
//! Inflow, outflow in x
//! Fixed walls in y
double IncompNSTestCase <UQChannelFlow>
:: init_pressure(const Vector&) const
{
    double p = 0;
    return p;
}

Vector IncompNSTestCase <UQChannelFlow>
:: init_velocity (const Vector& x) const
{
    Vector v(m_dim);
    v = 0.;
    v(0) = fourUmDivH2*x(1)*(H - x(1));
    return std::move(perturb_velocity(x, v));
}

Vector IncompNSTestCase <UQChannelFlow>
:: bdry_velocity(const Vector& x, const double) const
{
    Vector v(m_dim);
    v = 0.0;

    double tol = 1E-12;
    if (std::abs(x[0]) <= tol) {
        v = init_velocity(x);
    }

    return v;
}

Vector IncompNSTestCase <UQChannelFlow>
:: source (const Vector&, const double) const
{
    Vector f(m_dim);
    f = 0.0;
    return f;
}

void IncompNSTestCase <UQChannelFlow>
:: set_bdry_dirichlet(Array<int>& bdr_marker) const
{
    bdr_marker = 1;
    bdr_marker[2] = 0; // outflow
}

Vector IncompNSTestCase <UQChannelFlow>
:: perturb_velocity(const Vector& x, const Vector& v) const
{
    Vector vp(v);

    double perturbVal = 0;
    /*for (int k=0; k<m_rvars.Size(); k++) {
        double a = 0.5*(1 + m_rvars(k));
        perturbVal += a*sin(2*M_PI*k*(2*x(1) + a));
    }*/
    for (int k=0; k<m_rvars.Size()/2; k++) {
        perturbVal += m_rvars(2*k)
                *sin(2*M_PI*k*(x(1) + m_rvars(2*k+1)));
    }
    vp(0) *= (1 + m_gamma_vx*perturbVal);
    vp(1) = m_gamma_vy*perturbVal
            *oneDivH2*x(1)*(H - x(1));

    return vp;
}

// End of file
