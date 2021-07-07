#include "../../include/incompNS/coefficients.hpp"


//-------------------------------//
//  Exact Solution Coefficients  //
//-------------------------------//

//! Incompressible Navier-Stokes exact pressure coefficient
double IncompNSExactPressureCoeff
:: Eval (ElementTransformation& T,
         const IntegrationPoint& ip)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    return (m_testCase->pressure_sol(transip, GetTime()));
}

double IncompNSExactPressureCoeff
:: Eval (ElementTransformation& T,
         const IntegrationPoint& ip,
         double t)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    return (m_testCase->pressure_sol(transip, t));
}


//! Incompressible Navier-Stokes exact velocity coefficient
void IncompNSExactVelocityCoeff
:: Eval (Vector& v,
         ElementTransformation& T,
         const IntegrationPoint& ip)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    v = m_testCase->velocity_sol(transip, GetTime());
}

void IncompNSExactVelocityCoeff
:: Eval (Vector& v,
         ElementTransformation& T,
         const IntegrationPoint& ip,
         double t)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    v = m_testCase->velocity_sol(transip, t);
}


//---------------------------------//
//  Initial Solution Coefficients  //
//---------------------------------//

//! Incompressible Navier-Stokes initial pressure coefficient
double IncompNSInitialPressureCoeff
:: Eval (ElementTransformation& T,
         const IntegrationPoint& ip)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    return (m_testCase->init_pressure(transip));
}

//! Incompressible Navier-Stokes initial velocity coefficient
void IncompNSInitialVelocityCoeff
:: Eval (Vector& v,
         ElementTransformation& T,
         const IntegrationPoint& ip)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    v = m_testCase->init_velocity(transip);
}


//---------------------------------//
//  Boundary Velocity Coefficient  //
//---------------------------------//

//! Incompressible Navier-Stokes bdry velocity coefficient
void IncompNSBdryVelocityCoeff
:: Eval (Vector& v,
         ElementTransformation& T,
         const IntegrationPoint& ip)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    v = m_testCase->bdry_velocity(transip, GetTime());
}

void IncompNSBdryVelocityCoeff
:: Eval (Vector& v,
         ElementTransformation& T,
         const IntegrationPoint& ip,
         double t)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    v = m_testCase->bdry_velocity(transip, t);
}


//-----------------------//
//  Source Coefficients  //
//-----------------------//

//! Incompressible Navier-Stokes source coefficient
void IncompNSSourceCoeff
:: Eval (Vector& v,
         ElementTransformation& T,
         const IntegrationPoint& ip)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    v = m_testCase->source(transip, GetTime());
}

void IncompNSSourceCoeff
:: Eval (Vector& v,
         ElementTransformation& T,
         const IntegrationPoint& ip,
         double t)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    v = m_testCase->source(transip, t);
}
