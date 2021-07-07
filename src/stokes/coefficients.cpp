#include "../../include/stokes/coefficients.hpp"


//-------------------------------//
//  Exact Solution Coefficients  //
//-------------------------------//

// Stokes exact pressure coefficient
double StokesExactPressureCoeff
:: Eval (ElementTransformation& T,
         const IntegrationPoint& ip)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    return (m_testCase->pressure_sol(transip));
}


// Stokes exact velocity coefficient
void StokesExactVelocityCoeff
:: Eval (Vector& v,
         ElementTransformation& T,
         const IntegrationPoint& ip)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    v = m_testCase->velocity_sol(transip);
}


//---------------------------------//
//  Boundary Velocity Coefficient  //
//---------------------------------//

// Stokes boundary velocity coefficient
void StokesBdryVelocityCoeff
:: Eval (Vector& v,
         ElementTransformation& T,
         const IntegrationPoint& ip)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    v = m_testCase->bdry_velocity(transip);
}


//-----------------------//
//  Source Coefficients  //
//-----------------------//

// Stokes source coefficient
void StokesSourceCoeff :: Eval (Vector& v,
                                ElementTransformation& T,
                                const IntegrationPoint& ip)
{
    double x[3];
    Vector transip(x, 3);
    T.Transform(ip, transip);
    v = m_testCase->source(transip);
}

// End of file
