#ifndef STOKES_COEFFICIENTS_HPP
#define STOKES_COEFFICIENTS_HPP

#include "test_cases.hpp"


//-------------------------------//
//  Exact Solution Coefficients  //
//-------------------------------//

// class for Stokes
// exact pressure coefficient
class StokesExactPressureCoeff : public Coefficient
{
public:
    StokesExactPressureCoeff(std::shared_ptr
                             <StokesTestCases> testCase)
        : m_testCase (testCase) {}
    
    virtual double Eval(ElementTransformation &,
                        const IntegrationPoint &);
    
protected:
    std::shared_ptr<StokesTestCases> m_testCase;
};


// class for Stokes
// exact velocity coefficient
class StokesExactVelocityCoeff : public VectorCoefficient
{
public:
    StokesExactVelocityCoeff (std::shared_ptr
                                <StokesTestCases> testCase)
        : VectorCoefficient (testCase->get_dim()),
          m_testCase (testCase) {}

    virtual void Eval(Vector&, ElementTransformation&,
                      const IntegrationPoint&);
    
private:
    std::shared_ptr<StokesTestCases> m_testCase;
};


//---------------------------------//
//  Boundary Velocity Coefficient  //
//---------------------------------//

// class for Stokes
// boundary velocity coefficient
class StokesBdryVelocityCoeff : public VectorCoefficient
{
public:
    StokesBdryVelocityCoeff (std::shared_ptr
                                <StokesTestCases> testCase)
        : VectorCoefficient (testCase->get_dim()),
          m_testCase (testCase) {}

    virtual void Eval(Vector&, ElementTransformation&,
                      const IntegrationPoint&);

private:
    std::shared_ptr<StokesTestCases> m_testCase;
};


//-----------------------//
//  Source Coefficients  //
//-----------------------//

// class for Stokes source coefficient
class StokesSourceCoeff : public VectorCoefficient
{
public:
    StokesSourceCoeff (std::shared_ptr
                         <StokesTestCases> testCase)
        : VectorCoefficient (testCase->get_dim()),
          m_testCase (testCase) {}

    virtual void Eval(Vector&, ElementTransformation&,
                      const IntegrationPoint&);

private:
    std::shared_ptr<StokesTestCases> m_testCase;
};


#endif /// STOKES_COEFFICIENTS_HPP
