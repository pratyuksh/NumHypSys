#ifndef INCOMPNS_COEFFICIENTS_HPP
#define INCOMPNS_COEFFICIENTS_HPP

#include "test_cases.hpp"


//-------------------------------//
//  Exact Solution Coefficients  //
//-------------------------------//

//! Class for Incompressible Navier-Stokes
//! exact pressure coefficient
class IncompNSExactPressureCoeff : public Coefficient
{
public:
    IncompNSExactPressureCoeff(std::shared_ptr
                               <IncompNSTestCases> testCase)
        : m_testCase (testCase) {}
    
    virtual double Eval(ElementTransformation &,
                        const IntegrationPoint &);
    
    double Eval(ElementTransformation &,
                const IntegrationPoint &,
                double);
    
protected:
    std::shared_ptr<IncompNSTestCases> m_testCase;
};


//! Class for Incompressible Navier-Stokes
//! exact velocity coefficient
class IncompNSExactVelocityCoeff : public VectorCoefficient
{
public:
    IncompNSExactVelocityCoeff
    (std::shared_ptr <IncompNSTestCases> testCase)
        : VectorCoefficient (testCase->get_dim()),
          m_testCase (testCase) {}

    virtual void Eval(Vector&, ElementTransformation&,
                      const IntegrationPoint&);
    
    void Eval(Vector&, ElementTransformation&,
              const IntegrationPoint&, double);
    
private:
    std::shared_ptr<IncompNSTestCases> m_testCase;
};


//---------------------------------//
//  Initial Solution Coefficients  //
//---------------------------------//

//! Class for Incompressible Navier-Stokes
//! initial pressure coefficient
class IncompNSInitialPressureCoeff : public Coefficient
{
public:
    IncompNSInitialPressureCoeff
    (std::shared_ptr <IncompNSTestCases> testCase)
        : m_testCase (testCase) {}

    virtual double Eval(ElementTransformation &,
                        const IntegrationPoint &);

protected:
    std::shared_ptr<IncompNSTestCases> m_testCase;
};


//! Class for Incompressible Navier-Stokes
//! initial velocity coefficient
class IncompNSInitialVelocityCoeff
        : public VectorCoefficient
{
public:
    IncompNSInitialVelocityCoeff
    (std::shared_ptr <IncompNSTestCases> testCase)
        : VectorCoefficient (testCase->get_dim()),
          m_testCase (testCase) {}

    virtual void Eval(Vector&, ElementTransformation&,
                      const IntegrationPoint&);

private:
    std::shared_ptr<IncompNSTestCases> m_testCase;
};


//---------------------------------//
//  Boundary Velocity Coefficient  //
//---------------------------------//

//! Class for Incompressible Navier-Stokes
//! boundary velocity coefficient
class IncompNSBdryVelocityCoeff : public VectorCoefficient
{
public:
    IncompNSBdryVelocityCoeff
    (std::shared_ptr <IncompNSTestCases> testCase)
        : VectorCoefficient (testCase->get_dim()),
          m_testCase (testCase) {}

    virtual void Eval(Vector&, ElementTransformation&,
                      const IntegrationPoint&);

    void Eval(Vector&, ElementTransformation&,
              const IntegrationPoint&, double);

private:
    std::shared_ptr<IncompNSTestCases> m_testCase;
};


//-----------------------//
//  Source Coefficients  //
//-----------------------//

//! Class for Incompressible Navier-Stokes source coefficient
class IncompNSSourceCoeff : public VectorCoefficient
{
public:
    IncompNSSourceCoeff (std::shared_ptr
                         <IncompNSTestCases> testCase)
        : VectorCoefficient (testCase->get_dim()),
          m_testCase (testCase) {}

    virtual void Eval(Vector&, ElementTransformation&,
                      const IntegrationPoint&);

    void Eval(Vector&, ElementTransformation&,
              const IntegrationPoint&, double);

private:
    std::shared_ptr<IncompNSTestCases> m_testCase;
};


#endif /// INCOMPNS_COEFFICIENTS_HPP
