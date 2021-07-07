#include "../../include/stokes/observer.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>

#include "../../include/mymfem/utilities.hpp"

namespace fs = std::filesystem;


// Constructor
StokesObserver
:: StokesObserver (const nlohmann::json& config, int lx)
    : BaseObserver (config, lx)
{
    m_bool_error = config["eval_error"];
    
    std::string problem_type = config["problem_type"];
    m_output_dir = "../output/stokes_"+problem_type+"/";
    if (m_bool_dumpOut) {
        fs::create_directories(m_output_dir);
    }
    m_solName_suffix = "_lx"+std::to_string(lx);
}

// Dump solution
void StokesObserver
:: dump_sol (std::shared_ptr<GridFunction>& v,
             std::shared_ptr<GridFunction>& p) const
{
    if (m_bool_dumpOut)
    {
        std::string sol_name1
                = m_output_dir+"velocity"+m_solName_suffix;
        std::string sol_name2
                = m_output_dir+"pressure"+m_solName_suffix;
        std::cout << sol_name1 << std::endl;
        std::cout << sol_name2 << std::endl;

        std::ofstream sol_ofs1(sol_name1.c_str());
        sol_ofs1.precision(m_precision);
        v->Save(sol_ofs1);

        std::ofstream sol_ofs2(sol_name2.c_str());
        sol_ofs2.precision(m_precision);
        p->Save(sol_ofs2);
    }
}

// Compute L2 error for velocity and pressure
std::tuple <double, double, double, double> StokesObserver
:: eval_error_L2 (std::shared_ptr<StokesTestCases>& testCase,
                  std::shared_ptr<GridFunction>& v,
                  std::shared_ptr<GridFunction>& p) const
{
    double errvL2=0, errpL2=0;
    double vEL2=0, pEL2=0;

    int dim = v->FESpace()->GetMesh()->Dimension();
    ConstantCoefficient zero(0);
    VectorFunctionCoefficient zeroVec(dim, zeroFn);

    // error in velocity
    StokesExactVelocityCoeff vE_coeff(testCase);
    GridFunction vE(*v);
    vE.ProjectCoefficient(vE_coeff);
    vEL2 = vE.ComputeL2Error(zeroVec);
    errvL2 = v->ComputeL2Error(vE_coeff);

    // error in pressure
    StokesExactPressureCoeff pE_coeff(testCase);
    GridFunction pE(*p);
    pE.ProjectCoefficient(pE_coeff);
    pEL2 = pE.ComputeL2Error(zero);
    errpL2 = p->ComputeL2Error(pE_coeff);

    return {errvL2, vEL2, errpL2, pEL2};
}

// End of file
