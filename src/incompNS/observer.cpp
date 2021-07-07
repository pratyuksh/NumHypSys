#include "../../include/incompNS/observer.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>

#include "../../include/mymfem/utilities.hpp"

namespace fs = std::filesystem;


//! Constructors
IncompNSObserver
:: IncompNSObserver (const nlohmann::json& config, int lx)
    : BaseObserver (config, lx)
{
    std::string problem_type = config["problem_type"];

    m_bool_error = false;
    if (config.contains("eval_error")) {
        m_bool_error = config["eval_error"];
    }

    m_output_dir = "../output/incompNS_"+problem_type+"/";
    if (m_bool_dumpOut) {
        fs::create_directories(m_output_dir);
    }
    m_solName_suffix = "_lx"+std::to_string(lx);
}

IncompNSObserver
:: IncompNSObserver
(const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase, int lx)
    : IncompNSObserver (config, lx)
{
    m_testCase = testCase;
}

//! Computes L2 error for velocity and pressure
std::tuple <double, double, double, double> IncompNSObserver
:: eval_error_L2 (const double t,
                  std::shared_ptr<GridFunction>& v,
                  std::shared_ptr<GridFunction>& p) const
{
    double errvL2=0, errpL2=0;
    double vEL2=0, pEL2=0;

    int dim = v->FESpace()->GetMesh()->Dimension();
    ConstantCoefficient zero(0);
    VectorFunctionCoefficient zeroVec(dim, zeroFn);

    // error in velocity
    IncompNSExactVelocityCoeff vE_coeff(m_testCase);
    vE_coeff.SetTime(t);
    GridFunction vE(*v);
    vE.ProjectCoefficient(vE_coeff);
    vEL2 = vE.ComputeL2Error(zeroVec);
    errvL2 = v->ComputeL2Error(vE_coeff);

    // error in pressure
    IncompNSExactPressureCoeff pE_coeff(m_testCase);
    pE_coeff.SetTime(t);
    GridFunction pE(*p);
    pE.ProjectCoefficient(pE_coeff);
    pEL2 = pE.ComputeL2Error(zero);
    errpL2 = p->ComputeL2Error(pE_coeff);

    return {errvL2, vEL2, errpL2, pEL2};
}

//! Initializes vorticity and velocity component functions
void IncompNSObserver ::
init (std::shared_ptr<IncompNSFEM>& discr)
{
    m_discr = discr;
    m_vfes = m_discr->get_fespaces()[0];
    m_sfes = m_discr->get_fespaces()[1];

    auto ess_bdr_marker = m_discr->get_ess_bdr_marker();

    m_vort = std::make_unique<VorticityFunction>
            (m_sfes, m_vfes);
    m_vel = std::make_unique<VelocityFunction>
            (m_sfes, m_vfes);
}

//! Visualizes velocity components
void IncompNSObserver ::
visualize_velocities (std::shared_ptr<GridFunction>& v)
{
    if (m_bool_visualize)
    {
        std::shared_ptr <GridFunction> vx
                = std::make_shared<GridFunction>(m_sfes);
        std::shared_ptr <GridFunction> vy
                = std::make_shared<GridFunction>(m_sfes);
        (*m_vel)(*v, *vx, *vy);
        (*this)(vx);
        (*this)(vy);
    }
}

//! Visualizes vorticity
void IncompNSObserver ::
visualize_vorticity (std::shared_ptr<GridFunction>& v)
{
    if (m_bool_visualize)
    {
        std::shared_ptr <GridFunction> w
                = std::make_shared<GridFunction>(m_sfes);
        (*m_vort)(*v, *w);
        (*this)(w);
    }
}

//! Dumps solution
void IncompNSObserver
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

// End of file
