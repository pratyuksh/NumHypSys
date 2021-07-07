#include "../../include/incompNS/pobserver.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>

#include "../../include/mymfem/utilities.hpp"

namespace fs = std::filesystem;


//! Constructors
IncompNSParObserver
:: IncompNSParObserver (MPI_Comm comm,
                        const nlohmann::json& config,
                        int lx)
    : BaseParObserver(comm, config, lx)
{
    std::string problem_type = config["problem_type"];

    m_bool_error = false;
    if (config.contains("eval_error")) {
        m_bool_error = config["eval_error"];
    }

    std::string base_out_dir = "../output";
    if (config.contains("base_out_dir")) {
        base_out_dir = config["base_out_dir"];
    }

    std::string sub_out_dir = "data/incompNS_"+problem_type;
    if (config.contains("sub_out_dir")) {
        sub_out_dir = config["sub_out_dir"];
    }

    m_output_dir = base_out_dir+"/"+sub_out_dir+"/lx"
            +std::to_string(lx)+"/";
    if (m_bool_dumpOut) {
        fs::create_directories(m_output_dir);
    }
    m_solName_suffix = "_lx"+std::to_string(lx);

    m_dump_out_nsteps = 50;
    if (config.contains("dump_output_num_time_steps")) {
        m_dump_out_nsteps
                = config["dump_output_num_time_steps"];
    }
}

IncompNSParObserver
:: IncompNSParObserver
(MPI_Comm comm, const nlohmann::json& config,
 std::shared_ptr<IncompNSTestCases>& testCase, int lx)
    : IncompNSParObserver(comm, config, lx)
{
    m_testCase = testCase;
}

//! Computes L2 error for velocity and pressure
std::tuple <double, double, double, double>
IncompNSParObserver
:: eval_error_L2 (const double t,
                  std::shared_ptr<ParGridFunction>& v,
                  std::shared_ptr<ParGridFunction>& p) const
{
    double errvL2=0, errpL2=0;
    double vEL2=0, pEL2=0;

    int dim = p->ParFESpace()->GetParMesh()->Dimension();
    ConstantCoefficient zero(0);
    VectorFunctionCoefficient zeroVec(dim, zeroFn);

    // error in velocity
    IncompNSExactVelocityCoeff vE_coeff(m_testCase);
    vE_coeff.SetTime(t);
    ParGridFunction vE(v->ParFESpace());
    vE.ProjectCoefficient(vE_coeff);
    vEL2 = vE.ComputeL2Error(zeroVec);
    errvL2 = v->ComputeL2Error(vE_coeff);

    // error in pressure
    IncompNSExactPressureCoeff pE_coeff(m_testCase);
    pE_coeff.SetTime(t);
    ParGridFunction pE(p->ParFESpace());
    pE.ProjectCoefficient(pE_coeff);
    pEL2 = pE.ComputeL2Error(zero);
    errpL2 = p->ComputeL2Error(pE_coeff);

    return {errvL2, vEL2, errpL2, pEL2};
}

//! Initializes vorticity and velocity component functions
void IncompNSParObserver ::
init (std::shared_ptr<IncompNSParFEM>& discr)
{
    m_discr = discr;
    m_vfes = m_discr->get_fespaces()[0];
    m_sfes = m_discr->get_fespaces()[1];

    auto ess_bdr_marker = m_discr->get_ess_bdr_marker();

    m_vel = std::make_unique<ParVelocityFunction>
            (m_sfes, m_vfes);
    m_vort = std::make_unique<ParVorticityFunction>
            (m_sfes, m_vfes);
}

//! Visualizes velocity components
void IncompNSParObserver ::
visualize_velocities (std::shared_ptr<ParGridFunction>& v)
{
    if (m_bool_visualize)
    {
        std::shared_ptr <ParGridFunction> vx
                = std::make_shared<ParGridFunction>(m_sfes);
        std::shared_ptr <ParGridFunction> vy
                = std::make_shared<ParGridFunction>(m_sfes);
        (*m_vel)(*v, *vx, *vy);
        (*this)(vx);
        (*this)(vy);
    }
}

//! Visualizes vorticity
void IncompNSParObserver ::
visualize_vorticity (std::shared_ptr<ParGridFunction>& v)
{
    if (m_bool_visualize)
    {
        std::shared_ptr <ParGridFunction> w
                = std::make_shared<ParGridFunction>(m_sfes);
        (*m_vort)(*v, *w);
        (*this)(w);
    }
}

//! Dumps solution
void IncompNSParObserver
:: dump_sol (std::shared_ptr<ParGridFunction>& v,
             std::shared_ptr<ParGridFunction>& p) const
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
        v->SaveAsOne(sol_ofs1);

        std::ofstream sol_ofs2(sol_name2.c_str());
        sol_ofs2.precision(m_precision);
        p->SaveAsOne(sol_ofs2);
    }
}

//! Dump solution at given time ids
void IncompNSParObserver
:: dump_sol (std::shared_ptr<ParGridFunction>& v,
             std::shared_ptr<ParGridFunction>& p,
             int time_step_id) const
{
    if (!m_bool_dumpOut) { return; }

    if (time_step_id % m_dump_out_nsteps == 0)
    {
        std::string solName_suffix = m_solName_suffix
                +"_tId_"+std::to_string(time_step_id);

        std::string sol_name1 =
                m_output_dir+"velocity"+solName_suffix;
        std::string sol_name2 =
                m_output_dir+"pressure"+solName_suffix;
        std::string sol_name3 =
                m_output_dir+"vorticity"+solName_suffix;
        std::string sol_name4 =
                m_output_dir+"vx"+solName_suffix;
        std::string sol_name5 =
                m_output_dir+"vy"+solName_suffix;
        if (m_myrank == 0) {
            std::cout << sol_name1 << std::endl;
            std::cout << sol_name2 << std::endl;
            std::cout << sol_name3 << std::endl;
            std::cout << sol_name4 << std::endl;
            std::cout << sol_name5 << std::endl;
        }

        // velocity
        std::ofstream sol_ofs1(sol_name1.c_str());
        sol_ofs1.precision(m_precision);
        v->SaveAsOne(sol_ofs1);
        sol_ofs1.close();

        // pressure
        std::ofstream sol_ofs2(sol_name2.c_str());
        sol_ofs2.precision(m_precision);
        p->SaveAsOne(sol_ofs2);
        sol_ofs2.close();

        // vorticity
        std::ofstream sol_ofs3(sol_name3.c_str());
        sol_ofs3.precision(m_precision);
        std::shared_ptr <ParGridFunction> w
                = std::make_shared<ParGridFunction>(m_sfes);
        (*m_vort)(*v, *w);
        w->SaveAsOne(sol_ofs3);
        sol_ofs3.close();

        // velocity components
        std::ofstream sol_ofs4(sol_name4.c_str());
        std::ofstream sol_ofs5(sol_name5.c_str());
        sol_ofs4.precision(m_precision);
        sol_ofs5.precision(m_precision);
        std::shared_ptr <ParGridFunction> vx
                = std::make_shared<ParGridFunction>(m_sfes);
        std::shared_ptr <ParGridFunction> vy
                = std::make_shared<ParGridFunction>(m_sfes);
        (*m_vel)(*v, *vx, *vy);
        vx->SaveAsOne(sol_ofs4);
        vy->SaveAsOne(sol_ofs5);
        sol_ofs4.close();
        sol_ofs5.close();
    }
}

//! Dump solution at given time ids
void IncompNSParObserver
:: dump_sol (std::shared_ptr<ParGridFunction>& v,
             std::shared_ptr<ParGridFunction>& p,
             int sampleId,
             int timeStepId) const
{
    if (!m_bool_dumpOut) { return; }

    if (timeStepId % m_dump_out_nsteps == 0)
    {
        std::string solName_suffix
                = "_s"+std::to_string(sampleId)
                +"_tId"+std::to_string(timeStepId);

        std::string sol_name1 =
                m_output_dir+"velocity"+solName_suffix;
        std::string sol_name2 =
                m_output_dir+"pressure"+solName_suffix;
        if (m_myrank == 0) {
            std::cout << sol_name1 << std::endl;
            std::cout << sol_name2 << std::endl;
        }

        // velocity
        std::ofstream sol_ofs1(sol_name1.c_str());
        sol_ofs1.precision(m_precision);
        v->SaveAsOne(sol_ofs1);
        sol_ofs1.close();

        // pressure
        std::ofstream sol_ofs2(sol_name2.c_str());
        sol_ofs2.precision(m_precision);
        p->SaveAsOne(sol_ofs2);
        sol_ofs2.close();
    }
}

// End of file
