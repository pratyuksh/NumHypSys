#include "../../include/core/base_pobserver.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;


//! Constructor
BaseParObserver
:: BaseParObserver (MPI_Comm comm,
                    const nlohmann::json& config, int lx)
    : m_comm(comm)
{
    MPI_Comm_rank(m_comm, &m_myrank);
    MPI_Comm_size(m_comm, &m_nprocs);

    m_bool_visualize = false;
    if (config.contains("visualization")) {
        m_bool_visualize = config["visualization"];
    }

    m_bool_dumpOut = false;
    if (config.contains("dump_output")) {
        m_bool_dumpOut = config["dump_output"];
    }

    m_meshName_suffix = "_lx"+std::to_string(lx);
}

//! Visualizes grid function using GLVis
void BaseParObserver
:: visualize (std::shared_ptr<ParGridFunction>& u) const
{
    socketstream sout;
    if (m_bool_visualize)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;

        MPI_Barrier(u->ParFESpace()->GetParMesh()
                    ->GetComm());
        sout.open(vishost, visport);
        if (!sout)
        {
            if (m_myrank == 0) {
                cout << "Unable to connect to "
                        "GLVis server at "
                     << vishost << ':' << visport << endl;
            }
            m_bool_visualize = false;
            if (m_myrank == 0)
            {
                cout << "GLVis visualization disabled.\n";
            }
        }
        else
        {
            sout << "parallel " << m_nprocs
                 << " " << m_myrank << "\n";
            sout.precision(m_precision);
            sout << "solution\n"
                 << *u->ParFESpace()->GetParMesh() << *u;
            sout << "pause\n";
            sout << flush;
            if (m_myrank == 0) {
                cout << "GLVis visualization paused."
                     << " Press space (in the GLVis window) "
                        "to resume it.\n";
            }
      }
    }
}

//! Dumps mesh to file
void BaseParObserver
:: dump_mesh (std::shared_ptr<ParMesh>& pmesh) const
{
    if (m_bool_dumpOut)
    {
        /*ostringstream mesh_name;
        {
            mesh_name << (m_output_dir+"mesh"
                          +m_meshName_suffix+".")
                      << setfill('0') << setw(6) << m_myrank;
        }
        ofstream mesh_ofs(mesh_name.str().c_str());*/

        std::string mesh_name = m_output_dir
                +"pmesh"+m_meshName_suffix;
        std::cout << mesh_name << std::endl;

        std::ofstream mesh_ofs(mesh_name.c_str());
        mesh_ofs.precision(m_precision);
        pmesh->PrintAsOne(mesh_ofs);
        mesh_ofs.close();
    }
}


// End of file
