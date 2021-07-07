#ifndef BASE_POBSERVER_HPP
#define BASE_POBSERVER_HPP

#include "mfem.hpp"
using namespace mfem;

#include "config.hpp"


//! Base Parallel Observer
class BaseParObserver
{
public:
    //! Constructors
    BaseParObserver (MPI_Comm) { m_bool_visualize = false; }
    
    BaseParObserver (MPI_Comm, const nlohmann::json&, int);

    //! Operator shortcut to call the visualize routine
    void operator () (std::shared_ptr<ParGridFunction>& u)
    const {
        visualize(u);
    }

    //! Sets the boolean for visualization
    void set_bool_visualize (bool bool_visualize) {
        m_bool_visualize = bool_visualize;
    }

    //! Visualizes Paralell Grid Function using GlVis
    void visualize (std::shared_ptr<ParGridFunction>&) const;

    //! Dumps mesh to file
    void dump_mesh (std::shared_ptr<ParMesh>&) const;
    
protected:
    MPI_Comm m_comm;
    int m_myrank, m_nprocs;
    int m_precision = 8;

    mutable bool m_bool_visualize;

    bool m_bool_dumpOut;
    mutable std::string m_output_dir;
    std::string m_meshName_prefix;
    std::string m_meshName_suffix;
};


#endif /// BASE_POBSERVER_HPP
