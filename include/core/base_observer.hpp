#ifndef BASE_OBSERVER_HPP
#define BASE_OBSERVER_HPP

#include "mfem.hpp"
using namespace mfem;

#include "config.hpp"


//! Base Observer
class BaseObserver
{
public:
    //! Constructors
    BaseObserver () { m_bool_visualize = false; }

    BaseObserver (const nlohmann::json&, int);
    
    //! Operator shortcuts to call the visualize routine
    void operator () (std::shared_ptr<GridFunction>& u)
    const {
        visualize(u);
    }

    void operator () (GridFunction* u) const {
        visualize(u);
    }

    //! Visualizes Grid Function using GlVis
    void visualize (std::shared_ptr<GridFunction>& u)
    const {
        visualize(u.get());
    }

    //! Sets the boolean for visualization
    void set_bool_visualize (bool bool_visualize) {
        m_bool_visualize = bool_visualize;
    }

    //! Visualizes grid function using GLVis
    void visualize (GridFunction *) const;

    //! Dumps mesh to file
    void dump_mesh (std::shared_ptr<Mesh>&) const;
    
protected:
    int m_precision = 8;

    mutable bool m_bool_visualize;
    
    bool m_bool_dumpOut;
    mutable std::string m_output_dir;
    std::string m_meshName_prefix;
    std::string m_meshName_suffix;
};


#endif /// BASE_OBSERVER_HPP
