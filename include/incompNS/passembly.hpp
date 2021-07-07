#ifndef INCOMPNS_PASSEMBLY_HPP
#define INCOMPNS_PASSEMBLY_HPP

#include "assembly.hpp"
#include "../mymfem/utilities.hpp"

#include "mfem.hpp"
using namespace mfem;


namespace mymfem {

#if MFEM_VERSION == 40200
//! Matrix-free application of
//! convection + upwind numerical flux operators
class ApplyParMFConvNFluxOperator
        : public ApplyMFConvNFluxOperator
{
public:
    Vector operator()(ParGridFunction *u,
                      ParGridFunction *w) const;

    void AssembleSharedFaces(ParGridFunction *u,
                             ParGridFunction *w,
                             Vector& rhs) const;

    void get_face_nbr_dofs(Array<int>& vdofs,
                           Vector &el_data) const
    {
        m_face_nbr_data.GetSubVector(vdofs, el_data);
    }

private:
    mutable Vector m_face_nbr_data;
};
#endif

}

#endif /// INCOMPNS_PASSEMBLY_HPP
