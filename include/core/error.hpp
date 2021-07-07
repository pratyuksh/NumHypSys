#ifndef MYMFEM_ERROR_HPP
#define MYMFEM_ERROR_HPP

#include <memory>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "mfem.hpp"
#include "../mymfem/utilities.hpp"

using namespace mfem;


//! Class ComputeH1Error
//! computes the H1 Error between two grid functions
//! possible on different FE spaces.
class ComputeH1Error
{
public:
    //! Computes the H1 Error between two grid functions
    //! possible on different FE spaces.
    //! Uses PointLocator
    std::tuple <double, double, double, double>
    operator()(const GridFunction&, const GridFunction&,
               bool has_shared_vertices = false);

    //! Returns cell element ids and reference points for
    //! mesh2 in element of mesh1.
    //! Uses PointLocator
    std::pair < Array <int>, Array <IntegrationPoint> >
    get_ref_points (ElementTransformation& trans1,
                    int init_cell_id2);

    //! Computes the H1 Error between two grid functions
    //! possible on different FE spaces.
    //! Uses MFEM routines for the search
    std::tuple <double, double, double, double>
    test_slow(const GridFunction&, const GridFunction&);

    //! Returns cell element ids and reference points for
    //! mesh2 in element of mesh1.
    //! Uses MFEM routines for the search
    std::pair < Array <int>, Array <IntegrationPoint> >
    get_ref_points_slow (ElementTransformation&, Mesh&);

private:
    std::unique_ptr <const IntegrationRule> m_ir;
    std::unique_ptr <PointLocator> m_point_locator;
};


//! Class ComputeCauchyL1Error
//! computes the L1 Error between two grid functions
//! possible on different FE spaces.
class ComputeCauchyL1Error
{
public:
    //! Computes the L1 Error between two grid functions
    //! possible on different FE spaces.
    double operator()
    (const GridFunction&, const GridFunction&,
     bool has_shared_vertices = false);

    //! Returns cell element ids and reference points for
    //! mesh2 in element of mesh1.
    std::pair < Array <int>, Array <IntegrationPoint> >
    get_ref_points (ElementTransformation&,
                    int init_cell_id2);

    //! Computes the L1 Error between two grid functions
    //! defined on the same mesh
    double evalOnSameMesh
    (const GridFunction&, const GridFunction&);

private:
    std::unique_ptr <const IntegrationRule> m_ir;
    std::unique_ptr <PointLocator> m_point_locator;
};

//! Class ComputeCauchyL2Error
//! computes the L2 Error between two grid functions
//! possible on different FE spaces.
class ComputeCauchyL2Error
{
public:
    //! Computes the L2 Error between two grid functions,
    //! possible on different FE spaces.
    double operator()
    (const GridFunction&, const GridFunction&,
     bool has_shared_vertices = false);

    //! Computes the L2 Error between two grid functions,
    //! where one grid function is split into x-y components
    //! possible on different FE spaces.
    double operator()
    (const GridFunction&, const GridFunction&,
     const GridFunction&, const GridFunction&,
     bool has_shared_vertices = false);

    //! Returns cell element ids and reference points for
    //! mesh2 in element of mesh1.
    std::pair < Array <int>, Array <IntegrationPoint> >
    get_ref_points (ElementTransformation&,
                    int init_cell_id2);

    //! Computes the L2 Error between two grid functions
    //! defined on the same mesh
    double evalOnSameMesh
    (const GridFunction&, const GridFunction&);

    //! Computes the L2 Error between two grid functions,
    //! where one grid function is split into x-y components,
    //! defined on the same mesh
    double evalOnSameMesh
    (const GridFunction&, const GridFunction&,
     const GridFunction&, const GridFunction&);

private:
    std::unique_ptr <const IntegrationRule> m_ir;
    std::unique_ptr <PointLocator> m_point_locator;
};


#endif /// MYMFEM_ERROR_HPP
