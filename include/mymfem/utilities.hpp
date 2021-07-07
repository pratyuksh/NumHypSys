#ifndef MYMFEM_UTILITIES_HPP
#define MYMFEM_UTILITIES_HPP

#include "mfem.hpp"
using namespace mfem;

#include <memory>


//! Zero function
void zeroFn(const Vector&, Vector&);

//! Returns cell center of element i of given mesh
void get_element_center(std::shared_ptr<Mesh>& mesh,
                        int i, Vector& center);

//! Computes inner product of two dense matrices
double MatMatInnerProd(const DenseMatrix &A,
                       const DenseMatrix &B);

//! Returns the upper-triangular, including the diagonal,
//! for the input matrix A
SparseMatrix& get_upper_triangle(const SparseMatrix& A);

//! Computes the area of a domain
double compute_area(const FiniteElementSpace&);

//! Computes the mean of a function
double compute_mean(const GridFunction &);

//! Reads dofs for Raviart-Thomas Spaces from MFEM
void get_dofs(const GridFunction &u,
              const Array<int> &vdofs,
              Vector &el_dofs);

void get_dofs(const Vector &u,
              const Array<int> &vdofs,
              Vector &el_dofs);

//! Projects the normal component at the boundary
//! for the given function
void project_bdry_coefficient_normal
(GridFunction &u, VectorCoefficient &vCoeff,
 const Array<int> &bdr_attr);


//! Velocity components function
class VelocityFunction
{
public:
    //! Constructors
    VelocityFunction(FiniteElementSpace *sfes,
                     FiniteElementSpace *vfes)
        : m_sfes(sfes), m_vfes(vfes)
    {
        init();
    }

    VelocityFunction(FiniteElementSpace *sfes,
                     FiniteElementSpace *vfes,
                     Array<int> ess_bdr_marker)
        : m_sfes(sfes), m_vfes(vfes),
          m_ess_bdr_marker (ess_bdr_marker)
    {
        init();
    }

    //! Destructor
    ~VelocityFunction () {
        if (m_mass) { delete m_mass; }
        if (m_xProj) { delete m_xProj; }
        if (m_yProj) { delete m_yProj; }
        if (m_cgsolver) { delete m_cgsolver; }
    }

    //! Initializes components
    void init();

    //! Evaluates x-component and y-component of velocity
    void operator()(const GridFunction& v,
                    GridFunction& vx,
                    GridFunction& vy) const;

    //! Evaluates x-component of velocity
    Vector eval_xComponent(const GridFunction *v);

    //! Evaluates y-component of velocity
    Vector eval_yComponent(const GridFunction *v);

    //! Evaluates x-component and y-component of velocity
    std::pair<Vector, Vector> eval(const GridFunction *v);

private:
    FiniteElementSpace *m_sfes = nullptr;
    FiniteElementSpace *m_vfes = nullptr;
    Array<int> m_ess_bdr_marker;

    SparseMatrix *m_mass = nullptr;
    SparseMatrix *m_xProj = nullptr;
    SparseMatrix *m_yProj = nullptr;

    CGSolver *m_cgsolver = nullptr;

    Vector m_xComp;
    Vector m_yComp;
};

//! Vorticity function
class VorticityFunction
{
public:
    //! Constructors
    VorticityFunction(FiniteElementSpace *sfes,
                      FiniteElementSpace *vfes)
        : m_sfes(sfes), m_vfes(vfes)
    {
        init();
    }

    VorticityFunction(FiniteElementSpace *sfes,
                      FiniteElementSpace *vfes,
                      Array<int> ess_bdr_marker)
        : m_sfes(sfes), m_vfes(vfes),
          m_ess_bdr_marker (ess_bdr_marker)
    {
        init();
    }

    //! Destructor
    ~VorticityFunction () {
        if (m_mass) { delete m_mass; }
        if (m_proj) { delete m_proj; }
        if (m_cgsolver) { delete m_cgsolver; }
    }

    //! Initializes components
    void init();

    //! Evaluates vorticity
    void operator()(const GridFunction& v,
                    GridFunction& w) const;

private:
    FiniteElementSpace *m_sfes = nullptr;
    FiniteElementSpace *m_vfes = nullptr;
    Array<int> m_ess_bdr_marker;

    SparseMatrix *m_mass = nullptr;
    SparseMatrix *m_proj = nullptr;

    CGSolver *m_cgsolver = nullptr;
};


//! Class MeanFreePressure
//! makes the L2 pressure function mean free
class MeanFreePressure
{
public:
    //! Constructor
    MeanFreePressure (FiniteElementSpace *fes) {
        set(fes);
    }

    //! Applies the mean-free constraint
    //! to the input grid function
    void operator()(GridFunction&) const;

    //! Sets the FE space to be considered
    //! and the corresponding mass matrix and linear solver
    void set(FiniteElementSpace *fes) const;

    //! Returns mean of the given grid function
    double get(GridFunction& pressure) const;

    //! Returns area of the FE domain
    double get_area() const {
        return m_area;
    }

    //! Returns the precomputed quantities
    Vector& get_lfone() const {
        return m_lfone;
    }

    Vector& get_massInvLfone() const {
        return m_massInvLfone;
    }

private:
    mutable double m_area;
    mutable Vector m_lfone;
    mutable Vector m_massInvLfone;
};


//! Class PointLocator
//! maps a given set of physical points to the elements
//! of a mesh and also finds them in reference coordinates
//! wrt to those elements
class PointLocator
{
public:
    //! Constructor
    PointLocator (Mesh *mesh, bool has_shared_vertices=false)
        : m_has_shared_vertices (has_shared_vertices),
          m_mesh (mesh)
    {
        m_invTr = std::make_unique
                <InverseElementTransformation>();

        if (!m_has_shared_vertices) {
            m_vToEl.reset(m_mesh->GetVertexToElementTable());
        }
        else {
            m_shared_vertices = std::make_unique<Table>();
            get_shared_vertices_table();

            m_vToEl = std::make_unique<Table>();
            get_vertices_to_elements_table();
        }
    }

    //! Maps a set of physical points
    //! to elements of the mesh
    std::pair <int, IntegrationPoint> operator()
    (const Vector&, const int) const;

    //! Maps a physical point
    //! to an element of the mesh
    int operator() (const Vector&, const int,
                    int&, IntegrationPoint&) const;

    //! Generates a table of vertices,
    //! which are shared by different processors.
    //! Needed when a ParMesh is written to one file,
    //! the same vertices at the partition interfaces
    //! will have a different numbering for each processor
    void get_shared_vertices_table () const;

    //! Generates the vertex to element table,
    //! uses the shared vertices table
    void get_vertices_to_elements_table () const;

    //! Computes the reference coordinates
    //! of a physical point in given element
    inline std::pair <int, IntegrationPoint>
    compute_ref_point (const Vector& x, const int elId) const
    {
        IntegrationPoint ip;
        m_invTr->SetTransformation
                (*m_mesh->GetElementTransformation(elId));
        auto info = m_invTr->Transform(x, ip);
        return {info, ip};
    }

private:
    double m_TOL = 1E-12;
    bool m_has_shared_vertices = false;

    Mesh *m_mesh = nullptr;
    std::unique_ptr<InverseElementTransformation> m_invTr;

    std::unique_ptr<Table> m_vToEl;
    std::unique_ptr<Table> m_shared_vertices;
};


#endif /// MYMFEM_UTILITIES_HPP
