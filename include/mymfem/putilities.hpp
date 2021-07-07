#ifndef MYMFEM_PUTILITIES_HPP
#define MYMFEM_PUTILITIES_HPP

#include "mfem.hpp"
using namespace mfem;


//! Update and rebalances
void UpdateAndRebalance
(ParMesh* pmesh, ParFiniteElementSpace* fespace,
 ParGridFunction &x);

void UpdateAndRebalance
(ParMesh* pmesh, ParFiniteElementSpace* fespace,
 ParGridFunction &x1, ParGridFunction &x2);

void UpdateAndRebalance
(ParMesh* pmesh, ParFiniteElementSpace* fespace,
 ParGridFunction &x, ParBilinearForm* a, ParLinearForm* b);

//! Read dofs for Raviart-Thomas Spaces from MFEM
void get_dofs(const ParGridFunction &u,
              const Array<int> &vdofs,
              Vector &el_dofs);


//! Computes the area of a domain
double compute_area(const ParFiniteElementSpace&);


//! Velocity components function
class ParVelocityFunction
{
public:
    //! Constructors
    ParVelocityFunction(ParFiniteElementSpace *sfes,
                        ParFiniteElementSpace *vfes)
        : m_sfes(sfes), m_vfes(vfes)
    {
        init();
    }

    ParVelocityFunction(ParFiniteElementSpace *sfes,
                        ParFiniteElementSpace *vfes,
                        Array<int> ess_bdr_marker)
        : m_sfes(sfes), m_vfes(vfes),
          m_ess_bdr_marker (ess_bdr_marker)
    {
        init();
    }

    //! Destructor
    ~ParVelocityFunction () {
        if (m_mass) { delete m_mass; }
        if (m_xProj) { delete m_xProj; }
        if (m_yProj) { delete m_yProj; }
        if (m_cgsolver) { delete m_cgsolver; }
        if (m_amg) { delete m_amg; }
    }

    //! Initializes components
    void init();

    //! Evaluates x-component and y-component of velocity
    void operator()(const ParGridFunction& v,
                    ParGridFunction& vx,
                    ParGridFunction& vy) const;

    //! Evaluates x-component of velocity
    Vector eval_xComponent(const ParGridFunction *v);

    //! Evaluates y-component of velocity
    Vector eval_yComponent(const ParGridFunction *v);

    //! Evaluates x-component and y-component of velocity
    std::pair<Vector, Vector> eval(const ParGridFunction *v);

private:
    ParFiniteElementSpace *m_sfes = nullptr;
    ParFiniteElementSpace *m_vfes = nullptr;
    Array<int> m_ess_bdr_marker;

    HypreParMatrix *m_mass = nullptr;
    HypreParMatrix *m_xProj = nullptr;
    HypreParMatrix *m_yProj = nullptr;

    HyprePCG *m_cgsolver = nullptr;
    HypreBoomerAMG *m_amg = nullptr;

    Vector m_xComp;
    Vector m_yComp;
};

//! Vorticity function
class ParVorticityFunction
{
public:
    //! Constructors
    ParVorticityFunction(ParFiniteElementSpace *sfes,
                         ParFiniteElementSpace *vfes)
        : m_sfes(sfes), m_vfes(vfes)
    {
        init();
    }

    ParVorticityFunction(ParFiniteElementSpace *sfes,
                         ParFiniteElementSpace *vfes,
                         Array<int> ess_bdr_marker)
        : m_sfes(sfes), m_vfes(vfes),
          m_ess_bdr_marker (ess_bdr_marker)
    {
        init();
    }

    //! Destructor
    ~ParVorticityFunction () {
        if (m_mass) { delete m_mass; }
        if (m_proj) { delete m_proj; }
        if (m_cgsolver) { delete m_cgsolver; }
        if (m_amg) { delete m_amg; }
    }

    //! Initializes components
    void init();

    //! Evaluates vorticity
    void operator()(const ParGridFunction& v,
                    ParGridFunction& w) const;

private:
    ParFiniteElementSpace *m_sfes = nullptr;
    ParFiniteElementSpace *m_vfes = nullptr;
    Array<int> m_ess_bdr_marker;

    HypreParMatrix *m_mass = nullptr;
    HypreParMatrix *m_proj = nullptr;

    HyprePCG *m_cgsolver = nullptr;
    HypreBoomerAMG *m_amg = nullptr;
};


//! Class ParMeanFreePressure
//! makes the L2 pressure function mean free
class ParMeanFreePressure
{
public:
     //! Constructor
    ParMeanFreePressure (ParFiniteElementSpace *pfes)
    {
        MPI_Comm_rank(pfes->GetComm(), &m_myrank);
        set(pfes);
    }

    //! Applies the mean-free constraint
    //! to the input grid function
    void operator()(ParGridFunction&) const;

    //! Sets the FE space to be considered
    //! and the corresponding mass matrix and linear solver
    void set(ParFiniteElementSpace *pfes) const;

    //! Returns mean of the given grid function
    double get(ParGridFunction& pressure) const;

    //! Returns area of the FE domain
    double get_area() const {
        return m_area;
    }

    //! Returns the precomputed quantities
    HypreParVector* get_lfone() const {
        return m_lfone;
    }

    HypreParVector* get_massInvLfone() const {
        return m_massInvLfone;
    }

private:
    int m_myrank;
    mutable double m_area;
    mutable HypreParVector *m_lfone = nullptr;
    mutable HypreParVector *m_massInvLfone = nullptr;
};


//! Class ParPointLocator
//! maps a given set of physical points to the elements
//! of a mesh and also finds them in reference coordinates
//! wrt to those elements
class ParPointLocator
{
public:
    //! Constructor
    ParPointLocator (ParMesh *pmesh)
        : m_pmesh (pmesh) {}

    //! Maps a set of physical points
    //! to elements of the mesh
    std::pair <int, IntegrationPoint> operator()
    (const Vector&) const;

private:
    ParMesh *m_pmesh = nullptr;
};


namespace mymfem
{

//! Creates a monolithic HypreParMatrix from a block
HypreParMatrix * HypreParMatrixFromBlocks
(Array2D<HypreParMatrix *> &blocks,
 Array2D<double> *blockCoeff=nullptr);

void GatherBlockOffsetData
(MPI_Comm comm, const int rank, const int nprocs,
 const int num_loc, const Array<int> &offsets,
 std::vector<int> &all_num_loc, const int numBlocks,
 std::vector<std::vector<HYPRE_Int>> &blockProcOffsets,
 std::vector<HYPRE_Int> &procOffsets,
 std::vector<std::vector<int>> &procBlockOffsets,
 HYPRE_Int &firstLocal, HYPRE_Int &globalNum);

}


#endif /// MYMFEM_PUTILITIES_HPP
