#ifndef STOKES_DISCRETISATION_HPP
#define STOKES_DISCRETISATION_HPP

#include "mfem.hpp"
using namespace mfem;

#include <iostream>

#include "../mymfem/mybilinearform.hpp"
#include "../mymfem/mymixedbilinearform.hpp"

#include "../core/config.hpp"
#include "test_cases.hpp"
#include "coefficients.hpp"
#include "assembly.hpp"


class StokesFEM
{
public:
    StokesFEM (const nlohmann::json&);

    ~StokesFEM ();
    
    void set(std::shared_ptr<Mesh>&);
    void assemble_system();
    
    void assemble_rhs(BlockVector*);
    
    void assemble_preconditioner();
    
    void build_system_op();
    void build_system_matrix();

private:

    void assemble_source(Vector&);
    void assemble_bdry(Vector&);

    void apply_BCs(SparseMatrix&) const;
    void apply_BCs(Vector&) const;
    void apply_BCs(BlockVector*) const;

public:
    inline std::shared_ptr<StokesTestCases> get_test_case() const {
        return m_testCase;
    }
    
    inline Mesh* get_mesh() const {
        return m_fespaces[0]->GetMesh();
    }

    inline Array<FiniteElementCollection*> get_fecs() const {
        return m_fecs;
    }
    
    inline Array<FiniteElementSpace*> get_fespaces() const {
        return m_fespaces;
    }

    inline SparseMatrix* get_diffusion_matrix() const {
        return m_diffusion;
    }
    
    inline SparseMatrix* get_divu_matrix() const {
        return m_div;
    }
    
    inline BlockOperator* get_stokes_op() const {
        return m_stokesOp;
    }

    inline BlockDiagonalPreconditioner* get_stokes_pr() const {
        return m_stokesPr;
    }
    
    inline Array<int> get_block_offsets() const {
        return m_block_offsets;
    }

    inline SparseMatrix* get_stokes_mat() const {
        return m_stokesMat;
    }
    
    inline void print() const {
        std::cout << "Degree in x: " << m_deg << std::endl;
    }

private:
    const nlohmann::json& m_config;
    
    int m_ndim, m_deg;
    std::shared_ptr<StokesTestCases> m_testCase;
    
    Array<FiniteElementCollection*> m_fecs;
    Array<FiniteElementSpace*> m_fespaces;
    Array<int> m_block_offsets;
    
    double m_penalty;
    
    SparseMatrix *m_diffusion = nullptr;
    SparseMatrix *m_div = nullptr;
    SparseMatrix *m_divT = nullptr;
    SparseMatrix *m_openBdry = nullptr;

    GridFunction *m_vBdry = nullptr;
    BlockVector *m_rhsDirichlet = nullptr;

    BlockOperator *m_stokesOp = nullptr;
    BlockDiagonalPreconditioner *m_stokesPr = nullptr;
    
    Array<int> m_ess_bdr_marker;
    Array<int> m_nat_bdr_marker;
    Array<int> m_ess_tdof_list;
    
    SparseMatrix *m_stokesMat = nullptr;
};


#endif /// STOKES_DISCRETISATION_HPP
