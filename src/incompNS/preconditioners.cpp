#include "../../include/incompNS/discretisation.hpp"
#include "../../include/incompNS/operators.hpp"
#include "../../include/mymfem/myoperators.hpp"
#include <fmt/format.h>


//! Initializes preconditioner
void IncompNSFEM :: init_preconditioner
(const std::string precond_type)
{
    if (precond_type == "none") {
        return;
    }

    // allocate memory, if block diagonal preconditioner
    if (!m_incompNSPr && precond_type != "type4")
    {
        m_incompNSPr = new BlockDiagonalPreconditioner
                (m_block_offsets);
    }

    // allocate memory, if block triangular preconditioner
    if (!m_incompNSPr && precond_type == "type4")
    {
        m_incompNSPr = new mymfem::IncompNSBlockTriPr
                (m_block_offsets);
    }

    // initialize components
    // for type1 and type2 preconditioners
    if (precond_type == "type1")
    {
        if (!m_blockPr00)
        {
            auto dimR = m_fespaces[0]->GetTrueVSize();
            m_blockPr00 = new IdentityOperator(dimR);
        }

        // set preconditioner diagonal block 0
        auto blockDiagPr
                = static_cast
                <BlockDiagonalPreconditioner*>
                (m_incompNSPr);
        blockDiagPr->SetDiagonalBlock(0, m_blockPr00);
    }
    else if (precond_type == "type2")
    {
        if (!m_blockPr00)
        {
            m_blockPr00 = new GSSmoother();
            static_cast<Solver*>(m_blockPr00)
                    ->iterative_mode = false;
        }
    }
    if (precond_type == "type1" ||
            precond_type == "type2")
    {
        if (!m_blockPr11)
        {
            if (m_bool_mean_free_pressure) {
                m_blockPr11 = new mymfem::MeanFreePressureOp
                        (m_fespaces[1]);
            } else {
                auto dimW = m_fespaces[1]->GetTrueVSize();
                m_blockPr11 = new IdentityOperator(dimW);
            }
        }

        // set preconditioner diagonal block 1
        auto blockDiagPr
                = static_cast
                <BlockDiagonalPreconditioner*>
                (m_incompNSPr);
        blockDiagPr->SetDiagonalBlock(1, m_blockPr11);
    }

    // initialize components
    // for type3 and type4 preconditioners
    if (precond_type == "type3" ||
             precond_type == "type4")
    {
        if (!m_invMatS)
        {
            m_invMatS = new GSSmoother();
            m_invMatS->iterative_mode = false;
        }
        if (!m_blockPr11 && m_bool_mean_free_pressure)
        {
            m_blockPr11 = new mymfem::MeanFreePressureOp
                    (m_fespaces[1]);
        }
    }
}

//! Updates preconditioner
void IncompNSFEM :: update_preconditioner
(const std::string precond_type, const double)
{
    if (precond_type == "none") {
        return;
    }

    if (!m_block00Mat) {
        throw std::runtime_error(fmt::format(
            "Block00 Matrix not assembled"));
    }

    auto blockDiagPr
            = dynamic_cast<BlockDiagonalPreconditioner*>
            (m_incompNSPr);

    auto blockTriPr
            = dynamic_cast<mymfem::IncompNSBlockTriPr*>
            (m_incompNSPr);

    // type 1 preconditioner
    if (precond_type == "type1") {
        // Nothing to do for type1
    }
    // type 2 preconditioner
    else if (precond_type == "type2")
    {
        // set preconditioner diagonal block 0
        static_cast<Solver*>(m_blockPr00)
                ->SetOperator(*m_block00Mat);
        blockDiagPr->SetDiagonalBlock(0, m_blockPr00);
    }
    // type 3 preconditioner
    else if (precond_type == "type3")
    {
        if (m_blockPr00)
        {
            delete m_blockPr00;
            m_blockPr00 = nullptr;
        }
        if (m_matS)
        {
            delete m_matS;
            m_matS = nullptr;
        }

        SparseMatrix *block00InvMultDivT
                = Transpose(*m_div);
        m_block00Diag.SetSize(m_block00Mat->Height());

        m_block00Mat->GetDiag(m_block00Diag);
        for (int i = 0; i < m_block00Diag.Size(); i++) {
            block00InvMultDivT->
                    ScaleRow(i, 1./m_block00Diag(i));
        }
        m_matS = Mult(*m_div, *block00InvMultDivT);
        delete block00InvMultDivT;

        // update operators
        m_invMatS->SetOperator(*m_matS);
        m_blockPr00 = new DSmoother(*m_block00Mat);
        static_cast<Solver*>(m_blockPr00)
                ->iterative_mode = false;

        // update diagonal preconditioner blocks
        blockDiagPr->SetDiagonalBlock(0, m_blockPr00);
        if (m_bool_mean_free_pressure) {
            static_cast<mymfem::MeanFreePressureOp *>
                    (m_blockPr11)->SetOperator(m_invMatS);
            blockDiagPr->SetDiagonalBlock(1, m_blockPr11);
        } else {
            blockDiagPr->SetDiagonalBlock(1, m_invMatS);
        }
    }
    // type 4 preconditioner
    else if (precond_type == "type4")
    {
        if (m_blockPr00)
        {
            delete m_blockPr00;
            m_blockPr00 = nullptr;
        }
        if (m_matS)
        {
            delete m_matS;
            m_matS = nullptr;
        }

        SparseMatrix* block00InvMultDivT
                = Transpose(*m_div);
        m_block00Diag.SetSize(m_block00Mat->Height());

        m_block00Mat->GetDiag(m_block00Diag);
        for (int i = 0; i < m_block00Diag.Size(); i++) {
            block00InvMultDivT->
                    ScaleRow(i, 1./m_block00Diag(i));
        }
        m_matS = Mult(*m_div, *block00InvMultDivT);
        delete block00InvMultDivT;

        // update operators
        m_invMatS->SetOperator(*m_matS);
        m_blockPr00 = new DSmoother(*m_block00Mat);
        static_cast<Solver*>(m_blockPr00)
                ->iterative_mode = false;

        // update triangular preconditioner blocks
        if (m_bool_mean_free_pressure) {
            blockTriPr->set(m_block01Mat,
                            m_invMatS,
                            m_blockPr00,
                            m_blockPr11);
        } else {
            blockTriPr->set(m_block01Mat,
                            m_invMatS,
                            m_blockPr00);
        }
    }
    else
    {
        throw std::runtime_error(fmt::format(
            "Unknown preconditioner type for "
            "incompressible Navier-Stokes equations. [{}]",
                                     precond_type));
    }
}

