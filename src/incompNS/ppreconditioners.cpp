#include "../../include/incompNS/pdiscretisation.hpp"
#include "../../include/mymfem/mypoperators.hpp"
#include <fmt/format.h>


//! Initializes preconditioner
void IncompNSParFEM :: init_preconditioner
(const std::string precond_type)
{
    if (precond_type == "none") {
        return;
    }

    // allocate memory, if block diagonal preconditioner
    if (!m_incompNSPr &&
            !(precond_type == "type5"
             || precond_type == "type6"))
    {
        m_incompNSPr = new BlockDiagonalPreconditioner
                (m_block_trueOffsets);
    }

    // allocate memory, if block triangular preconditioner
    if (!m_incompNSPr &&
            (precond_type == "type5"
                         || precond_type == "type6"))
    {
        m_incompNSPr = new mymfem::IncompNSParBlockTriPr
                (m_block_trueOffsets);
    }

    // initialize components
    // for type1 and type2 preconditioners
    if (precond_type == "type1")
    {
        if (!m_blockPr00)
        {
            auto dimR = m_block_trueOffsets[1]
                    - m_block_trueOffsets[0];
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
            m_blockPr00 = new HypreBoomerAMG();
            static_cast<HypreBoomerAMG *>
                    (m_blockPr00)->iterative_mode = false;
            static_cast<HypreBoomerAMG *>
                    (m_blockPr00)->SetPrintLevel(0);
        }
    }
    if (precond_type == "type1" ||
            precond_type == "type2")
    {
        if (!m_blockPr11)
        {
            if (m_bool_mean_free_pressure) {
                m_blockPr11
                        = new mymfem::ParMeanFreePressureOp
                        (m_fespaces[1]);
            } else {
                auto dimW = m_block_trueOffsets[2]
                        - m_block_trueOffsets[1];
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
    // for type3, type4, type5 and type6 preconditioners
    if (precond_type == "type3" ||
            precond_type == "type5")
    {
        if (!m_invMatS)
        {
            m_invMatS = new HypreSmoother();
            m_invMatS->iterative_mode = false;
        }
    }
    else if (precond_type == "type4" ||
             precond_type == "type6")
    {
        if (!m_invMatS)
        {
            m_invMatS = new HypreBoomerAMG();
            static_cast<HypreBoomerAMG *>
                    (m_invMatS)->SetPrintLevel(0);
            HYPRE_BoomerAMGSetStrongThreshold
                    (*static_cast<HypreBoomerAMG *>
                     (m_invMatS), 0.30); // change this
        }
    }
    if (precond_type == "type3" ||
            precond_type == "type4" ||
            precond_type == "type5" ||
            precond_type == "type6")
    {
        if (!m_blockPr11 && m_bool_mean_free_pressure)
        {
            m_blockPr11
                    = new mymfem::ParMeanFreePressureOp
                    (m_fespaces[1]);
        }
    }
}

//! Updates preconditioner
void IncompNSParFEM :: update_preconditioner
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
            = dynamic_cast<mymfem::IncompNSParBlockTriPr*>
            (m_incompNSPr);

    // type 1 preconditioner
    if (precond_type == "type1")
    {
        // Nothing to do
    }
    // type 2 preconditioner
    else if (precond_type == "type2")
    {
        // set preconditioner diagonal block 0
        static_cast<HypreBoomerAMG *>
                (m_blockPr00)->SetOperator(*m_block00Mat);
        blockDiagPr->SetDiagonalBlock(0, m_blockPr00);
    }
    // type 3, 4, 5 or 6 preconditioner
    else if (precond_type == "type3" ||
             precond_type == "type4" ||
             precond_type == "type5" ||
             precond_type == "type6")
    {
        if (m_blockPr00) {
            delete m_blockPr00;
            m_blockPr00 = nullptr;
        }
        if (m_block00InvMultDivuT)
        {
            delete m_block00InvMultDivuT;
            m_block00InvMultDivuT = nullptr;
        }
        if (m_block00Diag) {
            delete m_block00Diag;
            m_block00Diag = nullptr;
        }
        if (m_matS) {
            delete m_matS;
            m_matS = nullptr;
        }

        m_block00InvMultDivuT = m_div->Transpose();
        m_block00Diag
                = new HypreParVector
                (m_block00Mat->GetComm(),
                 m_block00Mat->GetGlobalNumRows(),
                 m_block00Mat->GetRowStarts());
        m_block00Mat->GetDiag(*m_block00Diag);
        m_block00InvMultDivuT->InvScaleRows(*m_block00Diag);

        // set/update operators
        m_matS = ParMult(m_div, m_block00InvMultDivuT);
        m_invMatS->SetOperator(*m_matS);

        m_blockPr00 = new HypreDiagScale(*m_block00Mat);
        static_cast<HypreDiagScale *>
                (m_blockPr00)->iterative_mode = false;

        // update diagonal preconditioner blocks
        if (precond_type == "type3" ||
                     precond_type == "type4")
        {
            blockDiagPr->SetDiagonalBlock(0, m_blockPr00);
            if (m_bool_mean_free_pressure) {
                static_cast<mymfem::ParMeanFreePressureOp *>
                        (m_blockPr11)->SetOperator(m_invMatS);
                blockDiagPr->SetDiagonalBlock(1, m_blockPr11);
            } else {
                blockDiagPr->SetDiagonalBlock(1, m_invMatS);
            }
        }

        // update triangular preconditioner blocks
        if (precond_type == "type5" ||
                     precond_type == "type6")
        {
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
    }
    // type 7 preconditioner
    /*else if (precond_type == "type7")
    {
        if (m_blockPr00)
        {
            delete m_blockPr00;
            m_blockPr00 = nullptr;
        }
        // set preconditioner diagonal block 0
        m_blockPr00 = new HypreEuclid(*m_block00Mat);
        blockDiagPr->SetDiagonalBlock(0, m_blockPr00);
    }*/
    else
    {
        throw std::runtime_error(fmt::format(
            "Unknown preconditioner type for "
            "incompressible Navier-Stokes equations. [{}]",
                                     precond_type));
    }
}
