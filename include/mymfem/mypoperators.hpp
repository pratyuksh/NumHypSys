#ifndef MYMFEM_POPERATORS_HPP
#define MYMFEM_POPERATORS_HPP

#include "mfem.hpp"
using namespace mfem;

#include "putilities.hpp"


namespace mymfem {

//! Class ParMeanFreePressureOp
//! aplplies the mean-free constraint to an input pressure
class ParMeanFreePressureOp : public Operator
{
public:
    //! Constructors
    ParMeanFreePressureOp() {}
    
    ParMeanFreePressureOp(ParFiniteElementSpace *pfes)
    {
        MPI_Comm_rank(pfes->GetComm(), &m_myrank);

        m_meanFreePressure = new ParMeanFreePressure(pfes);
        m_area = m_meanFreePressure->get_area();
        m_lfone = m_meanFreePressure->get_lfone();
        m_massInvLfone
                = m_meanFreePressure->get_massInvLfone();

        height = m_lfone->Size();
        width = m_lfone->Size();
    }
    
    ParMeanFreePressureOp(ParFiniteElementSpace *pfes,
                          Operator* op)
    {
        m_meanFreePressure = new ParMeanFreePressure(pfes);
        m_area = m_meanFreePressure->get_area();
        m_lfone = m_meanFreePressure->get_lfone();
        m_massInvLfone
                = m_meanFreePressure->get_massInvLfone();

        height = m_lfone->Size();
        width = m_lfone->Size();
        SetOperator(op);
    }

    //! Destructor
    ~ParMeanFreePressureOp()
    {
        if (m_meanFreePressure) { delete m_meanFreePressure; }
    }
    
    //! Sets the external operator
    void SetOperator(Operator* op) {
        m_extOp = op;
    }
    
    //! Applies the mean-free pressure operator
    void Mult(const Vector &x, Vector& y) const override;

private:
    int m_myrank;
    double m_area;
    ParMeanFreePressure *m_meanFreePressure = nullptr;
    HypreParVector *m_lfone = nullptr;
    HypreParVector *m_massInvLfone = nullptr;

    Operator *m_extOp = nullptr;
};

}

#endif /// MYMFEM_OPERATORS_HPP
