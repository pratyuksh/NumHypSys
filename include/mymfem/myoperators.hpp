#ifndef MYMFEM_OPERATORS_HPP
#define MYMFEM_OPERATORS_HPP

#include "mfem.hpp"
using namespace mfem;

#include "utilities.hpp"


namespace mymfem {

//! Class MeanFreePressureOp
//! aplplies the mean-free constraint to an input pressure
class MeanFreePressureOp : public Operator
{
public:
    //! Constructors
    MeanFreePressureOp() {}
    
    MeanFreePressureOp(FiniteElementSpace *fes)
    {
        m_meanFreePressure = new MeanFreePressure(fes);
        m_area = m_meanFreePressure->get_area();
        m_lfone = m_meanFreePressure->get_lfone();
        m_massInvLfone
                = m_meanFreePressure->get_massInvLfone();

        height = m_lfone.Size();
        width = m_lfone.Size();
    }
    
    MeanFreePressureOp(FiniteElementSpace *fes, Operator* op)
    {
        m_meanFreePressure = new MeanFreePressure(fes);
        m_area = m_meanFreePressure->get_area();
        m_lfone = m_meanFreePressure->get_lfone();
        m_massInvLfone
                = m_meanFreePressure->get_massInvLfone();

        height = m_lfone.Size();
        width = m_lfone.Size();
        SetOperator(op);
    }

    //! Destructor
    ~MeanFreePressureOp() {
        if (m_meanFreePressure) { delete m_meanFreePressure; }
    }
    
    //! Sets the external operator
    void SetOperator(Operator* op) {
        m_extOp = op;
    }
    
    //! Applies the mean-free pressure operator
    void Mult(const Vector &x, Vector& y) const override;

private:
    double m_area;

    MeanFreePressure *m_meanFreePressure = nullptr;
    Vector m_lfone;
    Vector m_massInvLfone;

    Operator *m_extOp = nullptr;
};

}

#endif /// MYMFEM_OPERATORS_HPP
