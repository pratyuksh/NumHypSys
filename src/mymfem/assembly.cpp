#include "../../include/mymfem/assembly.hpp"
#include "../../include/mymfem/utilities.hpp"
#include <assert.h>


// Velocity x-component Projection Integrator
void mymfem::VelocityXProjectionIntegrator
:: AssembleElementMatrix2 (const FiniteElement &trial_fe,
                           const FiniteElement &test_fe,
                           ElementTransformation &Trans,
                           DenseMatrix &elmat)
{
    int dim = trial_fe.GetDim();
    int trial_nd = trial_fe.GetDof();
    int test_nd = test_fe.GetDof();

#ifdef MFEM_THREAD_SAFE
    DenseMatrix vshape(trial_nd, dim);
    Vector shape(test_nd);
#else
    vshape.SetSize(trial_nd, dim);
    shape.SetSize(test_nd);
#endif
    elmat.SetSize(test_nd, trial_nd);

    // x-components
    Vector nx(2);
    Vector vx(trial_nd);
    nx = 0.0;
    nx(0) = 1;

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
        int order = trial_fe.GetOrder()
                + test_fe.GetOrder();
        ir = &IntRules.Get(trial_fe.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);

        trial_fe.CalcVShape(Trans, vshape);
        test_fe.CalcShape(ip, shape);

        vshape.Mult(nx, vx);
        double w = ip.weight*Trans.Weight();

        shape *= w;
        for (int l=0; l<trial_nd; l++) {
            for (int k=0; k<test_nd; k++) {
                elmat(k, l) += shape(k)*vx(l);
            }
        }
    }
}

// Velocity y-component Projection Integrator
void mymfem::VelocityYProjectionIntegrator
:: AssembleElementMatrix2 (const FiniteElement &trial_fe,
                           const FiniteElement &test_fe,
                           ElementTransformation &Trans,
                           DenseMatrix &elmat)
{
    int dim = trial_fe.GetDim();
    int trial_nd = trial_fe.GetDof();
    int test_nd = test_fe.GetDof();

#ifdef MFEM_THREAD_SAFE
    DenseMatrix vshape(trial_nd, dim);
    Vector shape(test_nd);
#else
    vshape.SetSize(trial_nd, dim);
    shape.SetSize(test_nd);
#endif
    elmat.SetSize(test_nd, trial_nd);

    // y-component
    Vector ny(2);
    Vector vy(trial_nd);
    ny = 0.0;
    ny(1) = 1;

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
        int order = trial_fe.GetOrder()
                + test_fe.GetOrder();
        ir = &IntRules.Get(trial_fe.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);

        trial_fe.CalcVShape(Trans, vshape);
        test_fe.CalcShape(ip, shape);

        vshape.Mult(ny, vy);
        double w = ip.weight*Trans.Weight();

        shape *= w;
        for (int l=0; l<trial_nd; l++) {
            for (int k=0; k<test_nd; k++) {
                elmat(k, l) += shape(k)*vy(l);
            }
        }
    }
}

// Velocity to Vorticity Projection Integrator
void mymfem::VorticityProjectionIntegrator
:: AssembleElementMatrix2 (const FiniteElement &trial_fe,
                           const FiniteElement &test_fe,
                           ElementTransformation &Trans,
                           DenseMatrix &elmat)
{
    int dim = trial_fe.GetDim();
    int trial_nd = trial_fe.GetDof();
    int test_nd = test_fe.GetDof();

#ifdef MFEM_THREAD_SAFE
    DenseTensor gradvshape(dim, dim, trial_nd);
    Vector shape(test_nd);
#else
    gradvshape.SetSize(dim, dim, trial_nd);
    shape.SetSize(test_nd);
#endif
    elmat.SetSize(test_nd, trial_nd);

    const IntegrationRule *ir = IntRule;
    if (ir == nullptr)
    {
        int order = trial_fe.GetOrder()
                  + test_fe.GetOrder() - 1;
        ir = &IntRules.Get(trial_fe.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);

        trial_fe.CalcGradVShape(Trans, gradvshape);
        test_fe.CalcShape(ip, shape);
        double w = ip.weight*Trans.Weight();

        /*cout << "\nBasis Functions:\n";
        for (int k=0; k<trial_nd; k++) {
            for (int l1=0; l1<dim; l1++) {
                for (int l2=0; l2<dim; l2++)
                    cout << gradvshape(l1,l2,k) << " ";
                cout << "\n";
            }
            cout << "\n\n";
        }*/

        shape *= w;
        for (int l=0; l<trial_nd; l++) {
            for (int k=0; k<test_nd; k++) {
                elmat(k, l) += shape(k)*
                  (gradvshape(l)(1,0) - gradvshape(l)(0,1));
            }
        }
    }
    /*for (int l1=0; l1<test_nd; l1++) {
        for (int l2=0; l2<trial_nd; l2++) {
            cout << elmat(l1,l2) << " ";
        }
        cout << "\n";
    }
    cout << "\n\n";*/
}

// End of file
