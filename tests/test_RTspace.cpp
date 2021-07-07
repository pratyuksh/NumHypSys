#include <gtest/gtest.h>

#include "mfem.hpp"
using namespace mfem;

#include <iostream>
#include <fstream>
#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../include/core/config.hpp"


void eval_vshape_RT0_refEl(const double x, const double y,
                           DenseMatrix& vshape)
{
    int ndofs = 3;
    vshape.SetSize(ndofs, 2);
    vshape(0,0) = x  ; vshape(0,1) = y-1;
    vshape(1,0) = x  ; vshape(1,1) = y  ;
    vshape(2,0) = x-1; vshape(2,1) = y;
}

void eval_gradvshape_RT0_refEl(const double, const double,
                               DenseTensor& gradvshape)
{
    int ndofs = 3;
    gradvshape.SetSize(2,2,ndofs);
    gradvshape = 0;

    gradvshape(0,0,0) = gradvshape(1,1,0) = 1;
    gradvshape(0,0,1) = gradvshape(1,1,1) = 1;
    gradvshape(0,0,2) = gradvshape(1,1,2) = 1;
}

class RT1_refEl
{
public:
    RT1_refEl ()
    {
        double c = 1./3.;

        Vector xi(2);
        xi(0) = (1-1./sqrt(3))/2;
        xi(1) = (1+1./sqrt(3))/2;

        nodes.SetSize(8,2);
        T.SetSize(8, 8);

        // set nodes
        nodes(0,0) = xi(0); nodes(0,1) = 0;
        nodes(1,0) = xi(1); nodes(1,1) = 0;

        nodes(2,0) = xi(1); nodes(2,1) = xi(0);
        nodes(3,0) = xi(0); nodes(3,1) = xi(1);

        nodes(4,0) = 0; nodes(4,1) = xi(1);
        nodes(5,0) = 0; nodes(5,1) = xi(0);

        nodes(6,0) = c; nodes(6,1) = c;
        nodes(7,0) = c; nodes(7,1) = c;

        // set matrix T
        T = 0.0;

        // dofs 0 and 1
        for (int i=0; i<=1; i++)
        {
            T(1,i) = -(2*(1 - nodes(i,0) - nodes(i,1))-1);
            T(3,i) = -(2*nodes(i,0)-1);
            T(5,i) = -(2*nodes(i,1)-1);
            T(6,i) = -(2*nodes(i,1)-1)*(nodes(i,1) - c);
            T(7,i) = -(2*nodes(i,0)-1)*(nodes(i,1) - c);
        }

        // dofs 2 and 3
        for (int i=2; i<=3; i++)
        {
            T(0,i) = T(1,i) = (2*(1 - nodes(i,0) - nodes(i,1))-1);
            T(2,i) = T(3,i) = (2*nodes(i,0)-1);
            T(4,i) = T(5,i) = (2*nodes(i,1)-1);
            T(6,i) = (2*nodes(i,1)-1)*(nodes(i,0) + nodes(i,1) - 2*c);
            T(7,i) = (2*nodes(i,0)-1)*(nodes(i,0) + nodes(i,1) - 2*c);
        }

        // dofs 4 and 5
        for (int i=4; i<=5; i++)
        {
            T(0,i) = -(2*(1 - nodes(i,0) - nodes(i,1))-1);
            T(2,i) = -(2*nodes(i,0)-1);
            T(4,i) = -(2*nodes(i,1)-1);
            T(6,i) = -(2*nodes(i,1)-1)*(nodes(i,0) - c);
            T(7,i) = -(2*nodes(i,0)-1)*(nodes(i,0) - c);
        }

        // dofs 6 and 7
        int i=6;
        {
            T(1,i) = -(2*(1 - nodes(i,0) - nodes(i,1))-1);
            T(3,i) = -(2*nodes(i,0)-1);
            T(5,i) = -(2*nodes(i,1)-1);
            T(6,i) = -(2*nodes(i,1)-1)*(nodes(i,1) - c);
            T(7,i) = -(2*nodes(i,0)-1)*(nodes(i,1) - c);

            i++;
            T(0,i) = -(2*(1 - nodes(i,0) - nodes(i,1))-1);
            T(2,i) = -(2*nodes(i,0)-1);
            T(4,i) = -(2*nodes(i,1)-1);
            T(6,i) = -(2*nodes(i,1)-1)*(nodes(i,0) - c);
            T(7,i) = -(2*nodes(i,0)-1)*(nodes(i,0) - c);
        }

        /*for (int k=0; k<T.NumRows(); k++) {
            for (int l=0; l<T.NumCols(); l++)
                std::cout << T(k,l) << " ";
            std::cout << "\n";
        }*/

        Ti.Factor(T);

        /*DenseMatrix invT;
        Ti.GetInverseMatrix(invT);
        for (int k=0; k<T.NumRows(); k++) {
            for (int l=0; l<T.NumCols(); l++)
                std::cout << invT(k,l) << " ";
            std::cout << "\n";
        }*/
    }

    void eval_vshape(const double x, const double y,
                     DenseMatrix &vshape)
    {
        double c = 1./3.;

        int ndofs = 8;
        DenseMatrix vshape_buf(ndofs, 2);
        vshape_buf = 0.0;

        vshape_buf(0,0) = (2*(1-x-y)-1);
        vshape_buf(1,1) = (2*(1-x-y)-1);

        vshape_buf(2,0) = (2*x-1);
        vshape_buf(3,1) = (2*x-1);

        vshape_buf(4,0) = (2*y-1);
        vshape_buf(5,1) = (2*y-1);

        vshape_buf(6,0) = (2*y-1)*(x - c);
        vshape_buf(6,1) = (2*y-1)*(y - c);
        vshape_buf(7,0) = (2*x-1)*(x - c);
        vshape_buf(7,1) = (2*x-1)*(y - c);

        Ti.Mult(vshape_buf, vshape);
    }

private:
    DenseMatrix nodes;
    DenseMatrix T;
    DenseMatrixInverse Ti;
};


TEST(RTspace, vshape_deg0_test1)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"tri_elem1";
    Mesh mesh(mesh_file.c_str());
    assert(mesh.GetNE() == 1);

    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(0, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    int j=0; // only 1 element in the mesh
    {
        const FiniteElement *fe = R_space->GetFE(j);
        ElementTransformation *trans
                = R_space->GetElementTransformation(j);

        const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), 2);

        DenseMatrix vshape(3,2);
        DenseMatrix true_vshape(3,2);

        auto vshape_fn = [](Vector &x)
        {
            DenseMatrix vshape(3,2);
            vshape(0,0) = x(0)-2;
            vshape(0,1) = x(1);

            vshape(1,0) = x(0)-2;
            vshape(1,1) = x(1)-1;

            vshape(2,0) = x(0);
            vshape(2,1) = x(1);

            vshape *= 0.5;

            return vshape;
        };

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans->SetIntPoint(&ip);
            fe->CalcVShape(*trans, true_vshape);

            Vector eip;
            trans->Transform(ip, eip);
            vshape = vshape_fn(eip);

            /*std::cout << "\n\nvshape:" << std::endl;
            for (int k=0; k<vshape.NumRows(); k++) {
                for (int l=0; l<vshape.NumCols(); l++)
                    std::cout << vshape(k,l) << " ";
                std::cout << "\n";
            }

            std::cout << "\n\nTrue vshape:" << std::endl;
            for (int k=0; k<vshape.NumRows(); k++) {
                for (int l=0; l<vshape.NumCols(); l++)
                    std::cout << true_vshape(k,l) << " ";
                std::cout << "\n";
            }*/

            double TOL = 1E-10;
            true_vshape -= vshape;
            ASSERT_LE(true_vshape.FNorm(), TOL);
        }
    }

    delete hdiv_coll;
    delete R_space;
}


TEST(RTspace, gradvshape_deg0)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"tri_elem1";
    Mesh mesh(mesh_file.c_str());
    assert(mesh.GetNE() == 1);

    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(0, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    int ndofs = 3;
    auto gradvshape_fn = [](Vector &)
    {
        DenseTensor gradvshape(2,2,3);
        gradvshape = 0.0;
        gradvshape(0,0,0) = 0.5; gradvshape(1,1,0) = 0.5;
        gradvshape(0,0,1) = 0.5; gradvshape(1,1,1) = 0.5;
        gradvshape(0,0,2) = 0.5; gradvshape(1,1,2) = 0.5;
        return gradvshape;
    };

    int j=0;
    {
        const FiniteElement *fe = R_space->GetFE(j);
        ElementTransformation *trans
                = R_space->GetElementTransformation(j);

        const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), 2);

        DenseTensor true_gradvshape(2,2,3);
        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans->SetIntPoint(&ip);
            fe->CalcGradVShape(*trans, true_gradvshape);

            Vector eip;
            trans->Transform(ip, eip);
            auto gradvshape = gradvshape_fn(eip);
            for (int m=0; m < ndofs; m++)
            {
                /*std::cout << "\n\ngradvshape:" << std::endl;
                for (int k=0; k<gradvshape(m).NumRows(); k++) {
                    for (int l=0; l<gradvshape(m).NumCols(); l++)
                        std::cout << gradvshape(k,l,m) << " ";
                    std::cout << "\n";
                }

                std::cout << "\n\ntrue_gradvshape:" << std::endl;
                for (int k=0; k<gradvshape(m).NumRows(); k++) {
                    for (int l=0; l<gradvshape(m).NumCols(); l++)
                        std::cout << true_gradvshape(k,l,m) << " ";
                    std::cout << "\n";
                }*/

                true_gradvshape(m) -= gradvshape(m);
                double TOL = 1E-10;
                ASSERT_LE(true_gradvshape(m).FNorm(), TOL);
            }
        }
    }

    delete hdiv_coll;
    delete R_space;
}


TEST(RTspace, vshape_deg1)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"tri_elem2";
    Mesh mesh(mesh_file.c_str());

    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(1, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    for (int j=0; j<1; j++)
    {
        const FiniteElement *fe = R_space->GetFE(j);
        ElementTransformation *trans
                = R_space->GetElementTransformation(j);

        const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), 2);

        DenseMatrix vshape(8,2);
        DenseMatrix true_vshape(8,2);

        RT1_refEl rt1_refEl;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans->SetIntPoint(&ip);
            //fe->CalcVShape(trans->GetIntPoint(), true_vshape);
            fe->CalcVShape(*trans, true_vshape);

            DenseMatrix vshape_buf;
            rt1_refEl.eval_vshape(ip.x, ip.y, vshape_buf);
            MultABt(vshape_buf, trans->Jacobian(), vshape);
            vshape *= (1./trans->Weight());

            /*std::cout << "\n\nvshape:" << std::endl;
            for (int k=0; k<vshape.NumRows(); k++) {
                for (int l=0; l<vshape.NumCols(); l++)
                    std::cout << vshape(k,l) << " ";
                std::cout << "\n";
            }

            std::cout << "\n\nTrue vshape:" << std::endl;
            for (int k=0; k<vshape.NumRows(); k++) {
                for (int l=0; l<vshape.NumCols(); l++)
                    std::cout << true_vshape(k,l) << " ";
                std::cout << "\n";
            }*/

            double TOL = 1E-10;
            true_vshape -= vshape;
            ASSERT_LE(true_vshape.FNorm(), TOL);
            //std::cout << "\nError: " << true_vshape.FNorm() << std::endl;
        }
    }

    delete hdiv_coll;
    delete R_space;
}


/*TEST(RTspace, gradvshape_deg0)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"tri_elem2";
    Mesh mesh(mesh_file.c_str());

    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(0, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    int ndofs = 3;
    int dim = 2;

    auto gradvshape_fn = [](Vector &x)
    {
        DenseTensor gradvshape(3,2,2);
        gradvshape = 0.0;
        gradvshape(0,0,0) = 0.5; gradvshape(0,1,1) = 0.5;
        gradvshape(1,0,0) = 0.5; gradvshape(1,1,1) = 0.5;
        gradvshape(2,0,0) = 0.5; gradvshape(2,1,1) = 0.5;
        return gradvshape;
    };

    for (int j=0; j<1; j++)
    {
        const FiniteElement *fe = R_space->GetFE(j);
        ElementTransformation *trans
                = R_space->GetElementTransformation(j);

        const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), 2);

        //DenseTensor gradvshape(2,2,3);
        DenseTensor true_gradvshape(2,2,3);

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans->SetIntPoint(&ip);
            fe->CalcGradVShape(*trans, true_gradvshape);

            DenseMatrix J, Jinv;
            DenseTensor gradvshape_buf;
            J = trans->Jacobian();
            Jinv = trans->InverseJacobian();
            eval_gradvshape_RT0_refEl(ip.x, ip.y, gradvshape_buf);

            for (int m=0; m < ndofs; m++)
            {
                // apply Piola transformation
                // gradvshape = |J|^{-1}*J * gradvshape_buf * invJ
                DenseMatrix tempMat(dim, dim);
                Mult(J, gradvshape_buf(m), tempMat);
                Mult(tempMat, Jinv, gradvshape(m));
                gradvshape(m) *= (1.0 / trans->Weight());

                true_gradvshape(m) -= gradvshape(m);
                double TOL = 1E-10;
                ASSERT_LE(true_gradvshape(m).FNorm(), TOL);
            }
        }
    }

    delete hdiv_coll;
    delete R_space;
}*/
