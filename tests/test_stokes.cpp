#include <gtest/gtest.h>

#include "mfem.hpp"
using namespace mfem;

#include <iostream>
#include <fstream>
#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../include/mymfem/mybilinearform.hpp"
#include "../include/mymfem/utilities.hpp"

#include "../include/core/config.hpp"
#include "../include/stokes/assembly.hpp"


// test velocity coefficient
class TestVelocityCoeff : public VectorCoefficient
{
public:
    TestVelocityCoeff () : VectorCoefficient (2) {}

    void Eval (Vector& v, ElementTransformation& T,
               const IntegrationPoint& ip)
    {
        Vector transip(2);
        T.Transform(ip, transip);
        v = velocity(transip);
    }

    Vector velocity (const Vector& x) const
    {
        Vector v(2);
        v(0) = x(0);
        v(1) = -x(1);
        return v;
    }
};

TEST(RStokes, vshape_test1)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"ref_tri_elem";
    Mesh mesh(mesh_file.c_str());
    assert(mesh.GetNE() == 1);

    int deg = 1;
    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(deg, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    Vector true_v(2);
    true_v(0) = +3;
    true_v(1) = -2;
    VectorConstantCoefficient vCoeff(true_v);
    GridFunction vFn(R_space);
    vFn.ProjectCoefficient(vCoeff);

    int j=0; // only 1 element in the mesh
    {
        const FiniteElement *fe = R_space->GetFE(j);
        ElementTransformation *trans
                = R_space->GetElementTransformation(j);

        Array<int> vdofs;
        Vector fe_dofs(fe->GetDof());
        R_space->GetElementVDofs(j, vdofs);
        get_dofs(vFn, vdofs, fe_dofs);

        const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), 2);
        DenseMatrix vshape(fe->GetDof(),2);
        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans->SetIntPoint(&ip);
            fe->CalcVShape(*trans, vshape);

            Vector v(2);
            vshape.MultTranspose(fe_dofs, v);

            double TOL = 1E-10;
            v -= true_v;
            ASSERT_LE(v.Norml2(), TOL);
        }
    }

    delete hdiv_coll;
    delete R_space;
}

TEST(RStokes, gradvshape_test1)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"ref_tri_elem";
    Mesh mesh(mesh_file.c_str());
    assert(mesh.GetNE() == 1);

    int deg = 1;
    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(deg, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    Vector true_v(2);
    true_v(0) = +3;
    true_v(1) = -2;
    VectorConstantCoefficient vCoeff(true_v);
    GridFunction vFn(R_space);
    vFn.ProjectCoefficient(vCoeff);

    int j=0; // only 1 element in the mesh
    {
        const FiniteElement *fe = R_space->GetFE(j);
        ElementTransformation *trans
                = R_space->GetElementTransformation(j);

        Array<int> vdofs;
        Vector fe_dofs(fe->GetDof());
        R_space->GetElementVDofs(j, vdofs);
        get_dofs(vFn, vdofs, fe_dofs);

        const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), 0);
        DenseTensor gradvshape(2,2,fe->GetDof());
        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans->SetIntPoint(&ip);
            fe->CalcGradVShape(*trans, gradvshape);

            DenseMatrix gradu(2,2);
            gradu = 0.0;
            for (int k=0; k<fe->GetDof(); k++) {
                gradu.Add(fe_dofs(k), gradvshape(k));
            }

            double TOL = 1E-10;
            ASSERT_LE(gradu.FNorm(), TOL);
        }
    }

    delete hdiv_coll;
    delete R_space;
}

TEST(RStokes, gradvshape_test2)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"ref_tri_elem";
    Mesh mesh(mesh_file.c_str());
    assert(mesh.GetNE() == 1);

    int deg = 3;
    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(deg, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    TestVelocityCoeff vCoeff;
    GridFunction vFn(R_space);
    vFn.ProjectCoefficient(vCoeff);

    DenseMatrix true_gradu(2,2);
    true_gradu = 0.0;
    true_gradu(0,0) = 1;
    true_gradu(1,1) = -1;

    int j=0; // only 1 element in the mesh
    {
        const FiniteElement *fe = R_space->GetFE(j);
        ElementTransformation *trans
                = R_space->GetElementTransformation(j);

        Array<int> vdofs;
        Vector fe_dofs(fe->GetDof());
        R_space->GetElementVDofs(j, vdofs);
        get_dofs(vFn, vdofs, fe_dofs);

        const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), 2);
        DenseTensor gradvshape(2,2,fe->GetDof());
        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            trans->SetIntPoint(&ip);
            fe->CalcGradVShape(*trans, gradvshape);

            DenseMatrix gradu(2,2);
            gradu = 0.0;
            for (int k=0; k<fe->GetDof(); k++) {
                gradu.Add(fe_dofs(k), gradvshape(k));
            }

            double TOL = 1E-10;
            gradu -= true_gradu;
            ASSERT_LE(gradu.FNorm(), TOL);
        }
    }

    delete hdiv_coll;
    delete R_space;
}

TEST(Stokes, VectorFEDiffusion_test1)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"tri_elem1";
    Mesh mesh(mesh_file.c_str());
    assert(mesh.GetNE() == 1);

    Array<int> ess_bdr_marker;
    ess_bdr_marker.SetSize(mesh.bdr_attributes.Max());
    ess_bdr_marker = 1;

    GridFunction *gdum = nullptr; // dummy

    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(0, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    mymfem::MyBilinearForm *diffusion_form
            = new mymfem::MyBilinearForm(R_space);
    diffusion_form->MyAddDomainIntegrator
            (new mymfem::DiffusionIntegrator);
    diffusion_form->MyAssemble(gdum);
    diffusion_form->Finalize();
    SparseMatrix *diffusion_buf = diffusion_form->LoseMat();
    delete diffusion_form;
    DenseMatrix &diffusion = *(diffusion_buf->ToDenseMatrix());

    DenseMatrix true_diffusion(diffusion.NumRows(), diffusion.NumCols());
    true_diffusion = 0.0;
    true_diffusion(0,0) =  0.5;
    true_diffusion(0,1) = -0.5;
    true_diffusion(0,2) = -0.5;

    true_diffusion(1,0) = -0.5;
    true_diffusion(1,1) =  0.5;
    true_diffusion(1,2) =  0.5;

    true_diffusion(2,0) = -0.5;
    true_diffusion(2,1) =  0.5;
    true_diffusion(2,2) =  0.5;

    //true_diffusion.Print();
    //diffusion.Print();

    double TOL = 1E-10;
    diffusion -= true_diffusion;
    ASSERT_LE(diffusion.FNorm(), TOL);

    delete hdiv_coll;
    delete R_space;
}

TEST(Stokes, VectorFEDiffusionConsistencySymmetry_test1)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"tri_elem1";
    Mesh mesh(mesh_file.c_str());

    Array<int> ess_bdr_marker;
    ess_bdr_marker.SetSize(mesh.bdr_attributes.Max());
    ess_bdr_marker = 1;

    GridFunction *gdum = nullptr; // dummy

    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(0, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    mymfem::MyBilinearForm *diffusionCons_form
            = new mymfem::MyBilinearForm(R_space);
    diffusionCons_form->MyAddFaceIntegrator
            (new mymfem::DiffusionConsistencyIntegrator);
    diffusionCons_form->MyAssemble(gdum);
    diffusionCons_form->Finalize();
    SparseMatrix *diffusionCons_buf = diffusionCons_form->LoseMat();
    delete diffusionCons_form;
    DenseMatrix &diffusionCons = *(diffusionCons_buf->ToDenseMatrix());

    mymfem::MyBilinearForm *diffusionSymm_form
            = new mymfem::MyBilinearForm(R_space);
    diffusionSymm_form->MyAddFaceIntegrator
            (new mymfem::DiffusionSymmetryIntegrator);
    diffusionSymm_form->MyAssemble(gdum);
    diffusionSymm_form->Finalize();
    SparseMatrix *diffusionSymm_buf = diffusionSymm_form->LoseMat();
    delete diffusionSymm_form;
    DenseMatrix &diffusionSymm = *(diffusionSymm_buf->ToDenseMatrix());

    DenseMatrix true_diffusion(diffusionCons.NumRows(),
                               diffusionCons.NumCols());
    true_diffusion = 0.0;
    true_diffusion(0,0) = -0.5;
    true_diffusion(0,1) = +0.5;
    true_diffusion(0,2) = +0.5;

    true_diffusion(1,0) = +0.5;
    true_diffusion(1,1) = -0.5;
    true_diffusion(1,2) = -0.5;

    true_diffusion(2,0) = +0.5;
    true_diffusion(2,1) = -0.5;
    true_diffusion(2,2) = -0.5;

    //true_diffusion.Print();
    //diffusion.Print();

    double TOL = 1E-10;
    diffusionCons -= true_diffusion;
    diffusionSymm -= true_diffusion;
    ASSERT_LE(diffusionCons.FNorm(), TOL);
    ASSERT_LE(diffusionSymm.FNorm(), TOL);

    delete hdiv_coll;
    delete R_space;
}

TEST(Stokes, VectorFEDiffusionPenalty_test1)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"tri_elem1";
    Mesh mesh(mesh_file.c_str());

    Array<int> ess_bdr_marker;
    ess_bdr_marker.SetSize(mesh.bdr_attributes.Max());
    ess_bdr_marker = 1;

    GridFunction *gdum = nullptr; // dummy

    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(0, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    mymfem::MyBilinearForm *diffusionPen_form
            = new mymfem::MyBilinearForm(R_space);
    diffusionPen_form->MyAddFaceIntegrator
            (new mymfem::DiffusionPenaltyIntegrator(1.0));
    diffusionPen_form->MyAssemble(gdum);
    diffusionPen_form->Finalize();
    SparseMatrix *diffusionCons_buf = diffusionPen_form->LoseMat();
    delete diffusionPen_form;
    DenseMatrix &diffusionPen= *(diffusionCons_buf->ToDenseMatrix());

    DenseMatrix true_diffusion(diffusionPen.NumRows(),
                               diffusionPen.NumCols());
    true_diffusion = 0.0;
    true_diffusion(0,0) = 5./6;
    true_diffusion(0,1) = -7./12;
    true_diffusion(0,2) = 1./6;

    true_diffusion(1,0) = -7./12;
    true_diffusion(1,1) = 13./12;
    true_diffusion(1,2) = -5./12;

    true_diffusion(2,0) = 1./6;
    true_diffusion(2,1) = -5./12;
    true_diffusion(2,2) = 11./6;

    //true_diffusion.Print();
    //diffusionPen.Print();

    double TOL = 1E-10;
    diffusionPen -= true_diffusion;
    ASSERT_LE(diffusionPen.FNorm(), TOL);

    delete hdiv_coll;
    delete R_space;
}


TEST(Stokes, VectorFEDiffusion_test2)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"test_mesh1";
    Mesh mesh(mesh_file.c_str());
    
    Array<int> ess_bdr_marker;
    ess_bdr_marker.SetSize(mesh.bdr_attributes.Max());
    ess_bdr_marker = 1;
    
    GridFunction *gdum = nullptr; // dummy
    
    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(0, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);
            
    mymfem::MyBilinearForm *diffusion_form
            = new mymfem::MyBilinearForm(R_space);
    diffusion_form->MyAddDomainIntegrator
            (new mymfem::DiffusionIntegrator);
    diffusion_form->MyAssemble(gdum);
    diffusion_form->Finalize();
    SparseMatrix *diffusion_buf = diffusion_form->LoseMat();
    delete diffusion_form;
    DenseMatrix &diffusion = *(diffusion_buf->ToDenseMatrix());

    DenseMatrix true_diffusion(diffusion.NumRows(), diffusion.NumCols());
    true_diffusion = 0.0;
    true_diffusion(0,0) =  2;
    true_diffusion(0,1) = -1;
    true_diffusion(0,2) = -1;
    true_diffusion(0,3) = -1;
    true_diffusion(0,4) = -1;

    true_diffusion(1,0) = -1;
    true_diffusion(1,1) =  1;
    true_diffusion(1,2) =  1;

    true_diffusion(2,0) = -1;
    true_diffusion(2,1) =  1;
    true_diffusion(2,2) =  1;

    true_diffusion(3,0) = -1;
    true_diffusion(3,3) =  1;
    true_diffusion(3,4) =  1;

    true_diffusion(4,0) = -1;
    true_diffusion(4,3) =  1;
    true_diffusion(4,4) =  1;

    //true_diffusion.Print();
    //diffusion.Print();

    double TOL = 1E-10;
    diffusion -= true_diffusion;
    ASSERT_LE(diffusion.FNorm(), TOL);

    delete hdiv_coll;
    delete R_space;
}

TEST(Stokes, VectorFEDiffusionConsistencySymmetry_test2)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"test_mesh1";
    Mesh mesh(mesh_file.c_str());

    Array<int> ess_bdr_marker;
    ess_bdr_marker.SetSize(mesh.bdr_attributes.Max());
    ess_bdr_marker = 1;

    GridFunction *gdum = nullptr; // dummy

    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(0, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    mymfem::MyBilinearForm *diffusionCons_form
            = new mymfem::MyBilinearForm(R_space);
    diffusionCons_form->MyAddFaceIntegrator
            (new mymfem::DiffusionConsistencyIntegrator);
    diffusionCons_form->MyAssemble(gdum);
    diffusionCons_form->Finalize();
    SparseMatrix *diffusionCons_buf = diffusionCons_form->LoseMat();
    delete diffusionCons_form;
    DenseMatrix &diffusionCons = *(diffusionCons_buf->ToDenseMatrix());

    mymfem::MyBilinearForm *diffusionSymm_form
            = new mymfem::MyBilinearForm(R_space);
    diffusionSymm_form->MyAddFaceIntegrator
            (new mymfem::DiffusionSymmetryIntegrator);
    diffusionSymm_form->MyAssemble(gdum);
    diffusionSymm_form->Finalize();
    SparseMatrix *diffusionSymm_buf = diffusionSymm_form->LoseMat();
    delete diffusionSymm_form;
    DenseMatrix &diffusionSymm = *(diffusionSymm_buf->ToDenseMatrix());

    DenseMatrix true_diffusion(diffusionCons.NumRows(),
                               diffusionCons.NumCols());
    true_diffusion = 0.0;

    true_diffusion(1,0) = +1;
    true_diffusion(1,1) = -1;
    true_diffusion(1,2) = -1;

    true_diffusion(2,0) = +1;
    true_diffusion(2,1) = -1;
    true_diffusion(2,2) = -1;

    true_diffusion(3,0) = +1;
    true_diffusion(3,3) = -1;
    true_diffusion(3,4) = -1;

    true_diffusion(4,0) = +1;
    true_diffusion(4,3) = -1;
    true_diffusion(4,4) = -1;

    //true_diffusion.Print();
    //diffusionCons.Print();
    //diffusionSymm.Print();

    double TOL = 1E-10;
    diffusionCons -= true_diffusion;
    ASSERT_LE(diffusionCons.FNorm(), TOL);

    true_diffusion.Transpose();
    diffusionSymm -= true_diffusion;
    ASSERT_LE(diffusionSymm.FNorm(), TOL);

    delete hdiv_coll;
    delete R_space;
}


TEST(Stokes, consistency_quad_elem)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"ref_quad_elem";
    Mesh mesh(mesh_file.c_str());

    GridFunction *gdum = nullptr; // dummy

    int deg = 2;
    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(deg, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    Vector true_v(2);
    true_v(0) = +3;
    true_v(1) = -2;
    //VectorConstantCoefficient vCoeff(true_v);
    TestVelocityCoeff vCoeff;
    GridFunction v(R_space);
    v.ProjectCoefficient(vCoeff);

    mymfem::MyBilinearForm *diffusion_form
            = new mymfem::MyBilinearForm(R_space);
    diffusion_form->MyAddDomainIntegrator
            (new mymfem::DiffusionIntegrator);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionConsistencyIntegrator);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionSymmetryIntegrator);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionPenaltyIntegrator(1.0));
    diffusion_form->MyAssemble(gdum);
    diffusion_form->Finalize();
    SparseMatrix *diffusion = diffusion_form->LoseMat();
    delete diffusion_form;

    // Diffusion * V
    Vector buf1(R_space->GetTrueVSize());
    diffusion->Mult(v, buf1);

    // Right-hand side
    Vector buf2(R_space->GetTrueVSize());
    LinearForm bdryDiffusion_form(R_space);
    bdryDiffusion_form.AddBdrFaceIntegrator
            (new mymfem::BdryDiffusionConsistencyIntegrator(vCoeff));
    bdryDiffusion_form.AddBdrFaceIntegrator
            (new mymfem::BdryDiffusionPenaltyIntegrator(vCoeff, 1.0));
    bdryDiffusion_form.Assemble();
    buf2 = bdryDiffusion_form.GetData();

    double TOL = 1E-10;
    buf1 -= buf2;
    ASSERT_LE(buf1.Norml2(), TOL);

    delete hdiv_coll;
    delete R_space;
}

TEST(Stokes, consistency_tri_elem)
{
    std::string input_dir = "../input/";
    const std::string mesh_file = input_dir+"ref_tri_elem";
    Mesh mesh(mesh_file.c_str());

    GridFunction *gdum = nullptr; // dummy

    int deg = 2;
    FiniteElementCollection *hdiv_coll
            = new RT_FECollection(deg, mesh.Dimension());
    FiniteElementSpace *R_space
            = new FiniteElementSpace(&mesh, hdiv_coll);

    Vector true_v(2);
    true_v(0) = +3;
    true_v(1) = -2;
    //VectorConstantCoefficient vCoeff(true_v);
    TestVelocityCoeff vCoeff;
    GridFunction vFn(R_space);
    vFn.ProjectCoefficient(vCoeff);

    mymfem::MyBilinearForm *diffusion_form
            = new mymfem::MyBilinearForm(R_space);
    diffusion_form->MyAddDomainIntegrator
            (new mymfem::DiffusionIntegrator);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionConsistencyIntegrator);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionSymmetryIntegrator);
    diffusion_form->MyAddFaceIntegrator
            (new mymfem::DiffusionPenaltyIntegrator(1.0));
    diffusion_form->MyAssemble(gdum);
    diffusion_form->Finalize();
    SparseMatrix *diffusion = diffusion_form->LoseMat();
    delete diffusion_form;

    // Diffusion * V
    Vector buf1(R_space->GetTrueVSize());
    diffusion->Mult(vFn, buf1);

    // Right-hand side
    Vector buf2(R_space->GetTrueVSize());
    LinearForm bdryDiffusion_form(R_space);
    bdryDiffusion_form.AddBdrFaceIntegrator
            (new mymfem::BdryDiffusionConsistencyIntegrator(vCoeff));
    bdryDiffusion_form.AddBdrFaceIntegrator
            (new mymfem::BdryDiffusionPenaltyIntegrator(vCoeff, 1.0));
    bdryDiffusion_form.Assemble();
    buf2 = bdryDiffusion_form.GetData();

    double TOL = 1E-10;
    buf1 -= buf2;
    ASSERT_LE(buf1.Norml2(), TOL);

    delete hdiv_coll;
    delete R_space;
}


// End of file
