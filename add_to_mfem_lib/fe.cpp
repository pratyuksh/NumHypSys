
void FiniteElement::CalcGradVShape (
   const IntegrationPoint &ip, DenseTensor &gradvshape) const
{
   mfem_error ("FiniteElement::CalcGradVShape (ip, ...)\n"
               "   is not implemented for this class!");
}

void FiniteElement::CalcGradVShape (
   ElementTransformation &Trans, DenseTensor &gradvshape) const
{
   mfem_error ("FiniteElement::CalcGradVShape (trans, ...)\n"
               "   is not implemented for this class!");
}



void VectorFiniteElement::CalcVShape_RT (
   ElementTransformation &Trans, DenseMatrix &shape) const
{
   MFEM_ASSERT(MapType == H_DIV, "");
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(Dof, Dim);
#endif
   CalcVShape(Trans.GetIntPoint(), vshape);
   MultABt(vshape, Trans.Jacobian(), shape);
   shape *= (1.0 / Trans.Weight());
}

void VectorFiniteElement::CalcGradVShape_RT (
   ElementTransformation &Trans, DenseTensor &gradvshape) const
{
   MFEM_ASSERT(MapType == H_DIV, "");
#ifdef MFEM_THREAD_SAFE
   DenseTensor gradvshape_ref(Dof, Dim, Dim);
#endif
   const DenseMatrix &J = Trans.Jacobian();
   const DenseMatrix &Jinv = Trans.InverseJacobian();
   CalcGradVShape(Trans.GetIntPoint(), gradvshape_ref);

    // apply Piola transformation
    // gradvshape = (1/det(J)) * J * gradvshape_ref * invJ
    for (int j=0; j < Dof; j++)
    {
        DenseMatrix tempMat(Dim, Dim);
        tempMat = 0.0;
        for (int k=0; k<Dim; k++) // tempMat = J * gradvshape_ref
            for (int l=0; l<Dim; l++)
                for (int s=0; s<Dim; s++)
                    tempMat(k,l) += J(k,s)*gradvshape_ref(j,s,l);

        gradvshape(j) = 0.0;
        Mult(tempMat, Jinv, gradvshape(j));
        gradvshape(j) *= (1.0 / Trans.Weight());
    }
}



void RT_QuadrilateralElement ::
CalcGradVShape(const IntegrationPoint &ip,
               DenseTensor &gradvshape) const
{
    const int pp1 = Order;

#ifdef MFEM_THREAD_SAFE
    Vector shape_cx(pp1 + 1), shape_cy(pp1 + 1);
    Vector shape_ox(pp1), shape_oy(pp1);
    Vector dshape_cx(pp1 + 1), dshape_cy(pp1 + 1);
    Vector dshape_ox(pp1), dshape_oy(pp1);
#endif

    cbasis1d.Eval(ip.x, shape_cx, dshape_cx);
    obasis1d.Eval(ip.x, shape_ox, dshape_ox);
    cbasis1d.Eval(ip.y, shape_cy, dshape_cy);
    obasis1d.Eval(ip.y, shape_oy, dshape_oy);
   
    int o = 0;
    for (int j = 0; j < pp1; j++)
        for (int i = 0; i <= pp1; i++)
        {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
                idx = -1 - idx, s = -1;
            }
            else
            {
                s = +1;
            }
            gradvshape(idx, 0, 0) = s*dshape_cx(i)*shape_oy(j);
            gradvshape(idx, 0, 1) = s*shape_cx(i)*dshape_oy(j);
            gradvshape(idx, 1, 0) = 0;
            gradvshape(idx, 1, 1) = 0;
        }
    for (int j = 0; j <= pp1; j++)
        for (int i = 0; i < pp1; i++)
        {
            int idx, s;
            if ((idx = dof_map[o++]) < 0)
            {
                idx = -1 - idx, s = -1;
            }
            else
            {
                s = +1;
            }
            gradvshape(idx, 0, 0) = 0;
            gradvshape(idx, 0, 1) = 0;
            gradvshape(idx, 1, 0) = s*dshape_ox(i)*shape_cy(j);
            gradvshape(idx, 1, 1) = s*shape_ox(i)*dshape_cy(j);
        }
}



void RT_TriangleElement::CalcGradVShape(const IntegrationPoint &ip,
                                        DenseTensor &gradvshape) const
{
    const int p = Order - 1;

#ifdef MFEM_THREAD_SAFE
    Vector shape_x(p + 1), shape_y(p + 1), shape_l(p + 1);
    Vector dshape_x(p + 1), dshape_y(p + 1), dshape_l(p + 1);
    DenseTensor gradu(Dof, Dim, Dim);
#endif

    poly1d.CalcBasis(p, ip.x, shape_x, dshape_x);
    poly1d.CalcBasis(p, ip.y, shape_y, dshape_y);
    poly1d.CalcBasis(p, 1. - ip.x - ip.y, shape_l, dshape_l);

    int o = 0;
    for (int j = 0; j <= p; j++)
    {
        for (int i = 0; i + j <= p; i++)
        {
            int k = p - i - j;
            gradu(o,0,0) = (dshape_x(i)*shape_l(k)-shape_x(i)*dshape_l(k))*shape_y(j);
            gradu(o,0,1) = (dshape_y(j)*shape_l(k)-shape_y(j)*dshape_l(k))*shape_x(i);
            gradu(o,1,0) = 0.;
            gradu(o,1,1) = 0.;
            o++;

            gradu(o,0,0) = 0.;
            gradu(o,0,1) = 0.;
            gradu(o,1,0) = (dshape_x(i)*shape_l(k)-shape_x(i)*dshape_l(k))*shape_y(j);
            gradu(o,1,1) = (dshape_y(j)*shape_l(k)-shape_y(j)*dshape_l(k))*shape_x(i);
            o++;
        }
    }

    for (int i = 0; i <= p; i++)
    {
        int j = p - i;
        double s = shape_x(i)*shape_y(j);
        double dsdx = dshape_x(i)*shape_y(j);
        double dsdy = shape_x(i)*dshape_y(j);
        gradu(o,0,0) = s + (ip.x - c)*dsdx;
        gradu(o,0,1) = (ip.x - c)*dsdy;
        gradu(o,1,0) = (ip.y - c)*dsdx;
        gradu(o,1,1) = s + (ip.y - c)*dsdy;
        o++;
    }

    for (int k=0; k<Dim; k++)
    {
        Ti.Mult(gradu(k), gradvshape(k));
    }
}

