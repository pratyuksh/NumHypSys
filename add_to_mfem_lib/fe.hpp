/// Abstract class for Finite Elements
class FiniteElement
{
protected:
//...
//...
#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix vshape; // Dof x Dim
   mutable DenseTensor gradvshape_ref; // Dof x Dim x Dim
#endif

public:
                          
   /** @brief Evaluate the values of gradients all shape functions of a *vector* 
       finite element in reference space at the given point @a ip. */
   /** Each matrix of the result DenseTensor @a gradvshape contains the components of
       one vector shape function. The size (#Dof x #Dim x #Dim) of @a gradvshape 
       must be set in advance. */
   virtual void CalcGradVShape(const IntegrationPoint &ip,
                               DenseTensor &gradvshape) const;

   /** @brief Evaluate the values of gradients all shape functions of a *vector* 
       finite element in physical space at the point described by @a Trans. */
   /** Each matrix of the result DenseTensor @a gradvshape contains the components of
       one vector shape function. The size (SDim x SDim x #Dof) of @a gradvshape must 
       be set in advance, where SDim >= #Dim is the physical space dimension as
       described by @a Trans. */
   virtual void CalcGradVShape(ElementTransformation &Trans,
                               DenseTensor &gradvshape) const;

};


class VectorFiniteElement : public FiniteElement
{
//...
protected:
#ifndef MFEM_THREAD_SAFE
   mutable DenseMatrix J, Jinv;
#endif
   void CalcGradVShape_RT(ElementTransformation &Trans, 
                          DenseTensor &gradvshape) const;
//...
};


class RT_QuadrilateralElement : public VectorFiniteElement
{
//...
private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
   mutable Vector dshape_cx, dshape_cy, dshape_ox, dshape_oy;
#endif

public:
   virtual void CalcGradVShape(const IntegrationPoint &ip,
                               DenseTensor &gradvshape) const;
   virtual void CalcGradVShape(ElementTransformation &Trans, 
                               DenseTensor &gradvshape) const
   { CalcGradVShape_RT(Trans, gradvshape); }
//...
};

class RT_TriangleElement : public VectorFiniteElement
{
//...
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l;
   mutable Vector dshape_x, dshape_y, dshape_l;
   mutable DenseMatrix u;
   mutable Vector divu;
#endif

public:
   virtual void CalcGradVShape(const IntegrationPoint &ip,
                               DenseTensor &gradvshape) const;
   virtual void CalcGradVShape(ElementTransformation &Trans, 
                               DenseTensor &gradvshape) const
   { CalcGradVShape_RT(Trans, gradvshape); }
//...
};


#endif
