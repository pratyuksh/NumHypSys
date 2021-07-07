#include "../../include/mymfem/putilities.hpp"
#include "../../include/mymfem/assembly.hpp"


//! Update and rebalances
//! parallel mesh, fespace and grid function
void UpdateAndRebalance
(ParMesh* pmesh, ParFiniteElementSpace* fespace,
 ParGridFunction &x)
{
    fespace->Update();
    x.Update();

    if (pmesh->Nonconforming())
    {
        // Load balance the mesh.
        pmesh->Rebalance();
        fespace->Update();
        x.Update();
    }
    
    // Free any transformation matrices to save memory.
    fespace->UpdatesFinished();
}


//! Update and rebalances
//! parallel mesh, fespace and two grid function2
void UpdateAndRebalance
(ParMesh* pmesh, ParFiniteElementSpace* fespace,
 ParGridFunction &x1, ParGridFunction &x2)
{
   fespace->Update();
   x1.Update();
   x2.Update();

   if (pmesh->Nonconforming())
   {
      pmesh->Rebalance();
      fespace->Update();
      x1.Update();
      x2.Update();
   }
   fespace->UpdatesFinished();
}


//! Update and rebalances
//! parallel mesh, fespace, grid function,
//! bilinear and linear forms
void UpdateAndRebalance
(ParMesh* pmesh, ParFiniteElementSpace* fespace,
 ParGridFunction &x, ParBilinearForm* a,
 ParLinearForm* b)
{
   fespace->Update();
   x.Update();

   if (pmesh->Nonconforming())
   {
      pmesh->Rebalance();
      fespace->Update();
      x.Update();
   }

   a->Update();
   b->Update();

   fespace->UpdatesFinished();
}


//! Read dofs for Raviart-Thomas Spaces from MFEM
void get_dofs(const ParGridFunction &u,
              const Array<int> &vdofs,
              Vector &el_dofs)
{
    int idof, s;
    //cout << "\n\n";
    for (int k=0; k<vdofs.Size(); k++) {
        if (vdofs[k] >= 0) {
            idof = vdofs[k];
            s = +1;
        }
        else {
            idof = -1-vdofs[k];
            s = -1;
        }
        el_dofs(k) = s*u.Elem(idof);
        //cout << k << "\t"
        //     << vdofs[k] << "\t"
        //     << idof << "\t" << s << endl;
    }
}


//! Computes the are of a domain
double compute_area (const ParFiniteElementSpace& pfes)
{
    double area = 0;
    double myarea = 0;
    Mesh *mesh = pfes.GetMesh();

    int geom_type = pfes.GetFE(0)->GetGeomType();
    int order = pfes.GetOrder(0);
    std::unique_ptr <const IntegrationRule> ir;
    ir = std::make_unique<IntegrationRule>
            (IntRules.Get(geom_type, order));

    int n = ir->GetNPoints();
    for (int i=0; i < mesh->GetNE(); i++)
    {
        ElementTransformation *trans
                = mesh->GetElementTransformation(i);

        for (int k=0; k < n; k++)
        {
            const IntegrationPoint &ip = ir->IntPoint(k);
            trans->SetIntPoint(&ip);

            double coeff = ip.weight*trans->Weight();
            myarea += coeff;
        }
    }
    MPI_Allreduce(&myarea, &area, 1, MPI_DOUBLE, MPI_SUM,
                  pfes.GetComm());

    return area;
}


//! Velocity components function
//! Initializes components
void ParVelocityFunction :: init ()
{
    // initialize
    m_xComp.SetSize(m_sfes->GetTrueVSize());
    m_yComp.SetSize(m_sfes->GetTrueVSize());

    Vector *temp1=nullptr, *temp2=nullptr; // dummy

    // Assemble mass matrix
    ParBilinearForm *mass_form
            = new ParBilinearForm(m_sfes);
    mass_form->AddDomainIntegrator(new MassIntegrator);
    mass_form->Assemble();
    mass_form->Finalize();
    m_mass = mass_form->ParallelAssemble();
    delete mass_form;

    // Assemble x-component L2-projection matrix
    // \int_{\Omega} (u_h * nx) w_h dx
    ParMixedBilinearForm *xProj_form
            = new ParMixedBilinearForm(m_vfes, m_sfes);
    xProj_form->AddDomainIntegrator
            (new mymfem::VelocityXProjectionIntegrator);
    xProj_form->Assemble();
    xProj_form->Finalize();
    if (m_ess_bdr_marker.Size()) {
        xProj_form->EliminateTrialDofs(m_ess_bdr_marker,
                                       *temp1, *temp2);
    }
    m_xProj = xProj_form->ParallelAssemble();
    delete xProj_form;

    // Assemble y-component L2-projection matrix
    // \int_{\Omega} (u_h * ny) w_h dx
    ParMixedBilinearForm *yProj_form
            = new ParMixedBilinearForm(m_vfes, m_sfes);
    yProj_form->AddDomainIntegrator
            (new mymfem::VelocityYProjectionIntegrator);
    yProj_form->Assemble();
    yProj_form->Finalize();
    if (m_ess_bdr_marker.Size()) {
        yProj_form->EliminateTrialDofs(m_ess_bdr_marker,
                                       *temp1, *temp2);
    }
    m_yProj = yProj_form->ParallelAssemble();
    delete yProj_form;

    // AMG preconditioner
    m_amg = new HypreBoomerAMG();
    m_amg->SetPrintLevel(0);
    m_amg->SetOperator(*m_mass);

    // CG solver
    double tol = 1E-12;
    int maxIter = 1000;
    int verbose = 0;

    m_cgsolver = new HyprePCG(*m_mass);
    m_cgsolver->SetTol(tol);
    m_cgsolver->SetMaxIter(maxIter);
    m_cgsolver->SetPrintLevel(verbose);
    m_cgsolver->SetPreconditioner(*m_amg);
}

//! Evaluates x-component of velocity
Vector ParVelocityFunction
:: eval_xComponent(const ParGridFunction *v)
{
    Vector B(m_sfes->GetTrueVSize());
    Vector V(m_vfes->GetTrueVSize());
    v->GetTrueDofs(V);

    m_xProj->Mult(V, B);
    m_xComp = 0.;
    m_cgsolver->Mult(B, m_xComp);

    return m_xComp;
}

//! Evaluates y-component of velocity
Vector ParVelocityFunction
:: eval_yComponent(const ParGridFunction *v)
{
    Vector B(m_sfes->GetTrueVSize());
    Vector V(m_vfes->GetTrueVSize());
    v->GetTrueDofs(V);

    m_yProj->Mult(V, B);
    m_yComp = 0.;
    m_cgsolver->Mult(B, m_yComp);

    return m_yComp;
}

//! Evaluates x-component and y-component of velocity
std::pair<Vector, Vector> ParVelocityFunction
:: eval(const ParGridFunction * v)
{
    return {eval_xComponent(v), eval_yComponent(v)};
}

//! Evaluates x-component and y-component of velocity
void ParVelocityFunction :: operator()
(const ParGridFunction& v,
 ParGridFunction& vx, ParGridFunction& vy) const
{
    HypreParVector Bx(vx.ParFESpace());
    HypreParVector By(vy.ParFESpace());
    HypreParVector Vx(vx.ParFESpace());
    HypreParVector Vy(vy.ParFESpace());

    m_xProj->Mult(*(v.GetTrueDofs()), Bx);
    m_cgsolver->Mult(Bx, Vx);
    vx = Vx;

    m_yProj->Mult(*(v.GetTrueDofs()), By);
    m_cgsolver->Mult(By, Vy);
    vy = Vy;
}


//! Vorticity function
//! Initializes components
void ParVorticityFunction :: init ()
{
    Vector *temp1=nullptr, *temp2=nullptr; // dummy

    // Assemble vorticity mass matrix
    ParBilinearForm *mass_form
            = new ParBilinearForm(m_sfes);
    mass_form->AddDomainIntegrator(new MassIntegrator);
    mass_form->Assemble();
    mass_form->Finalize();
    m_mass = mass_form->ParallelAssemble();
    delete mass_form;

    // Assemble Velocity to Vorticity L2-projection matrix
    // velocity to vorticity projection form:
    // \int_{\Omega} curl(u_h) w_h dx
    ParMixedBilinearForm *proj_form
            = new ParMixedBilinearForm(m_vfes, m_sfes);
    proj_form->AddDomainIntegrator
            (new mymfem::VorticityProjectionIntegrator);
    proj_form->Assemble();
    proj_form->Finalize();
    if (m_ess_bdr_marker.Size()) {
        proj_form->EliminateTrialDofs(m_ess_bdr_marker,
                                      *temp1, *temp2);
    }
    m_proj = proj_form->ParallelAssemble();
    delete proj_form;

    // AMG preconditioner
    m_amg = new HypreBoomerAMG();
    m_amg->SetPrintLevel(0);
    m_amg->SetOperator(*m_mass);

    // CG solver
    double tol = 1E-12;
    int maxIter = 1000;
    int verbose = 0;

    m_cgsolver = new HyprePCG(*m_mass);
    m_cgsolver->SetTol(tol);
    m_cgsolver->SetMaxIter(maxIter);
    m_cgsolver->SetPrintLevel(verbose);
    m_cgsolver->SetPreconditioner(*m_amg);
}

//! Evaluates vorticity
void ParVorticityFunction :: operator()
(const ParGridFunction& v, ParGridFunction& w) const
{
    HypreParVector B(w.ParFESpace());
    HypreParVector W(w.ParFESpace());
    m_proj->Mult(*(v.GetTrueDofs()), B);
    m_cgsolver->Mult(B, W);
    w = W;
}


//! Sets the FE space to be considered
//! and the corresponding mass matrix and linear solver
void ParMeanFreePressure
:: set (ParFiniteElementSpace *pfes) const
{
    HypreParMatrix *mass = nullptr;
    m_massInvLfone = new HypreParVector(pfes);

    ParBilinearForm mass_form(pfes);
    mass_form.AddDomainIntegrator(new MassIntegrator);
    mass_form.Assemble();
    mass_form.Finalize();
    mass = mass_form.ParallelAssemble();

    ParLinearForm lfone_form(pfes);
    ConstantCoefficient one(1.0);
    lfone_form.AddDomainIntegrator
            (new DomainLFIntegrator(one));
    lfone_form.Assemble();
    m_lfone = lfone_form.ParallelAssemble();

    HyprePCG hpcg (*mass);
    hpcg.Mult(*m_lfone, *m_massInvLfone);
    delete mass;

    m_area = compute_area(*pfes);
}

//! Applies the mean-free constraint
//! to the input grid function
void ParMeanFreePressure
:: operator()(ParGridFunction &pressure) const
{
    double mean = get(pressure);
    pressure.AddDistribute(-mean, m_massInvLfone);
}

//! Returns mean of the given grid function
double ParMeanFreePressure
:: get(ParGridFunction& pressure) const
{
    double mean = 0;
    double mymean = ((*m_lfone)*pressure);
    //std::cout << m_myrank << "\t"
    //          << m_lfone->Size() << "\t"
    //          << pressure.Size() << "\t"
    //          << mymean << std::endl;
    MPI_Allreduce(&mymean, &mean, 1, MPI_DOUBLE, MPI_SUM,
                  pressure.ParFESpace()->GetComm());
    return mean/m_area;
}


//! Maps a set of physical points
//! to elements of the mesh
std::pair <int, IntegrationPoint> ParPointLocator
:: operator() (const Vector& x) const
{
    DenseMatrix pointMat(x.Size(),1);
    pointMat.SetCol(0, x);

    Array<int> elIds;
    Array<IntegrationPoint> ips;
    m_pmesh->FindPoints(pointMat, elIds, ips);

    return {elIds[0], ips[0]};
}


//! Creates a monolithic HypreParMatrix from a block
HypreParMatrix * mymfem::HypreParMatrixFromBlocks
(Array2D<HypreParMatrix *> &blocks,
Array2D<double> *blockCoeff)
{
   const int numBlockRows = blocks.NumRows();
   const int numBlockCols = blocks.NumCols();

   MFEM_VERIFY(numBlockRows > 0 &&
               numBlockCols > 0,
               "Invalid input to HypreParMatrixFromBlocks")

   if (blockCoeff != nullptr)
   {
      MFEM_VERIFY(numBlockRows == blockCoeff->NumRows() &&
                  numBlockCols == blockCoeff->NumCols(),
                  "Invalid input to "
                  "HypreParMatrixFromBlocks")
   }

   Array<int> rowOffsets(numBlockRows+1);
   Array<int> colOffsets(numBlockCols+1);

   int nonNullBlockRow0 = -1;
   for (int j=0; j<numBlockCols; ++j)
   {
      if (blocks(0,j) != nullptr)
      {
         nonNullBlockRow0 = j;
         break;
      }
   }

   MFEM_VERIFY(nonNullBlockRow0 >= 0,
               "Null row of blocks")
   MPI_Comm comm = blocks(0,nonNullBlockRow0)->GetComm();

   // Set offsets based on the number of rows
   // or columns in each block.
   rowOffsets = 0;
   colOffsets = 0;
   for (int i=0; i<numBlockRows; ++i)
   {
      for (int j=0; j<numBlockCols; ++j)
      {
         if (blocks(i,j) != nullptr)
         {
            const int nrows = blocks(i,j)->NumRows();
            const int ncols = blocks(i,j)->NumCols();

            MFEM_VERIFY(nrows > 0 &&
                        ncols > 0,
                        "Invalid block in "
                        "HypreParMatrixFromBlocks")

            if (rowOffsets[i+1] == 0)
            {
               rowOffsets[i+1] = nrows;
            }
            else
            {
               MFEM_VERIFY(rowOffsets[i+1] == nrows,
                           "Inconsistent blocks in "
                           "HypreParMatrixFromBlocks")
            }

            if (colOffsets[j+1] == 0)
            {
               colOffsets[j+1] = ncols;
            }
            else
            {
               MFEM_VERIFY(colOffsets[j+1] == ncols,
                           "Inconsistent blocks in "
                           "HypreParMatrixFromBlocks")
            }
         }
      }

      MFEM_VERIFY(rowOffsets[i+1] > 0,
              "Invalid input blocks")
      rowOffsets[i+1] += rowOffsets[i];
   }

   for (int j=0; j<numBlockCols; ++j)
   {
      MFEM_VERIFY(colOffsets[j+1] > 0,
              "Invalid input blocks")
      colOffsets[j+1] += colOffsets[j];
   }

   const int num_loc_rows = rowOffsets[numBlockRows];
   const int num_loc_cols = colOffsets[numBlockCols];

   int nprocs, rank;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nprocs);

   std::vector<int> all_num_loc_rows(nprocs);
   std::vector<int> all_num_loc_cols(nprocs);
   std::vector<HYPRE_Int> procRowOffsets(nprocs);
   std::vector<HYPRE_Int> procColOffsets(nprocs);
   std::vector<std::vector<HYPRE_Int>>
           blockRowProcOffsets(numBlockRows);
   std::vector<std::vector<HYPRE_Int>>
           blockColProcOffsets(numBlockCols);
   std::vector<std::vector<int>> procBlockRowOffsets(nprocs);
   std::vector<std::vector<int>> procBlockColOffsets(nprocs);

   HYPRE_Int first_loc_row, glob_nrows,
           first_loc_col, glob_ncols;
   GatherBlockOffsetData(comm, rank, nprocs,
                         num_loc_rows, rowOffsets,
                         all_num_loc_rows, numBlockRows,
                         blockRowProcOffsets,
                         procRowOffsets, procBlockRowOffsets,
                         first_loc_row,
                         glob_nrows);

   GatherBlockOffsetData(comm, rank, nprocs,
                         num_loc_cols, colOffsets,
                         all_num_loc_cols, numBlockCols,
                         blockColProcOffsets,
                         procColOffsets, procBlockColOffsets,
                         first_loc_col,
                         glob_ncols);

   std::vector<int> opI(num_loc_rows + 1);
   std::vector<int> cnt(num_loc_rows);

   for (int i = 0; i < num_loc_rows; ++i)
   {
      opI[i] = 0;
      cnt[i] = 0;
   }

   opI[num_loc_rows] = 0;

   Array2D<hypre_CSRMatrix *> csr_blocks(numBlockRows,
                                         numBlockCols);

   // Loop over all blocks, to determine nnz for each row.
   for (int i = 0; i < numBlockRows; ++i)
   {
      for (int j = 0; j < numBlockCols; ++j)
      {
         if (blocks(i, j) == nullptr)
         {
            csr_blocks(i, j) = nullptr;
         }
         else
         {
            csr_blocks(i, j)
                    = hypre_MergeDiagAndOffd(*blocks(i, j));

            for (int k = 0; k < csr_blocks(i, j)->num_rows;
                 ++k)
            {
               opI[rowOffsets[i] + k + 1] +=
                  csr_blocks(i, j)->i[k + 1]
                       - csr_blocks(i, j)->i[k];
            }
         }
      }
   }

   // Now opI[i] is nnz for row i-1.
   // Do a partial sum to get offsets.
   for (int i = 0; i < num_loc_rows; ++i)
   {
      opI[i + 1] += opI[i];
   }

   const int nnz = opI[num_loc_rows];

   std::vector<HYPRE_Int> opJ(nnz);
   std::vector<double> data(nnz);

   // Loop over all blocks, to set matrix data.
   for (int i = 0; i < numBlockRows; ++i)
   {
      for (int j = 0; j < numBlockCols; ++j)
      {
         if (csr_blocks(i, j) != nullptr)
         {
            const int nrows = csr_blocks(i, j)->num_rows;
            const double cij = blockCoeff ?
                        (*blockCoeff)(i, j) : 1.0;
#if MFEM_HYPRE_VERSION >= 21600
            const bool usingBigJ
                    = (csr_blocks(i, j)->big_j != NULL);
#endif

            for (int k = 0; k < nrows; ++k)
            {
               // process-local row
               const int rowg = rowOffsets[i] + k;
               const int nnz_k
                       = csr_blocks(i,j)->i[k+1]
                       -csr_blocks(i,j)->i[k];
               const int osk = csr_blocks(i, j)->i[k];

               for (int l = 0; l < nnz_k; ++l)
               {
                  // Find the column process offset
                  // for the block.
#if MFEM_HYPRE_VERSION >= 21600
                  const HYPRE_Int bcol
                          = usingBigJ ?
                          csr_blocks(i, j)->big_j[osk + l] :
                          csr_blocks(i, j)->j[osk + l];
#else
                  const HYPRE_Int bcol
                          = csr_blocks(i, j)->j[osk + l];
#endif

                  // find the processor 'bcolproc' that
                  // holds column 'bcol':
                  const auto &offs = blockColProcOffsets[j];
                  const int bcolproc =
                     std::upper_bound(offs.begin() + 1,
                                      offs.end(), bcol)
                     - offs.begin() - 1;

                  opJ[opI[rowg] + cnt[rowg]]
                          = procColOffsets[bcolproc] +
                          procBlockColOffsets[bcolproc][j]
                          + bcol
                          - blockColProcOffsets[j][bcolproc];
                  data[opI[rowg] + cnt[rowg]]
                          = cij
                          * csr_blocks(i, j)->data[osk + l];
                  cnt[rowg]++;
               }
            }
         }
      }
   }

   for (int i = 0; i < numBlockRows; ++i)
   {
      for (int j = 0; j < numBlockCols; ++j)
      {
         if (csr_blocks(i, j) != nullptr)
         {
            hypre_CSRMatrixDestroy(csr_blocks(i, j));
         }
      }
   }

   std::vector<HYPRE_Int> rowStarts2(2);
   rowStarts2[0] = first_loc_row;
   rowStarts2[1] = first_loc_row + all_num_loc_rows[rank];

   std::vector<HYPRE_Int> colStarts2(2);
   colStarts2[0] = first_loc_col;
   colStarts2[1] = first_loc_col + all_num_loc_cols[rank];

   MFEM_VERIFY(HYPRE_AssumedPartitionCheck(),
               "only 'assumed partition' mode is supported")

   return new HypreParMatrix
           (comm, num_loc_rows, glob_nrows, glob_ncols,
            opI.data(), opJ.data(), data.data(),
            rowStarts2.data(), colStarts2.data());
}

//! Helper function for HypreParMatrixFromBlocks.
//! Note that scalability to extremely large processor
//! counts is limited by the use of MPI_Allgather.
void mymfem::GatherBlockOffsetData
(MPI_Comm comm, const int rank, const int nprocs,
 const int num_loc, const Array<int> &offsets,
 std::vector<int> &all_num_loc, const int numBlocks,
 std::vector<std::vector<HYPRE_Int>> &blockProcOffsets,
 std::vector<HYPRE_Int> &procOffsets,
 std::vector<std::vector<int>> &procBlockOffsets,
 HYPRE_Int &firstLocal, HYPRE_Int &globalNum)
{
   std::vector<std::vector<int>>
           all_block_num_loc(numBlocks);

   MPI_Allgather(&num_loc, 1, MPI_INT,
                 all_num_loc.data(), 1, MPI_INT, comm);

   for (int j = 0; j < numBlocks; ++j)
   {
      all_block_num_loc[j].resize(nprocs);
      blockProcOffsets[j].resize(nprocs);

      const int blockNumRows = offsets[j + 1] - offsets[j];
      MPI_Allgather(&blockNumRows, 1, MPI_INT,
                    all_block_num_loc[j].data(), 1,
                    MPI_INT, comm);
      blockProcOffsets[j][0] = 0;
      for (int i = 0; i < nprocs - 1; ++i)
      {
         blockProcOffsets[j][i + 1]
                 = blockProcOffsets[j][i]
                 + all_block_num_loc[j][i];
      }
   }

   firstLocal = 0;
   globalNum = 0;
   procOffsets[0] = 0;
   for (int i = 0; i < nprocs; ++i)
   {
      globalNum += all_num_loc[i];
      if (rank == 0)
      {
         MFEM_VERIFY(globalNum >= 0,
                     "overflow in global size");
      }
      if (i < rank)
      {
         firstLocal += all_num_loc[i];
      }

      if (i < nprocs - 1)
      {
         procOffsets[i + 1] = procOffsets[i]
                 + all_num_loc[i];
      }

      procBlockOffsets[i].resize(numBlocks);
      procBlockOffsets[i][0] = 0;
      for (int j = 1; j < numBlocks; ++j)
      {
         procBlockOffsets[i][j]
                 = procBlockOffsets[i][j - 1]
                 + all_block_num_loc[j - 1][i];
      }
   }
}


// End of file
