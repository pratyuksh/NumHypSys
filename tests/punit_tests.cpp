#include <gtest/gtest.h>
#include <mpi.h>


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    int result = 0;
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}


// End of file
