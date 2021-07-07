CMAKE_MINIMUM_REQUIRED( VERSION 3.15 )

message ("Recognized hostname:\t" $ENV{HOSTNAME})
message ("Recognized user:\t" $ENV{USER})

project(NumHypSys)

# Manage dependencies
include ("cmake/GetJSON.cmake")
include ("cmake/GetFMT.cmake")
include("cmake/warning_flags.cmake")
include("cmake/clang_tidy.cmake")

# local system: nora
if (($ENV{HOSTNAME} MATCHES "nora") AND ($ENV{USER} MATCHES "prbansal"))
    
    message ("\n\n\t---------------------\t")
    message (    "\tBuild on local system\t")
    message (    "\t---------------------\t\n\n")
    
    # Manage dependencies
    include ("cmake/GetGTest.cmake")
    
    # compile flags
    #set(CMAKE_CXX_COMPILER "gcc")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -O1")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -msse2 -march=native")
    message ("Compiling with:\t" ${CMAKE_CXX_COMPILER})
    message ("CXX_DEBUG flags debug:\t" ${CMAKE_CXX_FLAGS_DEBUG})
    message ("CXX_RELEASE flags release:\t" ${CMAKE_CXX_FLAGS_RELEASE})
    
    # MPI libraries
    set(MPI_INC_DIR "/usr/include/openmpi-x86_64/")
    set(MPI_C_LIB "/usr/lib64/openmpi/lib/libmpi.so")
    set(MPI_CXX_LIB "/usr/lib64/openmpi/lib/libmpi_cxx.so")

    # external libraries
    set(PROJECT_LIB_DIR "$ENV{HOME}/Documents/softwares")
    
    # MFEM
    set(MFEM_DIR "${PROJECT_LIB_DIR}/mfem-4.1/build-parallel")
    set(MFEM_LIB_DIR "${PROJECT_LIB_DIR}/mfem-4.1/build-parallel")
    
    # HYPRE
    set(HYPRE_INC_DIR "${PROJECT_LIB_DIR}/hypre-2.10.0b/src/hypre/include")
    set(HYPRE_LIB_DIR "${PROJECT_LIB_DIR}/hypre-2.10.0b/src/hypre/lib")

    # PARMETIS
    set(PARMETIS_INC_DIR "${PROJECT_LIB_DIR}/parmetis-4.0.3/include")
    set(PARMETIS_LIB_DIR "${PROJECT_LIB_DIR}/parmetis-4.0.3/build/Linux-x86_64/libparmetis")
    
    # METIS
    set(METIS_INC_DIR "${PROJECT_LIB_DIR}/parmetis-4.0.3/metis/include")
    set(METIS_LIB_DIR "${PROJECT_LIB_DIR}/parmetis-4.0.3/build/Linux-x86_64/libmetis")

    # SuperLU_DIST
    #set(SUPERLU_INC_DIR "${PROJECT_LIB_DIR}/SuperLU_DIST_5.1.3/build/include")
    #set(SUPERLU_LIB_DIR "${PROJECT_LIB_DIR}/SuperLU_DIST_5.1.3/build/lib")
    
    # Eigen
    set(EIGEN_INC_DIR "${PROJECT_LIB_DIR}/eigen-3.3.7")
    
    # BLAS and LAPACK for linear algebra
    add_library(linear_algebra INTERFACE)
    target_link_libraries(linear_algebra INTERFACE -lblas -llapack)

elseif (($ENV{USER} MATCHES "prbansal"))
    
    MESSAGE("\n\n\t----------------------\t")
    MESSAGE(    "\tBuild on Euler cluster\t")
    MESSAGE(    "\t----------------------\t\n\n")
    
    # Manage dependencies
    include ("cmake/GetGTest_euler.cmake")
    
    # compile flags
    #set(CMAKE_CXX_COMPILER "gcc")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -O1")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -msse2 -march=native")
    MESSAGE ("Compiling with:\t" ${CMAKE_CXX_COMPILER})
    MESSAGE ("CXX_DEBUG flags debug:\t" ${CMAKE_CXX_FLAGS_DEBUG})
    MESSAGE ("CXX_RELEASE flags release:\t" ${CMAKE_CXX_FLAGS_RELEASE})
    
    # MPI libraries
    set(MPI_INC_DIR "/cluster/apps/gcc-8.2.0/openmpi-4.0.2-vvr7fdofwljfy4qgkkhao36r5qx44hni/include")
    set(MPI_C_LIB "/cluster/apps/gcc-8.2.0/openmpi-4.0.2-vvr7fdofwljfy4qgkkhao36r5qx44hni/lib/libmpi.so")
    set(MPI_CXX_LIB "/cluster/apps/gcc-8.2.0/openmpi-4.0.2-vvr7fdofwljfy4qgkkhao36r5qx44hni/lib/libmpi_cxx.so")

    # external libraries
    set(PROJECT_LIB_DIR "$ENV{HOME}/Documents/libraries")
    
    # MFEM
    set(MFEM_DIR "${PROJECT_LIB_DIR}/mfem-4.1-custom/build-parallel")
    set(MFEM_LIB_DIR "${PROJECT_LIB_DIR}/mfem-4.1-custom/build-parallel")
    
    # HYPRE
    set(HYPRE_INC_DIR "${PROJECT_LIB_DIR}/hypre/src/hypre/include")
    set(HYPRE_LIB_DIR "${PROJECT_LIB_DIR}/hypre/src/hypre/lib")
    
    # PARMETIS
    set(PARMETIS_INC_DIR "${PROJECT_LIB_DIR}/parmetis-4.0.3/include")
    set(PARMETIS_LIB_DIR "${PROJECT_LIB_DIR}/parmetis-4.0.3/build/Linux-x86_64/libparmetis")
    
    # METIS
    set(METIS_INC_DIR "${PROJECT_LIB_DIR}/parmetis-4.0.3/metis/include")
    set(METIS_LIB_DIR "${PROJECT_LIB_DIR}/parmetis-4.0.3/build/Linux-x86_64/libmetis")

    # SuperLU_DIST
    #set(SUPERLU_INC_DIR "${PROJECT_LIB_DIR}/SuperLU_DIST_5.1.3/build/include")
    #set(SUPERLU_LIB_DIR "${PROJECT_LIB_DIR}/SuperLU_DIST_5.1.3/build/lib")
    
    # Eigen
    set(EIGEN_INC_DIR "/cluster/apps/eigen/3.2.1/x86_64/gcc_4.8.2/serial/include/eigen3")
    
    # BLAS and LAPACK for linear algebra
    add_library(linear_algebra INTERFACE)
    target_link_libraries(linear_algebra INTERFACE -L$ENV{OPENBLAS_ROOT}/lib -lopenblas)
    target_link_libraries(linear_algebra INTERFACE -L$ENV{NETLIB_SCALAPACK_ROOT}/lib -lscalapack)

endif()

message ("\nMPI Libraries:")
MESSAGE(${MPI_INC_DIR})
MESSAGE(${MPI_C_LIB})
MESSAGE(${MPI_CXX_LIB})

message ("\nInclude directories:")
message ("MFEM_DIR: " ${MFEM_DIR})
message ("HYPRE_INC_DIR: " ${HYPRE_INC_DIR})
message ("PARMETIS_INC_DIR: " ${PARMETIS_INC_DIR})
message ("METIS_INC_DIR: " ${METIS_INC_DIR})
#message ("SUPERLU_INC_DIR: " ${SUPERLU_INC_DIR})

message ("\nLink libraries:")
message ("MFEM_LIB_DIR: " ${MFEM_LIB_DIR})
message ("HYPRE_LIB_DIR: " ${HYPRE_LIB_DIR})
message ("PARMETIS_LIB_DIR: " ${PARMETIS_LIB_DIR})
message ("METIS_LIB_DIR: " ${METIS_LIB_DIR})
#message ("SUPERLU_LIB_DIR: " ${SUPERLU_LIB_DIR})

include_directories(${MPI_INC_DIR}
                    ${MFEM_DIR}
                    ${HYPRE_INC_DIR}
                    ${METIS_INC_DIR}
                    ${PARMETIS_INC_DIR}
                    #${SUPERLU_INC_DIR}
                    ${EIGEN_INC_DIR})
    
link_directories(${MFEM_LIB_DIR}
                 ${HYPRE_LIB_DIR}
                 ${METIS_LIB_DIR}
                 ${PARMETIS_LIB_DIR})
                 #${SUPERLU_LIB_DIR})

    
##########
## Core ##
##########
add_library(core OBJECT)
target_compile_features(core PUBLIC cxx_std_17)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(core PUBLIC FMT)
target_link_libraries(core PUBLIC JSON)
target_link_libraries(core PUBLIC stdc++fs)
target_link_libraries(core PRIVATE warning_flags)
    
if(CLANG_TIDY_EXE)
    set_target_properties(core PROPERTIES CXX_CLANG_TIDY "${DO_CLANG_TIDY}")
endif()


############
## MyMFEM ##
############
    
# MyMFEM; serial
add_library(mymfem OBJECT)
target_compile_features(mymfem PUBLIC cxx_std_17)
target_include_directories(mymfem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(mymfem PUBLIC core)
target_link_libraries(mymfem PUBLIC linear_algebra)
target_link_libraries(mymfem PRIVATE -lstdc++ -lm -lpthread -lrt -fopenmp
                      ${MPI_C_LIB} ${MPI_CXX_LIB}
                      "${MFEM_LIB_DIR}/libmfem.a" -lmfem
                      "${HYPRE_LIB_DIR}/libHYPRE.a" -lHYPRE
                      "${METIS_LIB_DIR}/libmetis.a" -lmetis)
    
# MyMFEM; parallel
add_library(pmymfem OBJECT)
target_compile_features(pmymfem PUBLIC cxx_std_17)
target_include_directories(pmymfem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(pmymfem PUBLIC core)
target_link_libraries(pmymfem PUBLIC linear_algebra)
target_link_libraries(pmymfem PRIVATE -lstdc++ -lm -lpthread -lrt -fopenmp
                      #"${SUPERLU_LIB_DIR}/libsuperlu_dist.a"
                      "${PARMETIS_LIB_DIR}/libparmetis.a"
                      "${METIS_LIB_DIR}/libmetis.a")

############
## Stokes ##
############
    
# Stokes; serial
add_library(stokesObj OBJECT)
target_link_libraries(stokesObj PUBLIC core)

add_executable(stokes)
target_link_libraries(stokes PRIVATE core)
target_link_libraries(stokes PRIVATE stokesObj)
target_link_libraries(stokes PRIVATE mymfem)
    
    
##################################
## Incompressible Navier-Stokes ##
##################################
    
# Incomp NS; serial
add_library(incompNSObj OBJECT)
target_link_libraries(incompNSObj PUBLIC core)
    
add_executable(incompNS)
target_link_libraries(incompNS PRIVATE core)
target_link_libraries(incompNS PRIVATE incompNSObj)
target_link_libraries(incompNS PRIVATE mymfem)
    
# Incomp NS; parallel
add_library(pincompNSObj OBJECT)
target_link_libraries(pincompNSObj PUBLIC incompNSObj)
    
add_executable(pincompNS)
target_link_libraries(pincompNS PRIVATE core)
target_link_libraries(pincompNS PRIVATE incompNSObj)
target_link_libraries(pincompNS PRIVATE pincompNSObj)
target_link_libraries(pincompNS PRIVATE mymfem)
target_link_libraries(pincompNS PRIVATE pmymfem)

########
## UQ ##
########
    
add_library(uq OBJECT)
target_link_libraries(uq PUBLIC core)


#####################################
## UQ Incompressible Navier-Stokes ##
#####################################
    
# UQ Incomp NS; serial
add_executable(uq_incompNS)
target_link_libraries(uq_incompNS PRIVATE core)
target_link_libraries(uq_incompNS PRIVATE uq)
target_link_libraries(uq_incompNS PRIVATE incompNSObj)
target_link_libraries(uq_incompNS PRIVATE mymfem)

# UQ Incomp NS; parallel
add_executable(uq_pincompNS)
target_link_libraries(uq_pincompNS PRIVATE core)
target_link_libraries(uq_pincompNS PRIVATE uq)
target_link_libraries(uq_pincompNS PRIVATE incompNSObj)
target_link_libraries(uq_pincompNS PRIVATE pincompNSObj)
target_link_libraries(uq_pincompNS PRIVATE mymfem)
target_link_libraries(uq_pincompNS PRIVATE pmymfem)

    
#############
## Testing ##
#############

# Testing, serial
add_executable(unit_tests)
    
target_link_libraries(unit_tests PRIVATE core)
target_link_libraries(unit_tests PRIVATE mymfem)
target_link_libraries(unit_tests PRIVATE uq)
target_link_libraries(unit_tests PRIVATE incompNSObj)
target_link_libraries(unit_tests PRIVATE GTest)


# Testing, parallel
add_executable(punit_tests)
    
target_link_libraries(punit_tests PRIVATE core)
target_link_libraries(punit_tests PRIVATE mymfem)
target_link_libraries(punit_tests PRIVATE pmymfem)
target_link_libraries(punit_tests PRIVATE uq)
target_link_libraries(punit_tests PRIVATE incompNSObj)
target_link_libraries(punit_tests PRIVATE pincompNSObj)
target_link_libraries(punit_tests PRIVATE GTest)
    
#####################
## Post-processing ##
#####################
    
# post-processing, serial
add_executable(post_process)

target_link_libraries(post_process PRIVATE core)
target_link_libraries(post_process PRIVATE mymfem)

# post-processing, parallel
add_executable(ppost_process)

target_link_libraries(ppost_process PRIVATE core)
target_link_libraries(ppost_process PRIVATE uq)
target_link_libraries(ppost_process PRIVATE mymfem)
target_link_libraries(ppost_process PRIVATE pmymfem)

##############
## External ##
##############
    
# computeStats, serial
add_executable(computeStats)

target_link_libraries(computeStats PRIVATE core)
target_link_libraries(computeStats PRIVATE mymfem)
target_link_libraries(computeStats PRIVATE uq)

#
# add sources
add_subdirectory(src)
add_subdirectory(external)
add_subdirectory(tests)
