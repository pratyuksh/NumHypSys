target_sources(unit_tests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_utilities.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_error.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_RTspace.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_stokes.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_divergence.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_cfl.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_hashTable.cpp
)

target_sources(punit_tests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/punit_tests.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_uq_scheduler.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_uq_sampler.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_uq_stats.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_pdivergence.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_pcfl.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_phashTable.cpp
)
