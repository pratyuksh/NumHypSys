target_sources(stokesObj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/test_cases.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/coefficients.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/assembly.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/discretisation.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/observer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/solver.cpp
)

target_sources(incompNSObj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/assembly.cpp
)
