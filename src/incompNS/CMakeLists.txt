target_sources(incompNSObj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/test_cases.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/test_cases_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/coefficients.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/assembly.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/discretisation.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/preconditioners.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/observer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/solver.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/solver_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/backwardEuler.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/imexEuler.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/imexBdf2.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/bdf1Rk2.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/operators.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/utilities.cpp
)

target_sources(pincompNSObj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/passembly.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/pdiscretisation.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/ppreconditioners.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/pobserver.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/psolver.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/psolver_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/pbackwardEuler.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/pimexEuler.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/pimexBdf2.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/pbdf1Rk2.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/poperators.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/putilities.cpp
)
