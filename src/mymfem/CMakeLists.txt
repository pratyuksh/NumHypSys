target_sources(mymfem
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/utilities.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/mybilinearform.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/mymixedbilinearform.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/myoperators.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/assembly.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/cell_avgs.cpp
)

target_sources(pmymfem
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/putilities.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/myplinearform.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/mypbilinearform.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/mypmixedbilinearform.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/mypoperators.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/pcell_avgs.cpp
)
