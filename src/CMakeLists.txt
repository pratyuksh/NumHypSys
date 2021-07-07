add_subdirectory(core)
add_subdirectory(mymfem)
add_subdirectory(stokes)
add_subdirectory(incompNS)
add_subdirectory(uq)

target_sources(stokes
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stokes.cpp
)

target_sources(incompNS
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/incompNS.cpp
)

target_sources(pincompNS
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pincompNS.cpp
)

target_sources(uq_incompNS
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/uq_incompNS.cpp
)

target_sources(uq_pincompNS
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/uq_pincompNS.cpp
)

target_sources(post_process
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/post_process.cpp
)

target_sources(ppost_process
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ppost_process.cpp
)
