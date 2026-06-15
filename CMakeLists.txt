cmake_minimum_required(VERSION 3.18)
project(energy_s3 LANGUAGES CXX)

# ─── C++ standard ─────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Build type ────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ─── Paths ─────────────────────────────────────────────────────────────────
set(REPULSOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Repulsor"
    CACHE PATH "Path to the Repulsor source directory")

# ─── Target ────────────────────────────────────────────────────────────────
add_executable(energy_s3
    "${REPULSOR_DIR}/energy_s3.cpp"
)

# ─── Include dirs ──────────────────────────────────────────────────────────
# Repulsor.hpp is in REPULSOR_DIR; its Tensors sub-includes go through there.
target_include_directories(energy_s3 PRIVATE
    "${REPULSOR_DIR}"
    "${REPULSOR_DIR}/submodules/Tensors"
)

# ─── Compile options ───────────────────────────────────────────────────────
target_compile_options(energy_s3 PRIVATE
    # Optimisation
    -O3
    -march=native
    # Required for Tools.hpp __matrix_type__ attribute
    -fenable-matrix
    # OpenMP (comment out if you want single-threaded)
    # -fopenmp
    # Warnings
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-deprecated-declarations
)

# ─── macOS Accelerate framework ────────────────────────────────────────────
if(APPLE)
    target_compile_definitions(energy_s3 PRIVATE)
    target_link_libraries(energy_s3 PRIVATE
        "-framework Accelerate"
    )
endif()

# ─── AMD reordering (from SuiteSparse) ─────────────────────────────────────
# Uncomment if SuiteSparse is installed:
#   brew install suite-sparse
# option(USE_AMD "Use AMD matrix reordering from SuiteSparse" OFF)
# if(USE_AMD)
#     target_compile_definitions(energy_s3 PRIVATE REPULSOR_USE_AMD)
#     find_package(SuiteSparse REQUIRED COMPONENTS AMD)
#     target_link_libraries(energy_s3 PRIVATE SuiteSparse::AMD)
# endif()

# ─── Linux: OpenBLAS ───────────────────────────────────────────────────────
if(NOT APPLE)
    find_package(BLAS REQUIRED)
    target_link_libraries(energy_s3 PRIVATE BLAS::BLAS)
    target_link_libraries(energy_s3 PRIVATE pthread)
endif()

# ─── Install ───────────────────────────────────────────────────────────────
install(TARGETS energy_s3 DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/build/bin")

# ─── Print summary ─────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== energy_s3 build configuration ===")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Repulsor dir : ${REPULSOR_DIR}")
message(STATUS "  CXX compiler : ${CMAKE_CXX_COMPILER}")
message(STATUS "  Platform     : ${CMAKE_SYSTEM_NAME}")
message(STATUS "")
