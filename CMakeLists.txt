cmake_minimum_required(VERSION 3.16)
project(kessler_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_executable(kessler_sim src/main.cpp)

# OpenMP parallelise la boucle sur les objets. Facultatif : sans lui la
# simulation tourne, simplement sur un seul coeur.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(kessler_sim PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP introuvable — build mono-thread.")
endif()

# Binaire a la racine du depot, la ou les scripts Python le cherchent.
set_target_properties(kessler_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY         "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
)
