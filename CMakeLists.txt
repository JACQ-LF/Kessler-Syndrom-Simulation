cmake_minimum_required(VERSION 3.16)
project(kessler_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Le viewer tire raylib, Dear ImGui et rlImGui depuis GitHub au moment de la
# configuration. Sans reseau (ou sans besoin du viewer) :
#     cmake -B build -DKESSLER_BUILD_VIEWER=OFF
option(KESSLER_BUILD_VIEWER "Compiler le viewer 3D temps reel" ON)

# Binaires a la racine du depot, la ou les scripts Python les cherchent et
# d'ou les chemins relatifs (data/, output/) sont valides.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         "${CMAKE_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# Coeur orbital, partage par la simulation et le viewer
# ---------------------------------------------------------------------------

add_library(kessler_core STATIC src/core/orbital.cpp)
target_include_directories(kessler_core PUBLIC src)

# OpenMP parallelise la propagation. Facultatif : sans lui tout fonctionne,
# simplement sur un seul coeur.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(kessler_core PUBLIC OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP introuvable — propagation mono-thread.")
endif()

# ---------------------------------------------------------------------------
# Simulation en ligne de commande
# ---------------------------------------------------------------------------

add_executable(kessler_sim src/main.cpp)
target_link_libraries(kessler_sim PRIVATE kessler_core)

# ---------------------------------------------------------------------------
# Viewer 3D
# ---------------------------------------------------------------------------

if(KESSLER_BUILD_VIEWER)
    include(FetchContent)

    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES    OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG        5.5
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(raylib)

    # imgui et rlImGui n'ont pas de CMakeLists exploitable : SOURCE_SUBDIR
    # pointe volontairement vers un repertoire inexistant pour que
    # FetchContent se contente de telecharger les sources.
    # rlImGui utilise l'API de textures introduite par ImGui 1.92 : les deux
    # versions doivent rester accordees. Commit fige plutot que la branche
    # main, pour que la compilation reste reproductible.
    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.92.9b
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  pas-de-cmake)
    FetchContent_MakeAvailable(imgui)

    FetchContent_Declare(rlimgui
        GIT_REPOSITORY https://github.com/raylib-extras/rlImGui.git
        GIT_TAG        1550009359ad975927f7f0e4a3f47e3f27123ea9
        SOURCE_SUBDIR  pas-de-cmake)
    FetchContent_MakeAvailable(rlimgui)

    add_library(imgui_backend STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${rlimgui_SOURCE_DIR}/rlImGui.cpp)
    target_include_directories(imgui_backend PUBLIC
        ${imgui_SOURCE_DIR} ${rlimgui_SOURCE_DIR})
    target_link_libraries(imgui_backend PUBLIC raylib)
    # Evite la dependance aux en-tetes d'icones Font Awesome de rlImGui.
    target_compile_definitions(imgui_backend PUBLIC NO_FONT_AWESOME)

    add_executable(kessler_viewer src/viewer/viewer.cpp)
    target_link_libraries(kessler_viewer PRIVATE kessler_core imgui_backend raylib)

    # On garde volontairement le sous-systeme console : le viewer affiche ses
    # diagnostics de chargement au demarrage, et il se lance depuis un terminal.
endif()
