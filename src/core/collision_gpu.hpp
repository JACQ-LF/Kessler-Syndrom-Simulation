// Backend GPU (CUDA) de la detection : construction de la grille et filtre
// des paires candidates. Interne a collision.cpp ; compile seulement avec
// KESSLER_CUDA (voir collision_gpu.cu).
//
// Repartition du travail PAR OBJETS : un thread GPU par objet, les objets
// tries par cellule pour que les threads d'un meme warp aient des voisinages
// proches, donc des quantites de travail comparables. Une repartition par
// cellules donnerait tout le travail d'un nuage de fragments frais a un seul
// thread, et bloquerait les 31 autres de son warp.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "orbital.hpp"

namespace kessler {
namespace gpu {

struct FilterParams {
    double cell_km = 0.0;
    double dt = 0.0;
    double threshold_km = 0.0;
    double vmin2 = 0.0;
};

struct FilterReport {
    long long candidates = 0;
    long long co_orbiting = 0;
    int max_cell = 0;
    double upload_s = 0.0;    // etats envoyes au GPU
    double grid_s = 0.0;      // cles, tri, cellules
    double kernel_s = 0.0;    // parcours et filtre des paires
    double download_s = 0.0;  // paires retenues rapatriees
};

class Context;

// nullptr si aucun GPU CUDA n'est utilisable ; `message` dit pourquoi, ou
// donne le nom du GPU en cas de succes.
Context* create(std::string* message);
void destroy(Context* ctx);

// Filtre sur GPU les paires candidates d'un pas. `survivors` recoit les
// paires (i < j) a affiner, dans un ordre quelconque. Renvoie false en cas
// d'erreur CUDA (message rempli) : l'appelant repasse alors sur CPU.
bool filter(Context* ctx, const State* before, const unsigned char* alive, int n,
            const FilterParams& params, std::vector<std::pair<int, int>>& survivors,
            FilterReport& report, std::string* message);

}  // namespace gpu
}  // namespace kessler
