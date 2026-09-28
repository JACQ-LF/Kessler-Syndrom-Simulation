// Backend GPU (CUDA) de la detection : construction de la grille et
// pre-filtre des paires candidates. Interne a collision.cpp ; compile
// seulement avec KESSLER_CUDA (voir collision_gpu.cu).
//
// Repartition du travail PAR OBJETS : un thread GPU par objet, les objets
// tries par cellule pour que les threads d'un meme warp aient des voisinages
// proches, donc des quantites de travail comparables. Une repartition par
// cellules donnerait tout le travail d'un nuage de fragments frais a un seul
// thread, et bloquerait les 31 autres de son warp.
//
// Asynchrone : launch() met tout le travail en file sur le GPU et rend la
// main aussitot ; collect() attend et rapatrie le resultat. Entre les deux,
// le CPU propage les orbites.

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
    int max_cell = 0;
    double upload_s = 0.0;    // etats envoyes au GPU        (chronometre GPU)
    double grid_s = 0.0;      // cles, tri, cellules          (chronometre GPU)
    double kernel_s = 0.0;    // parcours et pre-filtre       (chronometre GPU)
    double download_s = 0.0;  // paires retenues rapatriees
    double wait_s = 0.0;      // attente du CPU dans collect() : la part non recouverte
};

class Context;

// nullptr si aucun GPU CUDA n'est utilisable ; `message` dit pourquoi, ou
// donne le nom du GPU en cas de succes.
Context* create(std::string* message);
void destroy(Context* ctx);

// Tampons hote verrouilles (seuls a permettre un transfert asynchrone) pour
// n objets : l'appelant y ecrit directement, en parallele, les etats de
// debut de pas et les vivants, sans copie intermediaire.
bool staging(Context* ctx, int n, State** states, unsigned char** alive, std::string* message);

// Met en file le pre-filtre des paires d'un pas, sur les tampons remplis
// apres staging(), et rend la main sans attendre.
bool launch(Context* ctx, int n, int alive_count, const FilterParams& params, std::string* message);

// Attend le travail lance par launch() et rapatrie les paires (i < j) que le
// pre-filtre garde, dans un ordre quelconque. false en cas d'erreur CUDA :
// l'appelant repasse alors sur CPU.
bool collect(Context* ctx, std::vector<std::pair<int, int>>& survivors, FilterReport& report,
             std::string* message);

}  // namespace gpu
}  // namespace kessler
