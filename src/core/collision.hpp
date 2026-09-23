// Detection des rapprochements et des collisions.
//
// Deux etages, comme en analyse de conjonction :
//   1. Grille de hachage : seules les paires assez proches en debut de pas
//      pour pouvoir se rencontrer pendant le pas sont examinees.
//   2. Instant de rapprochement maximal (TCA) calcule analytiquement sur le
//      pas, puis affine par interpolation d'Hermite. Une collision est donc
//      detectee meme si les deux objets se sont traverses entre deux pas :
//      la detection ne depend pas du pas d'integration.

#pragma once

#include <vector>

#include "orbital.hpp"

namespace kessler {

struct ScreeningConfig {
    double threshold_km = 5.0;   // distance en dessous de laquelle un rapprochement est retenu
    double radius_scale = 1.0;   // multiplie tous les rayons de collision

    // En dessous de cette vitesse relative, deux objets volent de concert
    // (lances ensemble, en formation) : leur distance varie a peine d'un pas
    // a l'autre, l'instant de rapprochement n'est pas defini, et un contact
    // ne serait pas une fragmentation hypervéloce. Ces paires sont comptees
    // a part (ScreeningStats::co_orbiting) au lieu d'etre traitees comme des
    // rencontres.
    double min_encounter_speed_km_s = 0.01;
};

struct Conjunction {
    int i = 0, j = 0;            // indices dans Simulation::objects, i < j
    double t = 0.0;              // instant du TCA, en secondes depuis l'epoch
    double miss_km = 0.0;        // distance au TCA
    double v_rel_km_s = 0.0;     // vitesse relative au TCA
    bool collision = false;      // miss_km < R_i + R_j
};

struct ScreeningStats {
    long long candidate_pairs = 0;  // paires issues de la grille
    long long refined_pairs = 0;    // paires passees a l'interpolation d'Hermite
    long long co_orbiting = 0;      // paires sous le seuil de distance mais trop lentes pour une rencontre
    double cell_km = 0.0;           // taille de cellule retenue pour ce pas
    double grid_s = 0.0;            // temps de construction de la grille
    double pairs_s = 0.0;           // temps de parcours des paires
};

// Rayon de collision equivalent, en km, tire de la categorie de section
// radar du catalogue — la seule information de taille disponible.
double collision_radius_km(const Object& o, double scale = 1.0);

// Rapprochements survenus pendant le pas [t0, t0 + dt], connaissant les etats
// en debut de pas (`before`) et en fin de pas (`after`, apres Simulation::step).
// Chaque rapprochement est rapporte une seule fois, dans le pas qui contient
// son TCA. Resultat trie par instant.
std::vector<Conjunction> screen_step(const std::vector<State>& before,
                                     const std::vector<Object>& after,
                                     double t0, double dt,
                                     const ScreeningConfig& cfg,
                                     ScreeningStats* stats = nullptr);

}  // namespace kessler
