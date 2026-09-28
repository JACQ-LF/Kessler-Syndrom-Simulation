// Filtre lineaire d'une paire candidate, partage entre le CPU (collision.cpp)
// et le GPU (collision_gpu.cu).
//
// Les deux backends appellent CETTE fonction, compilee a l'identique : meme
// arithmetique double, memes operations dans le meme ordre, et pas de
// contraction en FMA (nvcc --fmad=false ; GCC et MSVC n'en font pas par
// defaut sur x86-64). Les paires retenues sont donc exactement les memes,
// et les resultats identiques au bit pres quel que soit le backend.

#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define KESSLER_HD __host__ __device__ __forceinline__
#else
#define KESSLER_HD inline
#endif

namespace kessler {

// Deux objets proches subissent presque la meme gravite : leur mouvement
// relatif s'ecarte de la ligne droite seulement via le gradient de gravite,
// d'ordre mu / r^3 (1.4e-6 s^-2 au ras de l'atmosphere). Sur un pas dt, a
// vitesse relative v, l'ecart est borne par GRAVITY_GRADIENT * v * dt^3.
constexpr double GRAVITY_GRADIENT = 2e-6;  // s^-2, arrondi par exces

enum FilterResult : int {
    FILTER_REJECT = 0,       // ne peut pas passer sous le seuil pendant ce pas
    FILTER_REFINE = 1,       // a affiner (interpolation d'Hermite)
    FILTER_CO_ORBITING = 2,  // sous le seuil, mais trop lente pour une rencontre
};

// dr, dv : position et vitesse relatives (objet j moins objet i) en debut de
// pas, en km et km/s. vmin2 : carre de la vitesse relative minimale d'une
// rencontre.
KESSLER_HD int filter_pair(double rx, double ry, double rz,
                           double vx, double vy, double vz,
                           double dt, double threshold_km, double vmin2) {
    const double dv2 = vx * vx + vy * vy + vz * vz;
    const double rv = rx * vx + ry * vy + rz * vz;

    // Instant du rapprochement si le mouvement relatif etait rectiligne,
    // borne au pas.
    double ts = dv2 > 0 ? -rv / dv2 : 0.0;
    if (ts < 0.0) ts = 0.0;
    if (ts > dt) ts = dt;
    const double px = rx + vx * ts, py = ry + vy * ts, pz = rz + vz * ts;

    const double gate = threshold_km + GRAVITY_GRADIENT * sqrt(dv2) * dt * dt * dt + 1e-3;
    if (sqrt(px * px + py * py + pz * pz) > gate) return FILTER_REJECT;

    // Trop lente pour une rencontre : voir ScreeningConfig.
    if (dv2 < vmin2) return FILTER_CO_ORBITING;

    // Paire qui s'eloigne deja en debut de pas. Si sa trajectoire relative
    // est convexe sur le pas, la distance est minimale au debut du pas : le
    // rapprochement, s'il y en a eu un, appartient au pas precedent. La
    // derivee seconde de |p|^2 vaut 2 |dv|^2 + 2 dr.a, avec |a| <= G |dr| :
    // elle est positive des que |dv|^2 > G |dr|^2, ce qu'on verifie. Juste
    // apres une fragmentation, c'est le cas de toutes les paires de fragments
    // freres : les affiner toutes coutait 58 fois un pas normal.
    if (rv >= 0.0 && dv2 > GRAVITY_GRADIENT * (rx * rx + ry * ry + rz * rz)) return FILTER_REJECT;

    return FILTER_REFINE;
}

// Pre-filtre en simple precision, pour le GPU : sur les GPU grand public, la
// double precision tourne a 1/64 de la simple (RTX 3050 : ~85 GFLOPS contre
// ~5 TFLOPS), et le filtre exact en double y etait plus lent que sur CPU.
//
// Il est CONSERVATEUR : il ne rejette une paire que si filter_pair() la
// rejetterait a coup sur. Les paires qu'il garde repassent par filter_pair()
// en double sur CPU, qui fait le tri exact. Le resultat final est donc
// identique au bit pres ; seul le volume de travail change.
//
// Entrees : dr et dv en float, avec des erreurs absolues bornees par
// DR_ERR_KM et DV_ERR_KM_S (arrondis de leur construction, voir le noyau
// GPU : coordonnees relatives au coin de la cellule, vitesses en float).
// La chaine d'une dizaine d'operations en float ajoute une erreur relative
// de l'ordre de 1e-6 ; les marges en prennent dix fois plus.
constexpr float DR_ERR_KM = 5e-4f;     // 50 cm, pour ~10 cm d'erreur reelle
constexpr float DV_ERR_KM_S = 5e-6f;   // 5 mm/s, pour ~1 mm/s d'erreur reelle

KESSLER_HD bool prefilter_keep(float rx, float ry, float rz, float vx, float vy, float vz,
                               float dt, float threshold_km, float vmin2) {
    const float dv2 = vx * vx + vy * vy + vz * vz;
    const float rv = rx * vx + ry * vy + rz * vz;
    const float r2 = rx * rx + ry * ry + rz * rz;
    const float dvn = sqrtf(dv2), rn = sqrtf(r2);

    float ts = dv2 > 0.0f ? -rv / dv2 : 0.0f;
    if (ts < 0.0f) ts = 0.0f;
    if (ts > dt) ts = dt;
    const float px = rx + vx * ts, py = ry + vy * ts, pz = rz + vz * ts;
    const float d = sqrtf(px * px + py * py + pz * pz);

    // Distance : l'erreur sur dr passe telle quelle, celle sur dv est
    // multipliee par l'instant du rapprochement (au plus dt).
    const float gate = threshold_km + static_cast<float>(GRAVITY_GRADIENT) * dvn * dt * dt * dt + 1e-3f;
    const float margin = 1e-5f * (rn + dvn * dt) + DR_ERR_KM + DV_ERR_KM_S * dt;
    if (d > gate + margin) return false;

    // Rejet des paires en eloignement, seulement quand c'est sans
    // ambiguite : nettement plus rapides que le seuil de rencontre (sinon
    // filter_pair les compterait comme co-orbitales), nettement en
    // eloignement compte tenu des erreurs sur dr et dv, et nettement dans le
    // domaine convexe.
    const float rv_tol = 1e-5f * rn * dvn + DR_ERR_KM * dvn + DV_ERR_KM_S * rn;
    if (dv2 > 1.1f * vmin2 && rv > rv_tol &&
        dv2 > 1.01f * static_cast<float>(GRAVITY_GRADIENT) * r2)
        return false;
    return true;
}

// Cles de cellule sur 3 x 21 bits. Avec une cellule d'au moins 1 km, les
// coordonnees jusqu'a 1e6 km (au-dela de l'orbite lunaire) tiennent.
constexpr long long KEY_OFFSET = 1LL << 20;
constexpr unsigned long long NO_CELL = ~0ULL;

KESSLER_HD unsigned long long cell_key(long long ix, long long iy, long long iz) {
    return (static_cast<unsigned long long>(ix + KEY_OFFSET) << 42) |
           (static_cast<unsigned long long>(iy + KEY_OFFSET) << 21) |
            static_cast<unsigned long long>(iz + KEY_OFFSET);
}

KESSLER_HD void cell_coords(unsigned long long key, long long& ix, long long& iy, long long& iz) {
    const unsigned long long mask = (1ULL << 21) - 1;
    ix = static_cast<long long>((key >> 42) & mask) - KEY_OFFSET;
    iy = static_cast<long long>((key >> 21) & mask) - KEY_OFFSET;
    iz = static_cast<long long>(key & mask) - KEY_OFFSET;
}

// Les 13 cellules voisines "en avant" (ordre lexicographique) : combinees aux
// paires internes a une cellule, chaque paire de cellules voisines n'est
// visitee qu'une fois.
#define KESSLER_FORWARD_NEIGHBOURS                                            \
    {{1, -1, -1}, {1, -1, 0}, {1, -1, 1}, {1, 0, -1}, {1, 0, 0}, {1, 0, 1},  \
     {1, 1, -1},  {1, 1, 0},  {1, 1, 1},  {0, 1, -1}, {0, 1, 0}, {0, 1, 1},  \
     {0, 0, 1}}

}  // namespace kessler
