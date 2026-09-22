#include "collision.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kessler {

namespace {

// Deux objets proches subissent presque la meme gravite : leur mouvement
// relatif s'ecarte de la ligne droite seulement via le gradient de gravite,
// d'ordre mu / r^3 (1.4e-6 s^-2 au ras de l'atmosphere). Sur un pas dt, a
// vitesse relative v, l'ecart est borne par GRAVITY_GRADIENT * v * dt^3.
constexpr double GRAVITY_GRADIENT = 2e-6;  // s^-2, arrondi par exces

// Cles de cellule sur 3 x 21 bits. Avec une cellule d'au moins 1 km, les
// coordonnees jusqu'a 1e6 km (au-dela de l'orbite lunaire) tiennent.
constexpr int64_t KEY_OFFSET = int64_t(1) << 20;

uint64_t cell_key(int64_t ix, int64_t iy, int64_t iz) {
    return (static_cast<uint64_t>(ix + KEY_OFFSET) << 42) |
           (static_cast<uint64_t>(iy + KEY_OFFSET) << 21) |
            static_cast<uint64_t>(iz + KEY_OFFSET);
}

// Table de hachage plate a adressage ouvert : cle de cellule -> plage
// d'objets. Reconstruite a chaque pas par un tri par comptage en O(N) :
// ni allocation par cellule (une unordered_map en faisait des dizaines de
// milliers par pas), ni tri complet (std::sort coutait 4 fois le reste de la
// construction, alors qu'il suffit de regrouper les objets par cellule).
constexpr uint64_t NO_CELL = ~uint64_t(0);

class CellGrid {
public:
    struct Cell { int64_t x, y, z; int begin, end; };

    std::vector<Cell> cells;   // cellules occupees
    std::vector<int> order;    // indices d'objets, groupes par cellule

    // keys[i] : cle de cellule de l'objet i, ou NO_CELL pour l'ignorer.
    void build(const std::vector<uint64_t>& keys,
               const std::vector<std::array<int64_t, 3>>& coords) {
        const size_t n = keys.size();
        size_t cap = 16;
        while (cap < n * 2) cap <<= 1;
        mask_ = cap - 1;
        slot_key_.assign(cap, NO_CELL);
        slot_cell_.resize(cap);
        cells.clear();
        obj_cell_.resize(n);

        // Passe 1 : une cellule par cle distincte, et effectif de chacune.
        for (size_t i = 0; i < n; ++i) {
            if (keys[i] == NO_CELL) continue;
            size_t h = slot(keys[i]);
            while (slot_key_[h] != NO_CELL && slot_key_[h] != keys[i]) h = (h + 1) & mask_;
            if (slot_key_[h] == NO_CELL) {
                slot_key_[h] = keys[i];
                slot_cell_[h] = static_cast<int>(cells.size());
                cells.push_back({coords[i][0], coords[i][1], coords[i][2], 0, 0});
            }
            int c = slot_cell_[h];
            obj_cell_[i] = c;
            ++cells[c].end;  // provisoirement : effectif
        }

        // Sommes cumulees : plage de chaque cellule dans `order`.
        int run = 0;
        for (Cell& c : cells) { int count = c.end; c.begin = c.end = run; run += count; }

        // Passe 2 : rangement des objets.
        order.resize(run);
        for (size_t i = 0; i < n; ++i)
            if (keys[i] != NO_CELL) order[cells[obj_cell_[i]].end++] = static_cast<int>(i);
    }

    // Plage [debut, fin[ de `order` pour la cellule, vide si inoccupee.
    std::pair<int, int> find(uint64_t key) const {
        size_t h = slot(key);
        while (slot_key_[h] != NO_CELL) {
            if (slot_key_[h] == key) {
                const Cell& c = cells[slot_cell_[h]];
                return {c.begin, c.end};
            }
            h = (h + 1) & mask_;
        }
        return {0, 0};
    }

private:
    size_t slot(uint64_t k) const {
        // Finaliseur de splitmix64 : les cles voisines se dispersent bien.
        k ^= k >> 30; k *= 0xbf58476d1ce4e5b9ULL;
        k ^= k >> 27; k *= 0x94d049bb133111ebULL;
        k ^= k >> 31;
        return static_cast<size_t>(k) & mask_;
    }

    size_t mask_ = 0;
    std::vector<uint64_t> slot_key_;
    std::vector<int> slot_cell_;
    std::vector<int> obj_cell_;
};

// Interpolation cubique d'Hermite de la position relative sur le pas, pour
// s dans [0, 1]. Elle utilise positions ET vitesses aux deux bouts du pas —
// que le RK4 fournit deja — d'ou une erreur en dt^4 au lieu de dt^2.
struct Hermite {
    Vec3 p0, m0, p1, m1;  // m = vitesse relative * dt

    Vec3 pos(double s) const {
        double s2 = s * s, s3 = s2 * s;
        return p0 * (2 * s3 - 3 * s2 + 1) + m0 * (s3 - 2 * s2 + s)
             + p1 * (-2 * s3 + 3 * s2) + m1 * (s3 - s2);
    }
    Vec3 vel(double s) const {  // derivee par rapport a s
        double s2 = s * s;
        return p0 * (6 * s2 - 6 * s) + m0 * (3 * s2 - 4 * s + 1)
             + p1 * (-6 * s2 + 6 * s) + m1 * (3 * s2 - 2 * s);
    }
};

// Minimum de |p(s)|^2 sur [0, 1] par section doree. Sur un pas, le mouvement
// relatif est quasi rectiligne : la fonction est unimodale.
double closest_approach(const Hermite& h) {
    auto f = [&](double s) { Vec3 p = h.pos(s); return dot(p, p); };
    const double g = 0.6180339887498949;
    double a = 0.0, b = 1.0;
    double c = b - g * (b - a), d = a + g * (b - a);
    double fc = f(c), fd = f(d);
    for (int it = 0; it < 60; ++it) {
        if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); }
        else         { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); }
    }
    return 0.5 * (a + b);
}

}  // namespace

double collision_radius_km(const Object& o, double scale) {
    // SMALL < 0.1 m2, MEDIUM 0.1 - 1 m2, LARGE > 1 m2 de section radar.
    // Valeurs indicatives a calibrer ; les objets sans categorie sont
    // traites comme MEDIUM.
    double r_m = 0.4;
    if (o.rcs == "SMALL") r_m = 0.1;
    else if (o.rcs == "LARGE") r_m = 2.0;
    return r_m * 1e-3 * scale;
}

std::vector<Conjunction> screen_step(const std::vector<State>& before,
                                     const std::vector<Object>& after,
                                     double t0, double dt,
                                     const ScreeningConfig& cfg,
                                     ScreeningStats* stats) {
    auto t_start = std::chrono::steady_clock::now();
    const int n = static_cast<int>(before.size());

    // Taille de cellule : deux objets qui se rapprochent a moins du seuil
    // pendant le pas etaient, en debut de pas, a moins de
    // v_rel,max * dt + seuil + ecart a la ligne droite. Avec une cellule au
    // moins aussi grande, ils sont dans la meme cellule ou dans une voisine.
    double vmax = 0.0;
    for (int i = 0; i < n; ++i)
        if (after[i].alive) vmax = std::max(vmax, norm(before[i].v));
    double vrel_max = 2.0 * vmax;
    double cell = std::max(1.0, vrel_max * dt + cfg.threshold_km
                                    + GRAVITY_GRADIENT * vrel_max * dt * dt * dt);

    // --- Grille : cle de cellule de chaque objet vivant, puis regroupement ---
    std::vector<uint64_t> keys(n);
    std::vector<std::array<int64_t, 3>> coords(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        if (!after[i].alive) { keys[i] = NO_CELL; continue; }
        const Vec3& r = before[i].r;
        coords[i] = {static_cast<int64_t>(std::floor(r.x / cell)),
                     static_cast<int64_t>(std::floor(r.y / cell)),
                     static_cast<int64_t>(std::floor(r.z / cell))};
        keys[i] = cell_key(coords[i][0], coords[i][1], coords[i][2]);
    }
    CellGrid grid;
    grid.build(keys, coords);
    const std::vector<int>& order = grid.order;
    auto t_grid = std::chrono::steady_clock::now();

    // --- Parcours des paires ---
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    std::vector<std::vector<Conjunction>> found(nthreads);
    long long candidates = 0, refined = 0, co_orbiting = 0;
    const double vmin2 = cfg.min_encounter_speed_km_s * cfg.min_encounter_speed_km_s;

    // Evalue une paire candidate. Renvoie 0 si rejetee par le filtre lineaire,
    // 1 si elle est passee a l'affinage, 2 si c'est une paire co-orbitale.
    auto test_pair = [&](int a, int b, std::vector<Conjunction>& out) -> int {
        int i = std::min(a, b), j = std::max(a, b);

        // Filtre lineaire, depuis l'etat de debut de pas.
        Vec3 dr = before[j].r - before[i].r;
        Vec3 dv = before[j].v - before[i].v;
        double dv2 = dot(dv, dv);
        double ts = dv2 > 0 ? std::clamp(-dot(dr, dv) / dv2, 0.0, dt) : 0.0;
        double gate = cfg.threshold_km
                    + GRAVITY_GRADIENT * std::sqrt(dv2) * dt * dt * dt + 1e-3;
        if (norm(dr + dv * ts) > gate) return 0;

        // Trop lente pour une rencontre : voir ScreeningConfig.
        if (dv2 < vmin2) return 2;

        // Affinage : interpolation d'Hermite entre debut et fin de pas.
        Hermite h{dr, dv * dt,
                  after[j].s.r - after[i].s.r,
                  (after[j].s.v - after[i].s.v) * dt};
        double s = closest_approach(h);

        // Minimum colle a un bord : le TCA tombe dans le pas precedent
        // (objets deja en eloignement) ou le suivant (encore en approche).
        // Il y sera rapporte, pas ici.
        if (s <= 1e-9 || s >= 1.0 - 1e-9) return 1;

        double miss = norm(h.pos(s));
        if (miss > cfg.threshold_km) return 1;

        Conjunction c;
        c.i = i;
        c.j = j;
        c.t = t0 + s * dt;
        c.miss_km = miss;
        c.v_rel_km_s = norm(h.vel(s)) / dt;
        c.collision = miss < collision_radius_km(after[i], cfg.radius_scale)
                           + collision_radius_km(after[j], cfg.radius_scale);
        out.push_back(c);
        return 1;
    };

    // Les 13 voisines "en avant" (ordre lexicographique) : combinees aux
    // paires internes a chaque cellule, chaque paire de cellules voisines
    // n'est visitee qu'une fois — deux fois moins de recherches qu'en
    // parcourant les 27 voisines de chaque objet, et par cellule plutot que
    // par objet.
    static const int FORWARD[13][3] = {
        {1, -1, -1}, {1, -1, 0}, {1, -1, 1}, {1, 0, -1}, {1, 0, 0}, {1, 0, 1},
        {1, 1, -1},  {1, 1, 0},  {1, 1, 1},  {0, 1, -1}, {0, 1, 0}, {0, 1, 1},
        {0, 0, 1}};

    const int ncells = static_cast<int>(grid.cells.size());
    #pragma omp parallel for schedule(dynamic, 64) reduction(+ : candidates, refined, co_orbiting)
    for (int c = 0; c < ncells; ++c) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        std::vector<Conjunction>& out = found[tid];
        const CellGrid::Cell& cc = grid.cells[c];

        auto visit = [&](int a, int b) {
            ++candidates;
            int r = test_pair(a, b, out);
            if (r == 1) ++refined;
            else if (r == 2) ++co_orbiting;
        };

        for (int p = cc.begin; p < cc.end; ++p)
            for (int q = p + 1; q < cc.end; ++q) visit(order[p], order[q]);

        for (const auto& o : FORWARD) {
            auto range = grid.find(cell_key(cc.x + o[0], cc.y + o[1], cc.z + o[2]));
            for (int p = cc.begin; p < cc.end; ++p)
                for (int q = range.first; q < range.second; ++q) visit(order[p], order[q]);
        }
    }

    auto t_pairs = std::chrono::steady_clock::now();

    std::vector<Conjunction> out;
    for (auto& f : found) out.insert(out.end(), f.begin(), f.end());
    std::sort(out.begin(), out.end(),
              [](const Conjunction& a, const Conjunction& b) { return a.t < b.t; });

    if (stats) {
        stats->candidate_pairs = candidates;
        stats->refined_pairs = refined;
        stats->co_orbiting = co_orbiting;
        stats->cell_km = cell;
        stats->grid_s = std::chrono::duration<double>(t_grid - t_start).count();
        stats->pairs_s = std::chrono::duration<double>(t_pairs - t_grid).count();
    }
    return out;
}

}  // namespace kessler
