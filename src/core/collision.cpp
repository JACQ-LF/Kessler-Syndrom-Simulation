#include "collision.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>

#include "collision_filter.hpp"

#ifdef KESSLER_HAS_CUDA
#include "collision_gpu.hpp"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// Tout le code OpenMP de ce fichier se limite a OpenMP 2.0 (pas de
// reduction max, pas de taches) : c'est ce que MSVC supporte par defaut, et
// le build CUDA sous Windows passe par MSVC.

namespace kessler {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

int thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

int threads_in_team() {
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

// Threads a engager sur une phase lineaire de la grille (cles, tri,
// cellules) : chaque synchronisation coute plus cher que ~15 000 elements
// de travail. Mesure du tri par base sur i7-11800H :
//       cles   1 thread   meilleur
//     28 000    0.61 ms   0.61 ms (1 thread ; 16 threads : 1.68 ms)
//     60 000    1.38 ms   1.00 ms (4 threads)
//    250 000    5.47 ms   2.28 ms (8 threads)
//    500 000   11.92 ms   3.47 ms (16 threads)
int grid_threads(int n) {
    return std::max(1, std::min(thread_count(), n / 15000));
}

// ---------------------------------------------------------------------------
// Affinage
// ---------------------------------------------------------------------------

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

// Affine une paire retenue par le filtre (i < j). Renvoie true et remplit
// `out` si le rapprochement tombe dans ce pas et sous le seuil.
bool refine_pair(const std::vector<State>& before, const std::vector<Object>& after,
                 int i, int j, double t0, double dt, const ScreeningConfig& cfg,
                 Conjunction& out) {
    Hermite h{before[j].r - before[i].r, (before[j].v - before[i].v) * dt,
              after[j].s.r - after[i].s.r, (after[j].s.v - after[i].s.v) * dt};
    double s = closest_approach(h);

    // Minimum colle a un bord : le TCA tombe dans le pas precedent (objets
    // deja en eloignement) ou le suivant (encore en approche). Il y sera
    // rapporte, pas ici.
    if (s <= 1e-9 || s >= 1.0 - 1e-9) return false;

    double miss = norm(h.pos(s));
    if (miss > cfg.threshold_km) return false;

    out.i = i;
    out.j = j;
    out.t = t0 + s * dt;
    out.miss_km = miss;
    out.v_rel_km_s = norm(h.vel(s)) / dt;
    out.collision = miss < collision_radius_km(after[i], cfg.radius_scale)
                         + collision_radius_km(after[j], cfg.radius_scale);
    return true;
}

// Filtre d'une paire depuis les etats de debut de pas.
int filter(const std::vector<State>& before, int i, int j, double dt,
           double threshold_km, double vmin2) {
    const State& a = before[i];
    const State& b = before[j];
    return filter_pair(b.r.x - a.r.x, b.r.y - a.r.y, b.r.z - a.r.z,
                       b.v.x - a.v.x, b.v.y - a.v.y, b.v.z - a.v.z,
                       dt, threshold_km, vmin2);
}

// ---------------------------------------------------------------------------
// Grille, construite en parallele
// ---------------------------------------------------------------------------

struct KeyIdx {
    unsigned long long key;
    int idx;
};

struct Cell {
    unsigned long long key;
    int begin, end;  // plage dans le tableau trie des objets
};

// Tri par base (LSD, octet par octet), parallele et stable. Les octets
// identiques pour toutes les cles sont sautes ; en pratique, des qu'une
// coordonnee change de signe, tous les bits bas de son champ basculent (le
// decalage de 2^20 fait passer de 0x100000 a 0x0FFFFF), et les 8 passes sont
// necessaires. Remplace un tri par comparaison (std::sort coutait a lui seul
// plus que tout le reste de la construction de la grille).
void radix_sort(std::vector<KeyIdx>& a, std::vector<KeyIdx>& tmp,
                std::vector<std::array<int, 256>>& hist, unsigned long long varying) {
    const int n = static_cast<int>(a.size());
    const int team = grid_threads(n);
    tmp.resize(n);
    hist.resize(thread_count());
    for (int shift = 0; shift < 64; shift += 8) {
        if (((varying >> shift) & 0xFFULL) == 0) continue;
        #pragma omp parallel num_threads(team)
        {
            const int t = thread_id(), nt = threads_in_team();
            const int b0 = static_cast<int>(static_cast<long long>(n) * t / nt);
            const int b1 = static_cast<int>(static_cast<long long>(n) * (t + 1) / nt);
            std::array<int, 256>& h = hist[t];
            h.fill(0);
            for (int k = b0; k < b1; ++k) ++h[(a[k].key >> shift) & 0xFF];
            #pragma omp barrier
            #pragma omp single
            {
                // Position de depart de chaque (chiffre, thread) : les chiffres
                // dans l'ordre, et pour un meme chiffre les threads dans
                // l'ordre, ce qui garde le tri stable.
                int run = 0;
                for (int d = 0; d < 256; ++d)
                    for (int tt = 0; tt < nt; ++tt) {
                        int c = hist[tt][d];
                        hist[tt][d] = run;
                        run += c;
                    }
            }
            for (int k = b0; k < b1; ++k) tmp[h[(a[k].key >> shift) & 0xFF]++] = a[k];
        }
        a.swap(tmp);
    }
}

// Table de hachage a adressage ouvert : cle de cellule -> indice de cellule.
// Remplie en parallele par comparaison-echange atomique, reutilisee d'un pas
// a l'autre (seule sa remise a vide est refaite).
class CellTable {
public:
    // Vide la table et la dimensionne pour `capacity` cellules au plus.
    void reset(size_t capacity) {
        size_t cap = 16;
        while (cap < capacity * 2) cap <<= 1;
        if (cap > cap_) {
            keys_.reset(new std::atomic<unsigned long long>[cap]);
            cell_.reset(new int[cap]);
            cap_ = cap;
        }
        mask_ = cap - 1;
        const long long used = static_cast<long long>(cap);
        #pragma omp parallel for schedule(static) num_threads(grid_threads(static_cast<int>(capacity)))
        for (long long s = 0; s < used; ++s) keys_[s].store(NO_CELL, std::memory_order_relaxed);
    }

    // Version sequentielle : indice de la cellule `key`, cree avec l'indice
    // `next` si elle n'existe pas encore (et `created` passe a true).
    int find_or_insert(unsigned long long key, int next, bool& created) {
        size_t h = slot(key);
        for (;;) {
            unsigned long long k = keys_[h].load(std::memory_order_relaxed);
            if (k == key) { created = false; return cell_[h]; }
            if (k == NO_CELL) {
                keys_[h].store(key, std::memory_order_relaxed);
                cell_[h] = next;
                created = true;
                return next;
            }
            h = (h + 1) & mask_;
        }
    }

    // Version parallele : insere toutes les cellules (cles distinctes).
    void build(const std::vector<Cell>& cells) {
        const int ncells = static_cast<int>(cells.size());
        reset(cells.size());
        #pragma omp parallel for schedule(static) num_threads(grid_threads(ncells))
        for (int c = 0; c < ncells; ++c) {
            size_t h = slot(cells[c].key);
            for (;;) {
                unsigned long long expected = NO_CELL;
                if (keys_[h].compare_exchange_strong(expected, cells[c].key,
                                                     std::memory_order_relaxed)) {
                    cell_[h] = c;
                    break;
                }
                h = (h + 1) & mask_;
            }
        }
    }

    // Indice de la cellule de cle `key`, ou -1 si elle est vide.
    int find(unsigned long long key) const {
        size_t h = slot(key);
        for (;;) {
            unsigned long long k = keys_[h].load(std::memory_order_relaxed);
            if (k == key) return cell_[h];
            if (k == NO_CELL) return -1;
            h = (h + 1) & mask_;
        }
    }

private:
    size_t slot(unsigned long long k) const {
        // Finaliseur de splitmix64 : les cles voisines se dispersent bien.
        k ^= k >> 30; k *= 0xbf58476d1ce4e5b9ULL;
        k ^= k >> 27; k *= 0x94d049bb133111ebULL;
        k ^= k >> 31;
        return static_cast<size_t>(k) & mask_;
    }

    std::unique_ptr<std::atomic<unsigned long long>[]> keys_;
    std::unique_ptr<int[]> cell_;
    size_t cap_ = 0, mask_ = 0;
};

}  // namespace

double collision_radius_km(const Object& o, double scale) {
    // La moitie de la dimension caracteristique : 0.1 / 0.4 / 2 m pour les
    // categories SMALL / MEDIUM / LARGE du catalogue, L/2 pour un fragment.
    return 0.5 * o.size_m * 1e-3 * scale;
}

State interpolate_state(const State& a, const State& b, double dt, double s) {
    Hermite h{a.r, a.v * dt, b.r, b.v * dt};
    return {h.pos(s), h.vel(s) * (1.0 / dt)};
}

// ---------------------------------------------------------------------------
// Screener
// ---------------------------------------------------------------------------

struct Screener::Impl {
    // Tampons CPU, conserves d'un pas a l'autre.
    std::vector<KeyIdx> entries, tmp;
    std::vector<std::array<int, 256>> hist;
    std::vector<int> block_count, obj_cell;
    std::vector<Cell> cells;
    CellTable table;
    struct Work { int cell, begin, end; };
    std::vector<Work> work, light;
    std::vector<std::vector<Conjunction>> found;
    std::vector<double> busy;

#ifdef KESSLER_HAS_CUDA
    gpu::Context* gpu = nullptr;
    std::vector<unsigned char> alive;
    std::vector<std::pair<int, int>> survivors;
    ~Impl() { gpu::destroy(gpu); }
#endif

    // Grille et parcours sur CPU.
    void screen_cpu(const std::vector<State>& before, const std::vector<Object>& after,
                    double t0, double dt, double cell, const ScreeningConfig& cfg,
                    ScreeningStats& st);
    // Grille et filtre sur GPU, affinage sur CPU. false en cas d'erreur CUDA.
    bool screen_gpu(const std::vector<State>& before, const std::vector<Object>& after,
                    double t0, double dt, double cell, const ScreeningConfig& cfg,
                    ScreeningStats& st, std::string* message);
};

void Screener::Impl::screen_cpu(const std::vector<State>& before, const std::vector<Object>& after,
                                double t0, double dt, double cell, const ScreeningConfig& cfg,
                                ScreeningStats& st) {
    const auto t_start = Clock::now();
    const int n = static_cast<int>(before.size());
    const int nthreads = thread_count();

    // --- Objets vivants, compactes en (cle, indice) : comptage par bloc,
    //     sommes cumulees, puis ecriture. Chaque bloc note aussi les bits qui
    //     varient entre ses cles, pour sauter les octets constants du tri. ---
    block_count.assign(nthreads + 1, 0);
    entries.resize(n);
    std::vector<unsigned long long> bits(nthreads, 0), firsts(nthreads, NO_CELL);
    int alive = 0;
    #pragma omp parallel num_threads(grid_threads(n))
    {
        const int t = thread_id(), nt = threads_in_team();
        const int b0 = static_cast<int>(static_cast<long long>(n) * t / nt);
        const int b1 = static_cast<int>(static_cast<long long>(n) * (t + 1) / nt);
        int count = 0;
        for (int i = b0; i < b1; ++i) count += after[i].alive ? 1 : 0;
        block_count[t + 1] = count;
        #pragma omp barrier
        #pragma omp single
        {
            for (int tt = 0; tt < nt; ++tt) block_count[tt + 1] += block_count[tt];
            alive = block_count[nt];
        }
        int w = block_count[t];
        unsigned long long first = NO_CELL, v = 0;
        for (int i = b0; i < b1; ++i) {
            if (!after[i].alive) continue;
            const Vec3& r = before[i].r;
            unsigned long long key = cell_key(static_cast<long long>(std::floor(r.x / cell)),
                                              static_cast<long long>(std::floor(r.y / cell)),
                                              static_cast<long long>(std::floor(r.z / cell)));
            entries[w++] = {key, i};
            if (first == NO_CELL) first = key;
            v |= key ^ first;
        }
        firsts[t] = first;
        bits[t] = v;
    }
    entries.resize(alive);

    // Deux facons de regrouper les objets par cellule, qui donnent les memes
    // cellules donc les memes paires. Sous LARGE_GRID objets, un regroupement
    // sequentiel en une passe (table de hachage remplie au passage, comptage,
    // rangement) est le plus rapide : 1.8 ms par pas a 86 000 objets, contre
    // 2.6 ms pour le tri parallele. Au-dela, le tri par base parallele prend
    // le relais : 2.3 ms a 250 000 objets, contre 5.5 ms sur un thread.
    constexpr int LARGE_GRID = 150000;
    int ncells = 0;
    if (alive < LARGE_GRID) {
        table.reset(static_cast<size_t>(alive));
        cells.clear();
        obj_cell.resize(alive);
        for (int k = 0; k < alive; ++k) {
            bool created;
            int c = table.find_or_insert(entries[k].key, static_cast<int>(cells.size()), created);
            if (created) cells.push_back({entries[k].key, 0, 0});
            obj_cell[k] = c;
            ++cells[c].end;  // provisoirement : effectif
        }
        ncells = static_cast<int>(cells.size());
        int run = 0;
        for (Cell& c : cells) { int count = c.end; c.begin = c.end = run; run += count; }
        tmp.resize(alive);
        for (int k = 0; k < alive; ++k) tmp[cells[obj_cell[k]].end++] = entries[k];
        entries.swap(tmp);
        st.grid_s = seconds_since(t_start);
    } else {
        // Un bit varie entre deux cles s'il varie dans l'un des blocs ou entre
        // les premieres cles de deux blocs : (a^c) = (a^f1) ^ (f1^f2) ^ (f2^c).
        unsigned long long varying = 0, ref = NO_CELL;
        for (int t = 0; t < nthreads; ++t) {
            varying |= bits[t];
            if (firsts[t] == NO_CELL) continue;
            if (ref == NO_CELL) ref = firsts[t];
            varying |= firsts[t] ^ ref;
        }
        radix_sort(entries, tmp, hist, varying);

        // Cellules : debuts de plage (changement de cle).
        block_count.assign(nthreads + 1, 0);
        #pragma omp parallel num_threads(grid_threads(alive))
        {
            const int t = thread_id(), nt = threads_in_team();
            const int b0 = static_cast<int>(static_cast<long long>(alive) * t / nt);
            const int b1 = static_cast<int>(static_cast<long long>(alive) * (t + 1) / nt);
            int count = 0;
            for (int k = b0; k < b1; ++k)
                if (k == 0 || entries[k].key != entries[k - 1].key) ++count;
            block_count[t + 1] = count;
            #pragma omp barrier
            #pragma omp single
            {
                for (int tt = 0; tt < nt; ++tt) block_count[tt + 1] += block_count[tt];
                ncells = block_count[nt];
                cells.resize(ncells);
            }
            int w = block_count[t];
            for (int k = b0; k < b1; ++k)
                if (k == 0 || entries[k].key != entries[k - 1].key) cells[w++] = {entries[k].key, k, 0};
        }
        #pragma omp parallel for schedule(static) num_threads(grid_threads(ncells))
        for (int c = 0; c < ncells; ++c) cells[c].end = c + 1 < ncells ? cells[c + 1].begin : alive;

        table.build(cells);
        st.grid_s = seconds_since(t_start);
    }

    // --- Parcours des paires, reparti par OBJETS ---
    // Chaque cellule est coupee en paquets d'au plus CHUNK objets ; un paquet
    // traite les paires de ses objets avec la suite de la cellule et avec les
    // 13 voisines en avant. Les paquets des cellules surpeuplees (un nuage de
    // fragments frais) passent en tete et sont distribues un par un : sinon
    // un seul thread heritait de tout leur travail.
    const auto t_pairs = Clock::now();
    constexpr int CHUNK = 16;
    work.clear();
    light.clear();
    int max_cell = 0;
    for (int c = 0; c < ncells; ++c) {
        const Cell& cc = cells[c];
        const int size = cc.end - cc.begin;
        max_cell = std::max(max_cell, size);
        std::vector<Work>& dst = size > CHUNK ? work : light;
        for (int p = cc.begin; p < cc.end; p += CHUNK)
            dst.push_back({c, p, std::min(p + CHUNK, cc.end)});
    }
    const int nheavy = static_cast<int>(work.size());
    work.insert(work.end(), light.begin(), light.end());
    const int nwork = static_cast<int>(work.size());

    found.resize(nthreads);
    for (auto& f : found) f.clear();
    busy.assign(nthreads, 0.0);
    long long candidates = 0, refined = 0, co_orbiting = 0;
    const double vmin2 = cfg.min_encounter_speed_km_s * cfg.min_encounter_speed_km_s;
    static const int FORWARD[13][3] = KESSLER_FORWARD_NEIGHBOURS;

    #pragma omp parallel
    {
        const int tid = thread_id();
        const auto t_thread = Clock::now();
        std::vector<Conjunction>& out = found[tid];

        // Compteurs propres au thread, sommes une seule fois en sortie : des
        // compteurs partages incrementes par 16 threads a la fois seraient
        // faux (acces concurrents) et lents (une meme ligne de cache).
        long long my_candidates = 0, my_refined = 0, my_co_orbiting = 0;
        auto visit = [&](int a, int b) {
            ++my_candidates;
            const int i = std::min(a, b), j = std::max(a, b);
            const int r = filter(before, i, j, dt, cfg.threshold_km, vmin2);
            if (r == FILTER_CO_ORBITING) { ++my_co_orbiting; return; }
            if (r != FILTER_REFINE) return;
            ++my_refined;
            Conjunction c;
            if (refine_pair(before, after, i, j, t0, dt, cfg, c)) out.push_back(c);
        };
        auto process = [&](const Work& wk) {
            const Cell& cc = cells[wk.cell];
            // Paires internes a la cellule : chaque objet du paquet avec les
            // objets qui le suivent dans la cellule.
            for (int p = wk.begin; p < wk.end; ++p)
                for (int q = p + 1; q < cc.end; ++q) visit(entries[p].idx, entries[q].idx);
            long long x, y, z;
            cell_coords(cc.key, x, y, z);
            for (const auto& o : FORWARD) {
                const int nc = table.find(cell_key(x + o[0], y + o[1], z + o[2]));
                if (nc < 0) continue;
                const Cell& other = cells[nc];
                for (int p = wk.begin; p < wk.end; ++p)
                    for (int q = other.begin; q < other.end; ++q)
                        visit(entries[p].idx, entries[q].idx);
            }
        };

        #pragma omp for schedule(dynamic, 1) nowait
        for (int w = 0; w < nheavy; ++w) process(work[w]);
        #pragma omp for schedule(dynamic, 32) nowait
        for (int w = nheavy; w < nwork; ++w) process(work[w]);
        busy[tid] = seconds_since(t_thread);

        #pragma omp atomic
        candidates += my_candidates;
        #pragma omp atomic
        refined += my_refined;
        #pragma omp atomic
        co_orbiting += my_co_orbiting;
    }

    st.pairs_s = seconds_since(t_pairs);
    st.pairs_busy_s = 0.0;
    for (double b : busy) st.pairs_busy_s += b;
    st.threads = nthreads;
    st.candidate_pairs = candidates;
    st.refined_pairs = refined;
    st.co_orbiting = co_orbiting;
    st.max_cell_objects = max_cell;
    st.max_cell_pairs = static_cast<long long>(max_cell) * (max_cell - 1) / 2;
}

bool Screener::Impl::screen_gpu(const std::vector<State>& before, const std::vector<Object>& after,
                                double t0, double dt, double cell, const ScreeningConfig& cfg,
                                ScreeningStats& st, std::string* message) {
#ifdef KESSLER_HAS_CUDA
    const int n = static_cast<int>(before.size());
    alive.resize(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) alive[i] = after[i].alive ? 1 : 0;

    gpu::FilterParams params;
    params.cell_km = cell;
    params.dt = dt;
    params.threshold_km = cfg.threshold_km;
    params.vmin2 = cfg.min_encounter_speed_km_s * cfg.min_encounter_speed_km_s;
    gpu::FilterReport report;
    if (!gpu::filter(gpu, before.data(), alive.data(), n, params, survivors, report, message))
        return false;

    // Affinage des paires retenues, en double precision, sur CPU : meme code
    // que le backend CPU, donc memes resultats.
    const auto t_refine = Clock::now();
    const int nthreads = thread_count();
    found.resize(nthreads);
    for (auto& f : found) f.clear();
    const int ns = static_cast<int>(survivors.size());
    #pragma omp parallel for schedule(dynamic, 64)
    for (int k = 0; k < ns; ++k) {
        Conjunction c;
        if (refine_pair(before, after, survivors[k].first, survivors[k].second, t0, dt, cfg, c))
            found[thread_id()].push_back(c);
    }
    st.refine_s = seconds_since(t_refine);

    st.gpu = true;
    st.grid_s = report.grid_s;
    st.pairs_s = report.kernel_s + st.refine_s;
    st.transfer_s = report.upload_s + report.download_s;
    st.pairs_busy_s = 0.0;
    st.threads = 0;
    st.candidate_pairs = report.candidates;
    st.refined_pairs = ns;
    st.co_orbiting = report.co_orbiting;
    st.max_cell_objects = report.max_cell;
    st.max_cell_pairs = static_cast<long long>(report.max_cell) * (report.max_cell - 1) / 2;
    return true;
#else
    (void)before; (void)after; (void)t0; (void)dt; (void)cell; (void)cfg; (void)st;
    if (message) *message = "support GPU non compile";
    return false;
#endif
}

Screener::Screener(const ScreeningConfig& cfg) : cfg_(cfg), impl_(new Impl) {}
Screener::~Screener() = default;
Screener::Screener(Screener&&) noexcept = default;
Screener& Screener::operator=(Screener&&) noexcept = default;

bool Screener::gpu_compiled() {
#ifdef KESSLER_HAS_CUDA
    return true;
#else
    return false;
#endif
}

bool Screener::set_gpu(bool on, std::string* message) {
#ifdef KESSLER_HAS_CUDA
    if (!on) {
        gpu::destroy(impl_->gpu);
        impl_->gpu = nullptr;
        return true;
    }
    if (impl_->gpu) return true;
    impl_->gpu = gpu::create(message);
    return impl_->gpu != nullptr;
#else
    if (on && message)
        *message = "binaire compile sans CUDA (option CMake KESSLER_CUDA, build MSVC)";
    return !on;
#endif
}

bool Screener::gpu() const {
#ifdef KESSLER_HAS_CUDA
    return impl_->gpu != nullptr;
#else
    return false;
#endif
}

std::vector<Conjunction> Screener::screen(const std::vector<State>& before,
                                          const std::vector<Object>& after,
                                          double t0, double dt, ScreeningStats* stats) {
    ScreeningStats st;
    const int n = static_cast<int>(before.size());

    // Taille de cellule : deux objets qui se rapprochent a moins du seuil
    // pendant le pas etaient, en debut de pas, a moins de
    // v_rel,max * dt + seuil + ecart a la ligne droite. Avec une cellule au
    // moins aussi grande, ils sont dans la meme cellule ou dans une voisine.
    std::vector<double> vmax_thread(thread_count(), 0.0);
    #pragma omp parallel
    {
        double v = 0.0;
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i)
            if (after[i].alive) v = std::max(v, norm(before[i].v));
        vmax_thread[thread_id()] = v;
    }
    double vmax = 0.0;
    for (double v : vmax_thread) vmax = std::max(vmax, v);
    const double vrel_max = 2.0 * vmax;
    const double cell = std::max(1.0, vrel_max * dt + cfg_.threshold_km
                                          + GRAVITY_GRADIENT * vrel_max * dt * dt * dt);
    st.cell_km = cell;

    bool done = false;
    if (gpu()) {
        std::string message;
        done = impl_->screen_gpu(before, after, t0, dt, cell, cfg_, st, &message);
        // Une erreur CUDA ne doit pas arreter la simulation : on repasse
        // definitivement sur CPU.
        if (!done) set_gpu(false);
    }
    if (!done) impl_->screen_cpu(before, after, t0, dt, cell, cfg_, st);

    std::vector<Conjunction> out;
    for (auto& f : impl_->found) out.insert(out.end(), f.begin(), f.end());
    // Tri par instant, puis par paire : l'ordre ne doit dependre ni du nombre
    // de threads ni du backend.
    std::sort(out.begin(), out.end(), [](const Conjunction& a, const Conjunction& b) {
        if (a.t != b.t) return a.t < b.t;
        if (a.i != b.i) return a.i < b.i;
        return a.j < b.j;
    });
    if (stats) *stats = st;
    return out;
}

std::vector<Conjunction> screen_step(const std::vector<State>& before,
                                     const std::vector<Object>& after,
                                     double t0, double dt,
                                     const ScreeningConfig& cfg,
                                     ScreeningStats* stats) {
    Screener screener(cfg);
    return screener.screen(before, after, t0, dt, stats);
}

}  // namespace kessler
