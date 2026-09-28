// Backend GPU de la detection : grille et pre-filtre des paires sur CUDA.
//
// Par pas de simulation, tout en file sur un flux CUDA, sans attente du CPU :
//   1. envoi des etats de debut de pas (position, vitesse) et des vivants,
//      depuis de la memoire hote verrouillee (transfert asynchrone) ;
//   2. cle de cellule de chaque objet, tri par cle (CUB), cellules par
//      compression des plages de cles egales, voisines de chaque cellule ;
//   3. donnees compactes en float dans l'ordre trie (position relative au
//      coin de la cellule, vitesse) ;
//   4. un thread par objet : paires avec sa cellule et les 13 voisines en
//      avant, pre-filtre conservateur en simple precision, paires gardees
//      ajoutees a une liste compacte.
// Pendant ce temps, le CPU propage les orbites. Puis collect() rapatrie la
// liste, que le CPU repasse au filtre exact en double et affine.
//
// Aucune relecture bloquante en cours de route : le nombre de cellules et la
// cellule la plus peuplee restent sur le GPU, lus par les noyaux.

#include "collision_gpu.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "collision_filter.hpp"

namespace kessler {
namespace gpu {

static_assert(sizeof(State) == 6 * sizeof(double), "State doit etre 6 doubles contigus");

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

struct DevState {
    double rx, ry, rz, vx, vy, vz;
};

struct Counters {
    unsigned long long candidates;
    unsigned long long survivors;
    unsigned int max_cell;
    unsigned int pad;
};

// Premier indice k de [0, n) tel que keys[k] >= key (tableau trie).
__device__ int lower_bound(const unsigned long long* keys, int n, unsigned long long key) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (keys[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Cellule de cle `key`, ou -1 si elle est vide.
__device__ int find_cell(const unsigned long long* unique, int ncells, unsigned long long key) {
    int c = lower_bound(unique, ncells, key);
    return (c < ncells && unique[c] == key) ? c : -1;
}

__global__ void compute_keys(const DevState* s, const unsigned char* alive, int n, double cell,
                             unsigned long long* keys, int* idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    idx[i] = i;
    if (!alive[i]) { keys[i] = NO_CELL; return; }
    keys[i] = cell_key(static_cast<long long>(floor(s[i].rx / cell)),
                       static_cast<long long>(floor(s[i].ry / cell)),
                       static_cast<long long>(floor(s[i].rz / cell)));
}

// Les 13 voisines en avant de chaque cellule, calculees une fois par cellule
// plutot qu'une fois par objet (-1 si la voisine est vide). Au passage, la
// population de la cellule la plus chargee (statistique).
__global__ void cell_neighbours(const unsigned long long* unique, const int* counts,
                                const int* num_cells, int* nbr, Counters* counters) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int ncells = *num_cells;
    if (c >= ncells) return;
    atomicMax(&counters->max_cell, static_cast<unsigned int>(counts[c]));
    const int FORWARD[13][3] = KESSLER_FORWARD_NEIGHBOURS;
    long long x, y, z;
    cell_coords(unique[c], x, y, z);
    for (int o = 0; o < 13; ++o)
        nbr[c * 13 + o] = find_cell(unique, ncells, cell_key(x + FORWARD[o][0], y + FORWARD[o][1],
                                                             z + FORWARD[o][2]));
}

// Donnees compactes des objets, dans l'ordre trie par cellule : position
// relative au coin de la cellule et vitesse, en float. La soustraction
// position - coin se fait en double (positions de ~7000 km) ; le resultat,
// inferieur a une cellule (~600 km), tient au centimetre pres en float.
// 32 octets par objet au lieu de 48, et surtout des voisins CONTIGUS en
// memoire : les lectures d'un warp sont groupees au lieu d'etre dispersees.
__global__ void prepare_sorted(const DevState* s, const unsigned long long* sorted_keys,
                               const int* sorted_idx, int alive, double cell,
                               float4* local, float4* vel) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= alive) return;
    const DevState& o = s[sorted_idx[p]];
    long long x, y, z;
    cell_coords(sorted_keys[p], x, y, z);
    local[p] = make_float4(static_cast<float>(o.rx - static_cast<double>(x) * cell),
                           static_cast<float>(o.ry - static_cast<double>(y) * cell),
                           static_cast<float>(o.rz - static_cast<double>(z) * cell), 0.0f);
    vel[p] = make_float4(static_cast<float>(o.vx), static_cast<float>(o.vy),
                         static_cast<float>(o.vz), 0.0f);
}

// Un thread par objet vivant, dans l'ordre trie par cellule. Les threads d'un
// meme warp sont donc dans la meme cellule ou des cellules voisines : ils ont
// des voisinages proches, et des quantites de travail comparables.
__global__ void filter_pairs(const float4* local, const float4* vel,
                             const unsigned long long* sorted_keys, const int* sorted_idx, int alive,
                             const unsigned long long* unique, const int* counts,
                             const int* begins, const int* nbr, const int* num_cells, float cell,
                             float dt, float threshold, float vmin2,
                             int2* pairs, int capacity, Counters* counters) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= alive) return;

    const int c = find_cell(unique, *num_cells, sorted_keys[p]);
    const int begin = begins[c], k = counts[c], u = p - begin;
    const float4 la = local[p], va = vel[p];

    unsigned long long cand = 0;
    // q : position triee du voisin ; (ox, oy, oz) : decalage de sa cellule.
    // dr = decalage * cellule + (position locale du voisin - la sienne),
    // exactement comme dans le test de validation du pre-filtre.
    auto visit = [&](int q, float ox, float oy, float oz) {
        ++cand;
        const float4 lb = local[q], vb = vel[q];
        if (!prefilter_keep(ox * cell + (lb.x - la.x), oy * cell + (lb.y - la.y),
                            oz * cell + (lb.z - la.z), vb.x - va.x, vb.y - va.y, vb.z - va.z,
                            dt, threshold, vmin2))
            return;
        const int a = sorted_idx[p], b = sorted_idx[q];
        unsigned long long slot = atomicAdd(&counters->survivors, 1ULL);
        if (slot < static_cast<unsigned long long>(capacity)) pairs[slot] = make_int2(min(a, b), max(a, b));
    };

    // Paires internes a la cellule, reparties equitablement : l'objet u voit
    // les floor((k-1)/2) objets qui le suivent (en boucle), plus l'objet
    // diametralement oppose si k est pair et u < k/2. Chaque paire une fois,
    // et le meme travail pour tous les threads de la cellule — au lieu d'un
    // triangle ou le premier objet ferait k-1 tests et le dernier aucun.
    const int half = (k - 1) / 2;
    for (int m = 1; m <= half; ++m) visit(begin + (u + m) % k, 0.0f, 0.0f, 0.0f);
    if ((k % 2 == 0) && u < k / 2) visit(begin + u + k / 2, 0.0f, 0.0f, 0.0f);

    // Paires avec les 13 cellules voisines en avant.
    const int FORWARD[13][3] = KESSLER_FORWARD_NEIGHBOURS;
    for (int o = 0; o < 13; ++o) {
        const int nc = nbr[c * 13 + o];
        if (nc < 0) continue;
        const float ox = static_cast<float>(FORWARD[o][0]), oy = static_cast<float>(FORWARD[o][1]),
                    oz = static_cast<float>(FORWARD[o][2]);
        const int b0 = begins[nc], b1 = b0 + counts[nc];
        for (int q = b0; q < b1; ++q) visit(q, ox, oy, oz);
    }

    atomicAdd(&counters->candidates, cand);
}

}  // namespace

// ---------------------------------------------------------------------------

class Context {
public:
    std::string name;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev_start = nullptr, ev_uploaded = nullptr, ev_grid = nullptr, ev_done = nullptr;

    // Memoire GPU
    DevState* d_state = nullptr;
    unsigned char* d_alive = nullptr;
    unsigned long long *d_keys_in = nullptr, *d_keys_out = nullptr, *d_unique = nullptr;
    int *d_idx_in = nullptr, *d_idx_out = nullptr, *d_counts = nullptr, *d_begins = nullptr;
    int* d_nbr = nullptr;
    float4 *d_local = nullptr, *d_vel = nullptr;
    int* d_num_cells = nullptr;
    int2* d_pairs = nullptr;
    Counters* d_counters = nullptr;
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    // Memoire hote verrouillee : seule elle permet des transferts vraiment
    // asynchrones.
    DevState* h_state = nullptr;
    unsigned char* h_alive = nullptr;
    Counters* h_counters = nullptr;

    int cap_n = 0, cap_pairs = 0;

    // Travail en cours (entre launch et collect).
    bool pending = false;
    int alive = 0;
    FilterParams params;

    ~Context() {
        if (stream) cudaStreamSynchronize(stream);
        free_objects();
        cudaFree(d_num_cells); cudaFree(d_pairs); cudaFree(d_counters);
        cudaFreeHost(h_counters);
        if (ev_start) cudaEventDestroy(ev_start);
        if (ev_uploaded) cudaEventDestroy(ev_uploaded);
        if (ev_grid) cudaEventDestroy(ev_grid);
        if (ev_done) cudaEventDestroy(ev_done);
        if (stream) cudaStreamDestroy(stream);
    }

    void free_objects() {
        cudaFree(d_state); cudaFree(d_alive);
        cudaFree(d_keys_in); cudaFree(d_keys_out); cudaFree(d_unique);
        cudaFree(d_idx_in); cudaFree(d_idx_out); cudaFree(d_counts); cudaFree(d_begins);
        cudaFree(d_nbr); cudaFree(d_local); cudaFree(d_vel); cudaFree(d_temp);
        cudaFreeHost(h_state); cudaFreeHost(h_alive);
        d_state = nullptr; d_alive = nullptr;
        d_keys_in = d_keys_out = d_unique = nullptr;
        d_idx_in = d_idx_out = d_counts = d_begins = d_nbr = nullptr;
        d_local = d_vel = nullptr; d_temp = nullptr; temp_bytes = 0;
        h_state = nullptr; h_alive = nullptr;
        cap_n = 0;
    }
};

#define KESSLER_CUDA_CHECK(call)                                                    \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            if (message) *message = std::string(#call) + " : " + cudaGetErrorString(err_); \
            return false;                                                           \
        }                                                                           \
    } while (0)

namespace {

// Dimensionne les tampons pour n objets (avec de la marge : la population
// croit au fil d'une cascade, on evite de reallouer a chaque fragmentation).
bool ensure_objects(Context* ctx, int n, std::string* message) {
    if (n <= ctx->cap_n) return true;
    const int cap = n + n / 2 + 1024;
    ctx->free_objects();
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_state, sizeof(DevState) * cap));
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_alive, cap));
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_keys_in, sizeof(unsigned long long) * cap));
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_keys_out, sizeof(unsigned long long) * cap));
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_unique, sizeof(unsigned long long) * cap));
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_idx_in, sizeof(int) * cap));
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_idx_out, sizeof(int) * cap));
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_counts, sizeof(int) * cap));
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_begins, sizeof(int) * cap));
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_nbr, sizeof(int) * 13 * static_cast<size_t>(cap)));
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_local, sizeof(float4) * cap));
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_vel, sizeof(float4) * cap));
    KESSLER_CUDA_CHECK(cudaHostAlloc(&ctx->h_state, sizeof(DevState) * cap, cudaHostAllocDefault));
    KESSLER_CUDA_CHECK(cudaHostAlloc(&ctx->h_alive, cap, cudaHostAllocDefault));

    // Memoire de travail de CUB : la plus grande des trois operations.
    size_t b1 = 0, b2 = 0, b3 = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, b1, ctx->d_keys_in, ctx->d_keys_out,
                                    ctx->d_idx_in, ctx->d_idx_out, cap);
    cub::DeviceRunLengthEncode::Encode(nullptr, b2, ctx->d_keys_out, ctx->d_unique,
                                       ctx->d_counts, ctx->d_num_cells, cap);
    cub::DeviceScan::ExclusiveSum(nullptr, b3, ctx->d_counts, ctx->d_begins, cap);
    ctx->temp_bytes = b1;
    if (b2 > ctx->temp_bytes) ctx->temp_bytes = b2;
    if (b3 > ctx->temp_bytes) ctx->temp_bytes = b3;
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_temp, ctx->temp_bytes));
    ctx->cap_n = cap;
    return true;
}

bool ensure_pairs(Context* ctx, int capacity, std::string* message) {
    if (capacity <= ctx->cap_pairs) return true;
    cudaFree(ctx->d_pairs);
    ctx->cap_pairs = 0;
    KESSLER_CUDA_CHECK(cudaMalloc(&ctx->d_pairs, sizeof(int2) * capacity));
    ctx->cap_pairs = capacity;
    return true;
}

// Met en file le parcours des paires (les donnees triees sont deja pretes).
bool enqueue_filter(Context* ctx, std::string* message) {
    const int block = 256;
    const FilterParams& p = ctx->params;
    filter_pairs<<<(ctx->alive + block - 1) / block, block, 0, ctx->stream>>>(
        ctx->d_local, ctx->d_vel, ctx->d_keys_out, ctx->d_idx_out, ctx->alive, ctx->d_unique,
        ctx->d_counts, ctx->d_begins, ctx->d_nbr, ctx->d_num_cells, static_cast<float>(p.cell_km),
        static_cast<float>(p.dt), static_cast<float>(p.threshold_km), static_cast<float>(p.vmin2),
        ctx->d_pairs, ctx->cap_pairs, ctx->d_counters);
    KESSLER_CUDA_CHECK(cudaGetLastError());
    KESSLER_CUDA_CHECK(cudaMemcpyAsync(ctx->h_counters, ctx->d_counters, sizeof(Counters),
                                       cudaMemcpyDeviceToHost, ctx->stream));
    return true;
}

}  // namespace

Context* create(std::string* message) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        if (message) *message = "aucun GPU CUDA detecte";
        return nullptr;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        if (message) *message = "impossible de selectionner le GPU 0";
        return nullptr;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    Context* ctx = new Context;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (sm_%d%d, %d SM, %.1f Go)", prop.name, prop.major, prop.minor,
                  prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);
    ctx->name = buf;

    std::string err;
    bool ok = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking) == cudaSuccess &&
              cudaEventCreate(&ctx->ev_start) == cudaSuccess &&
              cudaEventCreate(&ctx->ev_uploaded) == cudaSuccess &&
              cudaEventCreate(&ctx->ev_grid) == cudaSuccess &&
              cudaEventCreate(&ctx->ev_done) == cudaSuccess &&
              cudaMalloc(&ctx->d_counters, sizeof(Counters)) == cudaSuccess &&
              cudaMalloc(&ctx->d_num_cells, sizeof(int)) == cudaSuccess &&
              cudaHostAlloc(&ctx->h_counters, sizeof(Counters), cudaHostAllocDefault) == cudaSuccess &&
              ensure_pairs(ctx, 1 << 16, &err);
    if (!ok) {
        if (message) *message = "initialisation impossible sur " + ctx->name;
        delete ctx;
        return nullptr;
    }
    if (message) *message = ctx->name;
    return ctx;
}

void destroy(Context* ctx) { delete ctx; }

bool staging(Context* ctx, int n, State** states, unsigned char** alive, std::string* message) {
    if (!ensure_objects(ctx, n > 0 ? n : 1, message)) return false;
    *states = reinterpret_cast<State*>(ctx->h_state);
    *alive = ctx->h_alive;
    return true;
}

bool launch(Context* ctx, int n, int alive_count, const FilterParams& params, std::string* message) {
    ctx->pending = false;
    ctx->params = params;
    ctx->alive = alive_count;
    if (n == 0 || alive_count == 0) return true;

    const int block = 256;
    cudaStream_t s = ctx->stream;
    KESSLER_CUDA_CHECK(cudaEventRecord(ctx->ev_start, s));
    KESSLER_CUDA_CHECK(cudaMemcpyAsync(ctx->d_state, ctx->h_state, sizeof(DevState) * n,
                                       cudaMemcpyHostToDevice, s));
    KESSLER_CUDA_CHECK(cudaMemcpyAsync(ctx->d_alive, ctx->h_alive, n, cudaMemcpyHostToDevice, s));
    KESSLER_CUDA_CHECK(cudaEventRecord(ctx->ev_uploaded, s));

    // Grille. Les morts (cle NO_CELL) finissent en queue de tri et sont
    // ignores. Le nombre de cellules n'est connu que du GPU : les operations
    // suivantes travaillent sur `alive_count` elements (majorant), les noyaux
    // lisent le vrai nombre en memoire GPU.
    compute_keys<<<(n + block - 1) / block, block, 0, s>>>(ctx->d_state, ctx->d_alive, n, params.cell_km,
                                                          ctx->d_keys_in, ctx->d_idx_in);
    KESSLER_CUDA_CHECK(cudaGetLastError());
    size_t bytes = ctx->temp_bytes;
    KESSLER_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(ctx->d_temp, bytes, ctx->d_keys_in, ctx->d_keys_out,
                                                       ctx->d_idx_in, ctx->d_idx_out, n, 0, 64, s));
    KESSLER_CUDA_CHECK(cudaMemsetAsync(ctx->d_counts, 0, sizeof(int) * alive_count, s));
    bytes = ctx->temp_bytes;
    KESSLER_CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(ctx->d_temp, bytes, ctx->d_keys_out, ctx->d_unique,
                                                          ctx->d_counts, ctx->d_num_cells, alive_count, s));
    bytes = ctx->temp_bytes;
    KESSLER_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(ctx->d_temp, bytes, ctx->d_counts, ctx->d_begins,
                                                     alive_count, s));
    KESSLER_CUDA_CHECK(cudaMemsetAsync(ctx->d_counters, 0, sizeof(Counters), s));
    cell_neighbours<<<(alive_count + block - 1) / block, block, 0, s>>>(
        ctx->d_unique, ctx->d_counts, ctx->d_num_cells, ctx->d_nbr, ctx->d_counters);
    KESSLER_CUDA_CHECK(cudaGetLastError());
    prepare_sorted<<<(alive_count + block - 1) / block, block, 0, s>>>(
        ctx->d_state, ctx->d_keys_out, ctx->d_idx_out, alive_count, params.cell_km, ctx->d_local, ctx->d_vel);
    KESSLER_CUDA_CHECK(cudaGetLastError());
    KESSLER_CUDA_CHECK(cudaEventRecord(ctx->ev_grid, s));

    // Parcours et pre-filtre des paires.
    if (!enqueue_filter(ctx, message)) return false;
    KESSLER_CUDA_CHECK(cudaEventRecord(ctx->ev_done, s));
    ctx->pending = true;
    return true;
}

bool collect(Context* ctx, std::vector<std::pair<int, int>>& survivors, FilterReport& report,
             std::string* message) {
    survivors.clear();
    report = FilterReport{};
    if (!ctx->pending) return true;
    ctx->pending = false;

    auto t_wait = Clock::now();
    KESSLER_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    report.wait_s = seconds_since(t_wait);

    float ms_upload = 0, ms_grid = 0, ms_filter = 0;
    cudaEventElapsedTime(&ms_upload, ctx->ev_start, ctx->ev_uploaded);
    cudaEventElapsedTime(&ms_grid, ctx->ev_uploaded, ctx->ev_grid);
    cudaEventElapsedTime(&ms_filter, ctx->ev_grid, ctx->ev_done);
    report.upload_s = ms_upload * 1e-3;
    report.grid_s = ms_grid * 1e-3;
    report.kernel_s = ms_filter * 1e-3;

    // Liste debordee (rare : juste apres une grosse fragmentation, ou le
    // pre-filtre ne peut pas trancher pour des fragments encore confondus) :
    // on l'agrandit et on relance le seul parcours, les donnees triees etant
    // toujours en place.
    const unsigned int max_cell = ctx->h_counters->max_cell;  // la relance le remet a zero
    if (ctx->h_counters->survivors > static_cast<unsigned long long>(ctx->cap_pairs)) {
        const int want = static_cast<int>(ctx->h_counters->survivors * 2);
        if (!ensure_pairs(ctx, want, message)) return false;
        auto t_retry = Clock::now();
        KESSLER_CUDA_CHECK(cudaMemsetAsync(ctx->d_counters, 0, sizeof(Counters), ctx->stream));
        if (!enqueue_filter(ctx, message)) return false;
        KESSLER_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        report.kernel_s += seconds_since(t_retry);
    }
    report.candidates = static_cast<long long>(ctx->h_counters->candidates);
    report.max_cell = static_cast<int>(max_cell);

    auto t_down = Clock::now();
    const int ns = static_cast<int>(ctx->h_counters->survivors);
    if (ns > 0) {
        static_assert(sizeof(int2) == sizeof(std::pair<int, int>), "int2 et std::pair<int,int> doivent coincider");
        survivors.resize(ns);
        KESSLER_CUDA_CHECK(cudaMemcpy(survivors.data(), ctx->d_pairs, sizeof(int2) * ns,
                                      cudaMemcpyDeviceToHost));
    }
    report.download_s = seconds_since(t_down);
    return true;
}

}  // namespace gpu
}  // namespace kessler
