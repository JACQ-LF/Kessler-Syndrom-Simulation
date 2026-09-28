// Detection des rapprochements et des collisions.
//
// Deux etages, comme en analyse de conjonction :
//   1. Grille de hachage : seules les paires assez proches en debut de pas
//      pour pouvoir se rencontrer pendant le pas sont examinees, puis passees
//      au filtre lineaire (collision_filter.hpp).
//   2. Instant de rapprochement maximal (TCA) calcule analytiquement sur le
//      pas, puis affine par interpolation d'Hermite. Une collision est donc
//      detectee meme si les deux objets se sont traverses entre deux pas :
//      la detection ne depend pas du pas d'integration.
//
// L'etage 1 peut tourner sur GPU (CUDA) quand le binaire est compile avec
// KESSLER_CUDA ; l'etage 2 reste sur CPU, en double precision. Les deux
// backends donnent des resultats identiques au bit pres.

#pragma once

#include <memory>
#include <string>
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
    double pairs_s = 0.0;           // temps de parcours des paires (horloge murale)
    double pairs_busy_s = 0.0;      // somme des temps de travail des threads sur les paires
    double refine_s = 0.0;          // temps d'affinage (inclus dans pairs_s sur CPU)
    double transfer_s = 0.0;        // transferts CPU <-> GPU
    double wait_s = 0.0;            // attente du GPU par le CPU (part non recouverte par la propagation)
    int threads = 1;                // threads engages dans le parcours
    int max_cell_objects = 0;       // objets dans la cellule la plus peuplee
    long long max_cell_pairs = 0;   // paires candidates issues de cette seule cellule
    bool gpu = false;               // parcours fait sur GPU
};

// Rayon de collision equivalent, en km : la moitie de la dimension
// caracteristique de l'objet (Object::size_m).
double collision_radius_km(const Object& o, double scale = 1.0);

// Etat d'un objet a la fraction s (0..1) d'un pas de duree dt, interpole par
// Hermite entre ses etats de debut (a) et de fin (b) de pas.
State interpolate_state(const State& a, const State& b, double dt, double s);

// Moteur de detection. A garder d'un pas a l'autre : il conserve ses
// tampons (grille, table de hachage, listes de travail, memoire GPU) au lieu
// de les reallouer a chaque pas.
class Screener {
public:
    explicit Screener(const ScreeningConfig& cfg = {});
    ~Screener();
    Screener(Screener&&) noexcept;
    Screener& operator=(Screener&&) noexcept;
    Screener(const Screener&) = delete;
    Screener& operator=(const Screener&) = delete;

    ScreeningConfig& config() { return cfg_; }
    const ScreeningConfig& config() const { return cfg_; }

    // Le binaire a-t-il ete compile avec le support GPU (KESSLER_CUDA) ?
    static bool gpu_compiled();

    // Active ou coupe le parcours sur GPU. Renvoie false si c'est impossible
    // (pas de support compile, pas de GPU CUDA) : le CPU reste alors en
    // service. `message` recoit le nom du GPU ou la raison de l'echec.
    bool set_gpu(bool on, std::string* message = nullptr);
    bool gpu() const;

    // Rapprochements survenus pendant le pas [t0, t0 + dt], connaissant les
    // etats en debut de pas (`before`) et en fin de pas (`after`, apres
    // Simulation::step). Chaque rapprochement est rapporte une seule fois,
    // dans le pas qui contient son TCA. Resultat trie par (instant, paire).
    std::vector<Conjunction> screen(const std::vector<State>& before,
                                    const std::vector<Object>& after,
                                    double t0, double dt, ScreeningStats* stats = nullptr);

    // La meme chose en deux temps, pour que le GPU travaille PENDANT que le
    // CPU propage : son pre-filtre n'a besoin que des etats de debut de pas.
    //     screener.begin(before, sim.objects, dt);   // lance le GPU, rend la main
    //     sim.step();                                // CPU et GPU en parallele
    //     screener.finish(before, sim.objects, t0, dt);
    // `objects` dans begin() : l'etat de la simulation AVANT le pas. Sans GPU,
    // begin() ne fait rien et finish() fait tout le travail.
    void begin(const std::vector<State>& before, const std::vector<Object>& objects, double dt);
    std::vector<Conjunction> finish(const std::vector<State>& before,
                                    const std::vector<Object>& after,
                                    double t0, double dt, ScreeningStats* stats = nullptr);

private:
    struct Impl;
    ScreeningConfig cfg_;
    std::unique_ptr<Impl> impl_;
};

// Commodite pour un appel isole : un Screener jetable, sur CPU. Dans une
// boucle, garder un Screener pour ne pas reallouer ses tampons a chaque pas.
std::vector<Conjunction> screen_step(const std::vector<State>& before,
                                     const std::vector<Object>& after,
                                     double t0, double dt,
                                     const ScreeningConfig& cfg,
                                     ScreeningStats* stats = nullptr);

}  // namespace kessler
