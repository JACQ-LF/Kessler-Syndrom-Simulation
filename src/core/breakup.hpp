// Fragmentation par collision : NASA Standard Breakup Model (SBM).
//
// Reference : Johnson, Krisko, Liou, Anz-Meador, "NASA's new breakup model of
// EVOLVE 4.0", Advances in Space Research 28 (9), 2001.
//
// Pour une collision entre une cible (la plus lourde) et un projectile :
//
//   1. Energie specifique E = 1/2 m_p v_rel^2 / m_t. Au-dessus de 40 J/g la
//      collision est catastrophique : les deux objets sont pulverises.
//      Sinon le projectile est detruit et la cible seulement ecornee.
//
//   2. Masse mise en jeu : M = m_t + m_p (catastrophique) ou m_p v_rel^2
//      (v en km/s, sinon). Nombre de fragments plus grands que L :
//            N(> L) = 0.1 M^0.75 L^-1.71
//      On suit ceux au-dessus de L_c ; les plus petits ne sont pas simules.
//
//   3. Chaque fragment : taille L tiree dans cette loi de puissance, rapport
//      surface/masse A/m tire dans la distribution du SBM (fonction de L),
//      masse m = A / (A/m) avec A = 0.540424 L^2.
//
//   4. Vitesse : celle de SON parent plus un increment Delta-v de direction
//      isotrope et de norme log-normale, log10(Dv [m/s]) ~ N(0.9 log10(A/m)
//      + 2.9, 0.4). Les fragments restent donc groupes autour de l'orbite de
//      leur parent — deux nuages distincts, comme observe apres la collision
//      Iridium 33 / Cosmos 2251 en 2009.
//
// Conservation : les fragments sont acceptes tant que leur masse cumulee ne
// depasse pas celle apportee par leur parent ; le reste correspond aux
// debris plus petits que L_c. La quantite de mouvement n'est conservee qu'en
// moyenne (Delta-v isotropes), limite connue du SBM.

#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "collision.hpp"
#include "orbital.hpp"

namespace kessler {

// ---------------------------------------------------------------------------
// Masses des objets du catalogue
// ---------------------------------------------------------------------------

class MassTable {
public:
    MassTable();  // valeurs par defaut, identiques a config/masses.csv

    // Lit un CSV "rcs,type,mass_kg" ; les cases absentes gardent leur valeur.
    // Renvoie false si le fichier est illisible (valeurs conservees).
    bool load(const std::string& path, std::string* error = nullptr);

    double mass_for(const std::string& rcs, ObjectType type) const;

private:
    static int rcs_index(const std::string& rcs);   // 0 SMALL .. 3 NONE
    double kg_[4][static_cast<int>(ObjectType::Count)];
};

// Renseigne masse et rapport A/m des objets du catalogue.
void assign_masses(std::vector<Object>& objs, const MassTable& table);

// ---------------------------------------------------------------------------
// Fragmentation
// ---------------------------------------------------------------------------

// Tirages aleatoires reproductibles d'un compilateur a l'autre. Le standard
// C++ fixe le generateur mt19937_64 bit a bit, mais pas les distributions :
// std::normal_distribution ou std::poisson_distribution different entre
// libstdc++ (MinGW) et la bibliotheque de Microsoft, et la meme graine
// donnait deux cascades differentes selon le compilateur.
class PortableRng {
public:
    explicit PortableRng(std::uint64_t seed) : engine_(seed) {}
    double uniform();                 // [0, 1), 53 bits
    double normal();                  // N(0, 1), Box-Muller
    long long poisson(double mean);   // Knuth (petite moyenne), PTRS de Hormann sinon

private:
    std::mt19937_64 engine_;
};

struct BreakupConfig {
    double lc_m = 0.10;                  // plus petit fragment suivi (m)
    double catastrophic_j_per_g = 40.0;  // seuil d'energie specifique
    std::uint64_t seed = 1;              // graine : un run est reproductible
};

struct BreakupEvent {
    int id = 0;
    double t = 0.0;                      // instant, en s depuis l'epoch
    int target = -1, projectile = -1;    // indices ; projectile = -1 si fictif
    std::string target_name, projectile_name;
    bool catastrophic = false;
    double v_rel_km_s = 0.0;
    double energy_j_per_g = 0.0;
    double target_mass_kg = 0.0, projectile_mass_kg = 0.0;
    double sbm_mass_kg = 0.0;            // M de la loi en nombre
    double expected_fragments = 0.0;     // N(> L_c) donne par la loi
    int fragments = 0;                   // fragments effectivement ajoutes
    double fragments_mass_kg = 0.0;
    int escaped = 0;                     // fragments sur trajectoire hyperbolique, ecartes
    Vec3 position;
};

class Fragmentation {
public:
    explicit Fragmentation(const BreakupConfig& cfg = {});

    BreakupConfig& config() { return cfg_; }
    const BreakupConfig& config() const { return cfg_; }

    // Applique les collisions detectees pendant le pas [t0, t0 + dt] :
    // detruit ou endommage les parents, et ajoute a sim.objects les
    // fragments, deja propages jusqu'a la fin du pas. `before` : etats en
    // debut de pas. Les rapprochements sans collision sont ignores.
    std::vector<BreakupEvent> apply(Simulation& sim, const std::vector<State>& before,
                                    double t0, double dt,
                                    const std::vector<Conjunction>& conjunctions);

    // Provoque, a l'instant courant, l'impact d'un projectile fictif de masse
    // donnee sur l'objet `target`, a la vitesse relative donnee. Le projectile
    // est place sur une orbite de meme vitesse que la cible, croisant sa
    // trajectoire a l'angle qui produit cette vitesse relative.
    BreakupEvent impact(Simulation& sim, int target, double projectile_mass_kg,
                        double v_rel_km_s);

    int events() const { return next_event_; }
    long long fragments_created() const { return fragments_created_; }

private:
    struct Parent {
        int index;          // -1 pour un projectile fictif
        State s;            // etat a l'instant de la collision
        double mass_kg;
        double size_m;
        std::string name;
    };

    BreakupEvent fragment(Simulation& sim, const Parent& target, const Parent& projectile,
                          double t, double remaining_s);

    double sample_area_to_mass_log(double log_l);

    BreakupConfig cfg_;
    PortableRng rng_;
    int next_event_ = 0;
    int next_id_ = FRAGMENT_ID_BASE;
    long long fragments_created_ = 0;
};

}  // namespace kessler
