#include "breakup.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace kessler {

namespace {

constexpr int NTYPES = static_cast<int>(ObjectType::Count);

// Section radar typique de chaque categorie, pour le rapport A/m des objets
// du catalogue (SMALL, MEDIUM, LARGE, NONE).
constexpr double CATEGORY_AREA_M2[4] = {0.05, 0.5, 5.0, 0.5};

// Masse par defaut d'un objet sans masse renseignee.
constexpr double FALLBACK_MASS_KG = 100.0;

// Densite apparente d'un projectile fictif, pour en deduire une taille :
// un satellite est surtout fait de vide, d'ou une valeur bien sous celle de
// l'aluminium. Ne sert qu'a borner la taille de ses fragments.
constexpr double PROJECTILE_DENSITY_KG_M3 = 300.0;

// Plafond de fragments par evenement : avec un L_c trop petit, la loi de
// puissance en produirait des millions.
constexpr long long MAX_FRAGMENTS_PER_EVENT = 200000;

double mass_or_default(const Object& o) {
    return o.mass_kg > 0 ? o.mass_kg : FALLBACK_MASS_KG;
}

// Fonction affine par morceaux, constante hors de [x0, x1].
double ramp(double x, double x0, double y0, double x1, double y1) {
    if (x <= x0) return y0;
    if (x >= x1) return y1;
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

}  // namespace

// ---------------------------------------------------------------------------
// Masses
// ---------------------------------------------------------------------------

MassTable::MassTable() {
    // Lignes : SMALL, MEDIUM, LARGE, NONE.
    // Colonnes : PAYLOAD, DEBRIS, ROCKET BODY, UNKNOWN (ordre d'ObjectType).
    const double defaults[4][NTYPES] = {
        {5.0, 0.5, 50.0, 1.0},
        {50.0, 5.0, 300.0, 10.0},
        {500.0, 50.0, 2000.0, 100.0},
        {100.0, 5.0, 500.0, 10.0},
    };
    for (int r = 0; r < 4; ++r)
        for (int t = 0; t < NTYPES; ++t) kg_[r][t] = defaults[r][t];
}

int MassTable::rcs_index(const std::string& rcs) {
    if (rcs == "SMALL") return 0;
    if (rcs == "MEDIUM") return 1;
    if (rcs == "LARGE") return 2;
    return 3;
}

double MassTable::mass_for(const std::string& rcs, ObjectType type) const {
    return kg_[rcs_index(rcs)][static_cast<int>(type)];
}

bool MassTable::load(const std::string& path, std::string* error) {
    std::ifstream in(path);
    if (!in) {
        if (error) *error = "impossible d'ouvrir " + path;
        return false;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#' || line.rfind("rcs,", 0) == 0) continue;

        std::stringstream ss(line);
        std::string rcs, type, mass;
        std::getline(ss, rcs, ',');
        std::getline(ss, type, ',');
        std::getline(ss, mass, ',');

        int t = -1;
        for (int k = 0; k < NTYPES; ++k)
            if (type == object_type_name(static_cast<ObjectType>(k))) t = k;
        bool rcs_ok = rcs == "SMALL" || rcs == "MEDIUM" || rcs == "LARGE" || rcs == "NONE";
        double kg = std::atof(mass.c_str());
        if (t < 0 || !rcs_ok || kg <= 0) {
            if (error) *error = path + ", ligne " + std::to_string(lineno) + " invalide : " + line;
            return false;
        }
        kg_[rcs_index(rcs)][t] = kg;
    }
    return true;
}

void assign_masses(std::vector<Object>& objs, const MassTable& table) {
    for (Object& o : objs) {
        if (o.is_fragment()) continue;
        o.mass_kg = table.mass_for(o.rcs, o.type);
        int r = o.rcs == "SMALL" ? 0 : o.rcs == "MEDIUM" ? 1 : o.rcs == "LARGE" ? 2 : 3;
        o.area_to_mass = CATEGORY_AREA_M2[r] / o.mass_kg;
    }
}

// ---------------------------------------------------------------------------
// Fragmentation
// ---------------------------------------------------------------------------

Fragmentation::Fragmentation(const BreakupConfig& cfg) : cfg_(cfg), rng_(cfg.seed) {}

// log10(A/m) d'un fragment de taille 10^log_l, selon le SBM.
double Fragmentation::sample_area_to_mass_log(double log_l) {
    std::normal_distribution<double> n01(0.0, 1.0);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double l = std::pow(10.0, log_l);

    // Fragments de plus de 11 cm : melange de deux lois normales.
    auto large = [&]() {
        double alpha = ramp(log_l, -1.95, 0.0, 0.55, 1.0);
        double mu1 = ramp(log_l, -1.1, -0.6, 0.0, -0.95);
        double s1 = ramp(log_l, -1.3, 0.1, -0.3, 0.3);
        double mu2 = ramp(log_l, -0.7, -1.2, -0.1, -2.0);
        double s2 = ramp(log_l, -0.5, 0.5, -0.3, 0.3);
        return u01(rng_) < alpha ? mu1 + s1 * n01(rng_) : mu2 + s2 * n01(rng_);
    };
    // Fragments de moins de 8 cm : une seule loi normale.
    auto small = [&]() {
        double mu = ramp(log_l, -1.75, -0.3, -1.25, -1.0);
        double s = log_l <= -3.5 ? 0.2 : 0.2 + 0.1333 * (log_l + 3.5);
        return mu + s * n01(rng_);
    };

    if (l >= 0.11) return large();
    if (l <= 0.08) return small();
    // Entre les deux, le SBM raccorde les modeles : on choisit l'un ou l'autre
    // avec une probabilite qui varie lineairement avec la taille.
    return u01(rng_) < (l - 0.08) / 0.03 ? large() : small();
}

BreakupEvent Fragmentation::fragment(Simulation& sim, const Parent& tgt, const Parent& prj,
                                     double t, double remaining_s) {
    BreakupEvent ev;
    ev.id = next_event_++;
    ev.t = t;
    ev.target = tgt.index;
    ev.projectile = prj.index;
    ev.target_name = tgt.name;
    ev.projectile_name = prj.name;
    ev.target_mass_kg = tgt.mass_kg;
    ev.projectile_mass_kg = prj.mass_kg;
    ev.position = tgt.s.r;

    const double vrel = norm(tgt.s.v - prj.s.v);  // km/s
    ev.v_rel_km_s = vrel;
    // 1/2 m_p (1000 v)^2 [J] / (1000 m_t) [g]
    ev.energy_j_per_g = 500.0 * prj.mass_kg * vrel * vrel / tgt.mass_kg;
    ev.catastrophic = ev.energy_j_per_g >= cfg_.catastrophic_j_per_g;

    // Masse que chaque parent apporte aux fragments.
    double budget[2];   // 0 : cible, 1 : projectile
    if (ev.catastrophic) {
        budget[0] = tgt.mass_kg;
        budget[1] = prj.mass_kg;
    } else {
        // M = m_p v^2 : le projectile, plus les ejectas arraches a la cible.
        // Sous le seuil, M < 0.08 m_t : la cible survit.
        budget[1] = prj.mass_kg;
        budget[0] = std::max(0.0, prj.mass_kg * vrel * vrel - prj.mass_kg);
    }
    const double M = budget[0] + budget[1];
    ev.sbm_mass_kg = M;

    // Sort des parents, avant tout ajout (qui pourrait deplacer le tableau).
    if (prj.index >= 0) sim.objects[prj.index].alive = false;
    if (tgt.index >= 0) {
        if (ev.catastrophic) sim.objects[tgt.index].alive = false;
        else sim.objects[tgt.index].mass_kg = std::max(tgt.mass_kg - budget[0], 0.0);
    }

    const double lc = cfg_.lc_m;
    const double a = 1.71;
    ev.expected_fragments = M > 0 ? 0.1 * std::pow(M, 0.75) * std::pow(lc, -a) : 0.0;

    long long n = 0;
    if (ev.expected_fragments > 0) {
        std::poisson_distribution<long long> pois(ev.expected_fragments);
        n = std::min(pois(rng_), MAX_FRAGMENTS_PER_EVENT);
    }

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::normal_distribution<double> n01(0.0, 1.0);
    const Parent* parents[2] = {&tgt, &prj};
    double used[2] = {0.0, 0.0};
    const double p_target = M > 0 ? budget[0] / M : 0.0;
    sim.objects.reserve(sim.objects.size() + static_cast<size_t>(n));

    for (long long k = 0; k < n; ++k) {
        int side = u01(rng_) < p_target ? 0 : 1;
        const Parent& par = *parents[side];

        // Taille : loi de puissance N(> L) ~ L^-1.71, tronquee a [L_c, taille
        // du parent] et tiree par inversion de sa fonction de repartition.
        const double lmax = par.size_m;
        if (lmax <= lc) continue;  // parent plus petit que L_c : rien de suivi
        const double r = std::pow(lmax / lc, -a);
        const double L = lc * std::pow(1.0 - u01(rng_) * (1.0 - r), -1.0 / a);

        const double chi = sample_area_to_mass_log(std::log10(L));  // log10(A/m)
        const double area = 0.540424 * L * L;
        const double m = area / std::pow(10.0, chi);

        // Conservation de la masse : ce parent n'a plus de quoi fournir ce
        // fragment. Les suivants, plus petits, peuvent encore passer.
        if (used[side] + m > budget[side]) continue;
        used[side] += m;

        // Delta-v : norme log-normale, jamais plus que la vitesse d'impact,
        // direction isotrope.
        double dv_ms = std::pow(10.0, 0.9 * chi + 2.9 + 0.4 * n01(rng_));
        dv_ms = std::min(dv_ms, vrel * 1000.0);
        const double cz = 2.0 * u01(rng_) - 1.0;
        const double phi = 2.0 * PI * u01(rng_);
        const double sz = std::sqrt(std::max(0.0, 1.0 - cz * cz));
        const Vec3 dir{sz * std::cos(phi), sz * std::sin(phi), cz};
        const State birth{par.s.r, par.s.v + dir * (dv_ms * 1e-3)};

        // Fragment sur une trajectoire hyperbolique (energie orbitale >= 0) :
        // il quitte l'attraction terrestre et ne fera jamais partie de
        // l'environnement orbital. Le garder serait nuisible en plus d'etre
        // inutile : la grille de detection dimensionne ses cellules sur la
        // vitesse du plus rapide des objets, et ces fragments a plus de
        // 15 km/s les faisaient doubler.
        if (0.5 * dot(birth.v, birth.v) - MU_EARTH / norm(birth.r) >= 0.0) {
            ++ev.escaped;
            continue;
        }

        Object f;
        f.id = next_id_++;
        f.name = "FRAG " + std::to_string(ev.id) + "-" + std::to_string(ev.fragments);
        f.type = ObjectType::Debris;
        f.rcs = area < 0.1 ? "SMALL" : area < 1.0 ? "MEDIUM" : "LARGE";
        f.size_m = L;
        f.mass_kg = m;
        f.area_to_mass = std::pow(10.0, chi);
        f.event = ev.id;
        f.s = sim.propagate_state(birth, t, remaining_s);
        f.alive = norm(f.s.r) >= R_EARTH;
        sim.objects.push_back(std::move(f));

        ++ev.fragments;
        ev.fragments_mass_kg += m;
    }
    fragments_created_ += ev.fragments;
    return ev;
}

std::vector<BreakupEvent> Fragmentation::apply(Simulation& sim, const std::vector<State>& before,
                                               double t0, double dt,
                                               const std::vector<Conjunction>& conjunctions) {
    std::vector<BreakupEvent> out;
    for (const Conjunction& c : conjunctions) {  // triees par instant
        if (!c.collision) continue;
        // Un parent deja detruit plus tot dans ce pas ne peut plus percuter.
        if (!sim.objects[c.i].alive || !sim.objects[c.j].alive) continue;

        // Etats des deux objets a l'instant exact de la collision.
        const double s = (c.t - t0) / dt;
        auto parent = [&](int idx) {
            const Object& o = sim.objects[idx];
            return Parent{idx, interpolate_state(before[idx], o.s, dt, s),
                          mass_or_default(o), o.size_m, o.name};
        };
        Parent a = parent(c.i), b = parent(c.j);
        const Parent& target = a.mass_kg >= b.mass_kg ? a : b;
        const Parent& projectile = a.mass_kg >= b.mass_kg ? b : a;
        out.push_back(fragment(sim, target, projectile, c.t, t0 + dt - c.t));
    }
    return out;
}

BreakupEvent Fragmentation::impact(Simulation& sim, int target, double projectile_mass_kg,
                                   double v_rel_km_s) {
    const Object& o = sim.objects[target];
    const Vec3 r = o.s.r, v = o.s.v;

    // Le projectile a la meme vitesse horizontale que la cible, tournee autour
    // de la verticale locale d'un angle theta : |v_p - v| = 2 |v_h| sin(theta/2).
    const Vec3 k = r * (1.0 / norm(r));
    const double vh = norm(v - k * dot(v, k));
    double theta = v_rel_km_s >= 2.0 * vh ? PI : 2.0 * std::asin(v_rel_km_s / (2.0 * vh));
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(rng_) < 0.5) theta = -theta;  // croisement par un cote ou l'autre
    const Vec3 vp = v * std::cos(theta) + cross(k, v) * std::sin(theta)
                  + k * (dot(k, v) * (1.0 - std::cos(theta)));  // Rodrigues

    Parent tgt{target, o.s, mass_or_default(o), o.size_m, o.name};
    Parent prj{-1, State{r, vp}, projectile_mass_kg,
               std::cbrt(projectile_mass_kg / PROJECTILE_DENSITY_KG_M3), "PROJECTILE"};
    // Au sens du SBM, la cible est la plus lourde des deux.
    if (prj.mass_kg > tgt.mass_kg) return fragment(sim, prj, tgt, sim.t, 0.0);
    return fragment(sim, tgt, prj, sim.t, 0.0);
}

}  // namespace kessler
