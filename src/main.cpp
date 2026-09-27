// Kessler_Sim - simulation en ligne de commande
//
// Lit l'etat initial (ECI, km et km/s) de tous les objets catalogues depuis
// le fichier produit par spacetrack_export.py, puis propage chaque objet
// avec un integrateur RK4 sous :
//   - gravite terrestre (point materiel + J2)
//   - perturbation lunaire (corps tiers, ephemeride analytique Meeus)
// Le Soleil, la trainee et la pression de radiation ne sont pas modelises.
//
// La dynamique elle-meme vit dans src/core/orbital.cpp, partagee avec le
// viewer temps reel (src/viewer/viewer.cpp).
//
// Compilation : cmake -B build && cmake --build build
//
// Utilisation, DEPUIS LA RACINE DU DEPOT (les chemins sont relatifs au cwd) :
//   kessler_sim [fichier] [duree_h] [pas_s] [periode_sortie_s] [ids_norad] [options]
//
// Options (placables n'importe ou) :
//   --screen             detecte les rapprochements et collisions -> conjunctions.csv
//   --conj-km X          seuil de rapprochement retenu, en km (defaut 5 ; implique --screen)
//   --radius-scale S     multiplie les rayons de collision (defaut 1)
//   --min-vrel V         vitesse relative minimale d'une rencontre, en m/s (defaut 10) ;
//                        en dessous, la paire vole de concert et n'est pas une rencontre
//   --threads N          nombre de threads (defaut : tous les threads logiques)
//
// Fragmentation (NASA Standard Breakup Model, voir src/core/breakup.hpp) :
//   --breakup            les collisions detectees fragmentent les objets (implique --screen)
//   --lc L               plus petit fragment suivi, en m (defaut 0.1)
//   --masses FICHIER     table des masses du catalogue (defaut config/masses.csv)
//   --seed N             graine du tirage des fragments (defaut 1)
//   --impact NORAD       provoque l'impact d'un projectile fictif sur cet objet (implique --breakup)
//   --impact-mass KG     masse du projectile (defaut 10)
//   --impact-vrel KMS    vitesse relative de l'impact (defaut 10)
//   --impact-at H        instant de l'impact, en heures (defaut 0)
//
// Les fichiers produits vont dans output/ : final_state.csv, snapshots,
// conjunctions.csv, breakups.csv (une ligne par fragmentation) et
// population.csv (population d'objets heure par heure, pour suivre une cascade).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/breakup.hpp"
#include "core/collision.hpp"
#include "core/orbital.hpp"
#include "core/threads.hpp"

using namespace kessler;

namespace {

// Nom entre guillemets pour le CSV : certains noms du catalogue contiennent
// des virgules.
std::string csv_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) out += (c == '"') ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    bool screen = false, breakup = false, conj_explicit = false;
    int threads = 0;   // 0 = defaut d'OpenMP
    ScreeningConfig scfg;
    BreakupConfig bcfg;
    std::string masses_path = "config/masses.csv";
    int impact_id = -1;
    double impact_mass = 10.0, impact_vrel = 10.0, impact_at_h = 0.0;
    for (int k = 1; k < argc; ++k) {
        std::string a = argv[k];
        if (a == "--screen") {
            screen = true;
        } else if (a == "--breakup") {
            breakup = screen = true;
        } else if (a == "--lc" && k + 1 < argc) {
            bcfg.lc_m = std::atof(argv[++k]);
        } else if (a == "--masses" && k + 1 < argc) {
            masses_path = argv[++k];
        } else if (a == "--seed" && k + 1 < argc) {
            bcfg.seed = std::strtoull(argv[++k], nullptr, 10);
        } else if (a == "--impact" && k + 1 < argc) {
            impact_id = std::atoi(argv[++k]);
            breakup = screen = true;
        } else if (a == "--impact-mass" && k + 1 < argc) {
            impact_mass = std::atof(argv[++k]);
        } else if (a == "--impact-vrel" && k + 1 < argc) {
            impact_vrel = std::atof(argv[++k]);
        } else if (a == "--impact-at" && k + 1 < argc) {
            impact_at_h = std::atof(argv[++k]);
        } else if (a == "--conj-km" && k + 1 < argc) {
            scfg.threshold_km = std::atof(argv[++k]);
            screen = conj_explicit = true;
        } else if (a == "--radius-scale" && k + 1 < argc) {
            scfg.radius_scale = std::atof(argv[++k]);
        } else if (a == "--threads" && k + 1 < argc) {
            threads = std::atoi(argv[++k]);
        } else if (a == "--min-vrel" && k + 1 < argc) {
            scfg.min_encounter_speed_km_s = std::atof(argv[++k]) * 1e-3;  // saisi en m/s
        } else if (a.rfind("--", 0) == 0) {
            std::cerr << "Option inconnue : " << a << "\n";
            return 1;
        } else {
            pos.push_back(a);
        }
    }

    std::string path = pos.size() > 0 ? pos[0] : "data/satellites_20260801_1000Z.txt";
    double duration = (pos.size() > 1 ? std::atof(pos[1].c_str()) : 24.0) * 3600.0;
    double dt = pos.size() > 2 ? std::atof(pos[2].c_str()) : 10.0;
    double out_every = pos.size() > 3 ? std::atof(pos[3].c_str()) : 0.0;  // 0 = etat final seulement

    // Optionnel : ids NORAD a enregistrer (separes par des virgules) -> trajectories.csv
    // echantillonne toutes les out_every secondes, a la place des snapshots complets.
    std::vector<int> track_ids;
    if (pos.size() > 4) {
        std::stringstream ss(pos[4]);
        std::string tok;
        while (std::getline(ss, tok, ',')) track_ids.push_back(std::atoi(tok.c_str()));
    }

    std::filesystem::create_directories(OUT_DIR);
    int used_threads = configure_threads(threads);

    Simulation sim;
    int dropped = 0;
    sim.objects = load_catalog(path, &dropped);
    sim.epoch_jd = read_epoch_jd(path);
    sim.dt = dt;
    if (sim.objects.empty()) {
        std::cerr << "Aucun objet charge depuis " << path << "\n";
        return 1;
    }
    if (dropped)
        std::cerr << "Attention : " << dropped << " lignes ignorees (format inattendu).\n";

    // Masses du catalogue : fichier de configuration s'il est la, sinon les
    // valeurs par defaut (identiques a celles du fichier livre).
    MassTable masses;
    {
        std::string err;
        if (!std::filesystem::exists(masses_path)) {
            if (breakup) std::cout << masses_path << " absent : masses par defaut.\n";
        } else if (!masses.load(masses_path, &err)) {
            std::cerr << "Table des masses : " << err << "\n";
            return 1;
        }
    }
    assign_masses(sim.objects, masses);
    const size_t catalog_count = sim.objects.size();

    int impact_index = -1;
    if (impact_id >= 0) {
        for (size_t i = 0; i < sim.objects.size(); ++i)
            if (sim.objects[i].id == impact_id) impact_index = static_cast<int>(i);
        if (impact_index < 0) {
            std::cerr << "--impact : aucun objet de NORAD " << impact_id << " dans le catalogue.\n";
            return 1;
        }
    }

    std::cout << sim.objects.size() << " objets charges. Duree " << duration / 3600.0
              << " h, pas " << dt << " s\n"
              << "Threads : " << used_threads << "\n"
              << "Epoch : " << jd_to_utc_string(sim.epoch_jd) << "\n"
              << "Lune a l'epoch : distance " << norm(sim.moon()) << " km\n";

    // Reference pour mesurer la derive du demi-grand axe en fin de course.
    std::vector<State> initial;
    initial.reserve(sim.objects.size());
    for (const auto& o : sim.objects) initial.push_back(o.s);

    long steps = static_cast<long>(std::ceil(duration / dt));
    long next_out = out_every > 0 ? static_cast<long>(out_every / dt) : 0;
    long snap = 0;

    // Les snapshots d'un run precedent, plus nombreux, resteraient dans
    // output/ et le viewer rejouerait un melange de deux simulations.
    if (next_out > 0 && track_ids.empty()) {
        int removed = 0;
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(OUT_DIR, ec)) {
            std::string n = e.path().filename().string();
            if (n.rfind("snapshot_", 0) == 0 && e.path().extension() == ".csv")
                if (std::filesystem::remove(e.path(), ec)) ++removed;
        }
        if (removed)
            std::cout << removed << " snapshots du run precedent supprimes.\n";
    }

    std::vector<size_t> tracked;
    std::ofstream traj, moon_out;
    if (!track_ids.empty()) {
        for (size_t i = 0; i < sim.objects.size(); ++i)
            if (std::find(track_ids.begin(), track_ids.end(), sim.objects[i].id) != track_ids.end())
                tracked.push_back(i);
        traj.open(OUT_DIR + "/trajectories.csv");
        traj << "t_s,id,name,x_km,y_km,z_km\n";
        moon_out.open(OUT_DIR + "/moon.csv");
        moon_out << "t_s,x_km,y_km,z_km\n";
        if (next_out == 0) next_out = std::max(1L, static_cast<long>(60.0 / dt));
    }

    auto record = [&](double t) {
        for (size_t i : tracked) {
            const Object& o = sim.objects[i];
            traj << t << ',' << o.id << ',' << o.name << ','
                 << o.s.r.x << ',' << o.s.r.y << ',' << o.s.r.z << '\n';
        }
        Vec3 m = moon_position(sim.epoch_jd + t / 86400.0);
        moon_out << t << ',' << m.x << ',' << m.y << ',' << m.z << '\n';
    };
    if (!tracked.empty()) record(0.0);

    // --- Detection des rapprochements ---
    std::ofstream conj_out;
    std::vector<State> before;
    long long n_conj = 0, n_coll = 0, total_candidates = 0, total_refined = 0, max_co_orbiting = 0;
    double closest = 1e30, cell_km = 0;
    int closest_i = -1, closest_j = -1;
    double t_closest = 0;
    long long below[4] = {0, 0, 0, 0};           // < 0.1, < 1, < 2, < 5 km
    const double bins[4] = {0.1, 1.0, 2.0, 5.0};
    double t_prop = 0, t_screen = 0, t_grid = 0, t_pairs = 0;
    if (breakup && !conj_explicit) {
        // En mode fragmentation, seules les collisions importent : le seuil
        // se cale sur la plus grande somme de rayons possible. Un seuil plus
        // large ferait affiner des paires pour rien — leur nombre croit comme
        // le carre du seuil — et, dans les nuages de fragments frais, des
        // centaines de milliers de passages entre fragments freres.
        double max_size_m = 0;
        for (const Object& o : sim.objects) max_size_m = std::max(max_size_m, o.size_m);
        scfg.threshold_km = max_size_m * 1e-3 * scfg.radius_scale;
    }
    if (screen) {
        conj_out.open(OUT_DIR + "/conjunctions.csv");
        conj_out << "t_s,id1,id2,name1,name2,miss_km,v_rel_km_s,collision\n";
        conj_out << std::setprecision(10);
        std::cout << "Detection : seuil " << scfg.threshold_km * 1000.0 << " m"
                  << (breakup && !conj_explicit ? " (enveloppe de collision)" : "")
                  << ", rayons x" << scfg.radius_scale << "\n";
    }

    // --- Fragmentation ---
    Fragmentation frag(bcfg);
    std::ofstream breakup_out, pop_out;
    long long n_catastrophic = 0, n_breakups = 0, n_escaped = 0;
    if (breakup) {
        breakup_out.open(OUT_DIR + "/breakups.csv");
        breakup_out << "t_s,event,target_id,target_name,projectile_id,projectile_name,"
                       "catastrophic,v_rel_km_s,energy_j_per_g,target_mass_kg,"
                       "projectile_mass_kg,sbm_mass_kg,expected_fragments,fragments,"
                       "fragments_mass_kg,escaped\n";
        breakup_out << std::setprecision(8);
        pop_out.open(OUT_DIR + "/population.csv");
        pop_out << "t_s,t_days,alive,alive_catalog,alive_fragments,breakups,catastrophic,"
                   "fragments_created\n";
        std::cout << "Fragmentation (SBM) : L_c = " << bcfg.lc_m << " m, graine "
                  << bcfg.seed << "\n";
    }

    auto log_breakup = [&](const BreakupEvent& ev) {
        ++n_breakups;
        if (ev.catastrophic) ++n_catastrophic;
        auto id_of = [&](int idx) { return idx >= 0 ? std::to_string(sim.objects[idx].id) : std::string(); };
        breakup_out << ev.t << ',' << ev.id << ',' << id_of(ev.target) << ','
                    << csv_quote(ev.target_name) << ',' << id_of(ev.projectile) << ','
                    << csv_quote(ev.projectile_name) << ',' << ev.catastrophic << ','
                    << ev.v_rel_km_s << ',' << ev.energy_j_per_g << ',' << ev.target_mass_kg << ','
                    << ev.projectile_mass_kg << ',' << ev.sbm_mass_kg << ','
                    << ev.expected_fragments << ',' << ev.fragments << ','
                    << ev.fragments_mass_kg << ',' << ev.escaped << '\n';
        breakup_out.flush();   // un run long doit pouvoir se suivre en cours de route
        n_escaped += ev.escaped;

        // En cascade, des milliers d'evenements : la console n'en montre que
        // les premiers, le detail est dans breakups.csv.
        const long long MAX_PRINTED = 20;
        if (n_breakups > MAX_PRINTED) {
            if (n_breakups == MAX_PRINTED + 1)
                std::cout << "  ... (suite dans " << OUT_DIR << "/breakups.csv)\n";
            return;
        }
        // Flux local : ne pas laisser std::cout en format fixe pour la suite.
        std::ostringstream msg;
        msg << std::fixed << "  t = " << std::setprecision(2) << ev.t / 3600.0 << " h : "
            << ev.target_name << " / " << ev.projectile_name << ", "
            << std::setprecision(1) << ev.v_rel_km_s << " km/s, "
            << std::setprecision(0) << ev.energy_j_per_g << " J/g -> "
            << (ev.catastrophic ? "catastrophique" : "non catastrophique") << ", "
            << ev.fragments << " fragments\n";
        std::cout << msg.str();
    };

    long last_logged_hour = -1;
    auto log_population = [&]() {
        long alive = 0, alive_frag = 0;
        for (const Object& o : sim.objects)
            if (o.alive) { ++alive; if (o.is_fragment()) ++alive_frag; }
        pop_out << sim.t << ',' << sim.t / 86400.0 << ',' << alive << ',' << alive - alive_frag
                << ',' << alive_frag << ',' << n_breakups << ',' << n_catastrophic << ','
                << frag.fragments_created() << std::endl;   // vide le tampon : suivi en direct
    };
    if (breakup) { log_population(); last_logged_hour = 0; }

    bool impact_done = impact_index < 0;

    using clock = std::chrono::steady_clock;
    for (long n = 0; n < steps; ++n) {
        if (!impact_done && sim.t >= impact_at_h * 3600.0) {
            impact_done = true;
            log_breakup(frag.impact(sim, impact_index, impact_mass, impact_vrel));
        }
        if (screen) {
            before.resize(sim.objects.size());
            for (size_t i = 0; i < sim.objects.size(); ++i) before[i] = sim.objects[i].s;
        }
        double t0 = sim.t;

        auto c0 = clock::now();
        sim.step();
        auto c1 = clock::now();
        t_prop += std::chrono::duration<double>(c1 - c0).count();

        if (screen) {
            ScreeningStats st;
            std::vector<Conjunction> found = screen_step(before, sim.objects, t0, sim.dt, scfg, &st);
            t_screen += std::chrono::duration<double>(clock::now() - c1).count();
            total_candidates += st.candidate_pairs;
            total_refined += st.refined_pairs;
            max_co_orbiting = std::max(max_co_orbiting, st.co_orbiting);
            cell_km = st.cell_km;
            t_grid += st.grid_s;
            t_pairs += st.pairs_s;
            for (const Conjunction& c : found) {
                const Object& a = sim.objects[c.i];
                const Object& b = sim.objects[c.j];
                conj_out << c.t << ',' << a.id << ',' << b.id << ','
                         << csv_quote(a.name) << ',' << csv_quote(b.name) << ','
                         << c.miss_km << ',' << c.v_rel_km_s << ',' << c.collision << '\n';
                ++n_conj;
                if (c.collision) ++n_coll;
                for (int k = 0; k < 4; ++k) if (c.miss_km < bins[k]) ++below[k];
                if (c.miss_km < closest) {
                    closest = c.miss_km; closest_i = c.i; closest_j = c.j; t_closest = c.t;
                }
            }
            if (breakup)
                for (const BreakupEvent& ev : frag.apply(sim, before, t0, sim.dt, found))
                    log_breakup(ev);
        }

        if (breakup && static_cast<long>(sim.t / 3600.0) > last_logged_hour) {
            last_logged_hour = static_cast<long>(sim.t / 3600.0);
            log_population();
        }

        if (next_out > 0 && (n + 1) % next_out == 0) {
            if (!tracked.empty())
                record(sim.t);
            else
                write_snapshot(OUT_DIR + "/snapshot_" + std::to_string(snap++) + ".csv",
                               sim.objects, sim.t);
        }
    }

    write_snapshot(OUT_DIR + "/final_state.csv", sim.objects, sim.t);

    // Bilan sur les objets du catalogue (les fragments n'ont pas d'etat initial).
    long lost = 0, destroyed = 0;
    double max_da = 0;
    int max_id = 0;
    for (size_t i = 0; i < catalog_count; ++i) {
        const Object& o = sim.objects[i];
        if (!o.alive) {
            if (norm(o.s.r) < R_EARTH) ++lost; else ++destroyed;
            continue;
        }
        double da = std::fabs(elements(o.s).a - elements(initial[i]).a);
        if (da > max_da) { max_da = da; max_id = o.id; }
    }
    if (breakup) std::cout << "Objets du catalogue detruits par collision : " << destroyed << "\n";
    std::cout << "Objets sous la surface terrestre : " << lost << "\n"
              << "Plus forte variation de demi-grand axe : " << max_da << " km (NORAD "
              << max_id << ")\n"
              << "Etat final ecrit dans " << OUT_DIR << "/final_state.csv\n";

    if (screen) {
        std::cout << "\n--- Rapprochements (seuil " << scfg.threshold_km << " km) ---\n"
                  << "Retenus : " << n_conj << "   dont collisions : " << n_coll << "\n";
        for (int k = 0; k < 4; ++k)
            if (bins[k] <= scfg.threshold_km)
                std::cout << "  a moins de " << bins[k] << " km : " << below[k] << "\n";
        if (closest_i >= 0) {
            std::cout << "Plus proche : " << closest * 1000.0 << " m entre "
                      << sim.objects[closest_i].name << " (" << sim.objects[closest_i].id << ") et "
                      << sim.objects[closest_j].name << " (" << sim.objects[closest_j].id << "), a t = "
                      << t_closest / 3600.0 << " h\n";
        }
        std::cout << "Paires volant de concert (< " << scfg.min_encounter_speed_km_s * 1e3
                  << " m/s) sous le seuil, non traitees comme rencontres : jusqu'a "
                  << max_co_orbiting << " par pas\n";
        std::cout << "Paires candidates : " << total_candidates / std::max(1L, steps)
                  << " par pas en moyenne (cellule " << cell_km << " km), "
                  << total_refined << " affinees au total\n"
                  << "Temps : propagation " << t_prop << " s, detection " << t_screen
                  << " s (grille " << t_grid << " s, paires " << t_pairs << " s)\n"
                  << "Detail dans " << OUT_DIR << "/conjunctions.csv\n";
    }

    if (breakup) {
        long alive_frag = 0;
        for (size_t i = catalog_count; i < sim.objects.size(); ++i)
            if (sim.objects[i].alive) ++alive_frag;
        std::cout << "\n--- Fragmentation ---\n"
                  << "Evenements : " << n_breakups << " dont " << n_catastrophic
                  << " catastrophiques\n"
                  << "Fragments crees : " << frag.fragments_created() << ", encore en orbite : "
                  << alive_frag << "\n"
                  << "Fragments echappes (trajectoire hyperbolique, ecartes) : " << n_escaped << "\n"
                  << "Detail dans " << OUT_DIR << "/breakups.csv et " << OUT_DIR
                  << "/population.csv\n";
    }
    return 0;
}
