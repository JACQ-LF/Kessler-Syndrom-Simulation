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
//
// Les fichiers produits vont dans output/.

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

#include "core/collision.hpp"
#include "core/orbital.hpp"

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
    bool screen = false;
    ScreeningConfig scfg;
    for (int k = 1; k < argc; ++k) {
        std::string a = argv[k];
        if (a == "--screen") {
            screen = true;
        } else if (a == "--conj-km" && k + 1 < argc) {
            scfg.threshold_km = std::atof(argv[++k]);
            screen = true;
        } else if (a == "--radius-scale" && k + 1 < argc) {
            scfg.radius_scale = std::atof(argv[++k]);
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

    std::cout << sim.objects.size() << " objets charges. Duree " << duration / 3600.0
              << " h, pas " << dt << " s\n"
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
    if (screen) {
        conj_out.open(OUT_DIR + "/conjunctions.csv");
        conj_out << "t_s,id1,id2,name1,name2,miss_km,v_rel_km_s,collision\n";
        conj_out << std::setprecision(10);
        std::cout << "Detection : seuil " << scfg.threshold_km << " km, rayons x"
                  << scfg.radius_scale << "\n";
    }

    using clock = std::chrono::steady_clock;
    for (long n = 0; n < steps; ++n) {
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

    long lost = 0;
    double max_da = 0;
    int max_id = 0;
    for (size_t i = 0; i < sim.objects.size(); ++i) {
        if (!sim.objects[i].alive) { ++lost; continue; }
        double da = std::fabs(elements(sim.objects[i].s).a - elements(initial[i]).a);
        if (da > max_da) { max_da = da; max_id = sim.objects[i].id; }
    }
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
    return 0;
}
