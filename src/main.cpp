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
//   kessler_sim [fichier] [duree_h] [pas_s] [periode_sortie_s] [ids_norad]
// Les fichiers produits vont dans output/.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/orbital.hpp"

using namespace kessler;

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "data/satellites_20260801_1000Z.txt";
    double duration = (argc > 2 ? std::atof(argv[2]) : 24.0) * 3600.0;
    double dt = argc > 3 ? std::atof(argv[3]) : 10.0;
    double out_every = argc > 4 ? std::atof(argv[4]) : 0.0;  // 0 = etat final seulement

    // Optionnel : ids NORAD a enregistrer (separes par des virgules) -> trajectories.csv
    // echantillonne toutes les out_every secondes, a la place des snapshots complets.
    std::vector<int> track_ids;
    if (argc > 5) {
        std::stringstream ss(argv[5]);
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

    for (long n = 0; n < steps; ++n) {
        sim.step();
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
    return 0;
}
