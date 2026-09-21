// Kessler_Sim - propagation orbitale simple dans le systeme Terre-Lune
//
// Lit l'etat initial (ECI, km et km/s) de tous les objets catalogues depuis
// le fichier produit par spacetrack_export.py, puis propage chaque objet
// avec un integrateur RK4 sous :
//   - gravite terrestre (point materiel + J2)
//   - perturbation lunaire (corps tiers, ephemeride analytique Meeus)
// Le Soleil, la trainee et la pression de radiation ne sont pas modelises.
//
// Compilation : cmake -B build && cmake --build build --config Release
//               (ou : g++ -O2 -std=c++17 -fopenmp src/main.cpp -o kessler_sim)
//
// Utilisation, DEPUIS LA RACINE DU DEPOT (les chemins sont relatifs au cwd) :
//   kessler_sim [fichier] [duree_h] [pas_s] [periode_sortie_s] [ids_norad]
// Les fichiers produits vont dans OUT_DIR (voir plus bas).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double DEG = PI / 180.0;

constexpr double MU_EARTH = 398600.4418;   // km^3/s^2
constexpr double MU_MOON = 4902.800066;    // km^3/s^2
constexpr double R_EARTH = 6378.137;       // km (equatorial)
constexpr double R_MOON = 1737.4;          // km
constexpr double J2 = 1.08262668e-3;

const std::string OUT_DIR = "output";   // dossier des fichiers produits

// Epoch commun du fichier d'entree : 2026-08-01 10:00:00 UTC
constexpr double JD_EPOCH = 2461253.916666667;

struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
};
double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

struct State { Vec3 r, v; };

struct Object {
    int id = 0;
    std::string name, type, rcs;
    State s;
    bool alive = true;   // false une fois sous la surface terrestre
};

// ---------------------------------------------------------------------------
// Ephemeride lunaire (Meeus, chap. 47, termes principaux) -> ECI equatorial
// ---------------------------------------------------------------------------
Vec3 moon_position(double jd) {
    double T = (jd - 2451545.0) / 36525.0;
    auto wrap = [](double d) { return std::fmod(std::fmod(d, 360.0) + 360.0, 360.0); };

    double Lp = wrap(218.3164477 + 481267.88123421 * T);  // longitude moyenne
    double D  = wrap(297.8501921 + 445267.1114034 * T);   // elongation moyenne
    double M  = wrap(357.5291092 + 35999.0502909 * T);    // anomalie moyenne du Soleil
    double Mp = wrap(134.9633964 + 477198.8675055 * T);   // anomalie moyenne de la Lune
    double F  = wrap(93.2720950 + 483202.0175233 * T);    // argument de latitude
    D *= DEG; M *= DEG; Mp *= DEG; F *= DEG;

    // Longitude ecliptique (deg), latitude (deg), distance (km)
    double lon = Lp + 6.288774 * std::sin(Mp) + 1.274027 * std::sin(2 * D - Mp)
               + 0.658314 * std::sin(2 * D) + 0.213618 * std::sin(2 * Mp)
               - 0.185116 * std::sin(M) - 0.114332 * std::sin(2 * F)
               + 0.058793 * std::sin(2 * D - 2 * Mp) + 0.057066 * std::sin(2 * D - M - Mp)
               + 0.053322 * std::sin(2 * D + Mp) + 0.045758 * std::sin(2 * D - M);
    double lat = 5.128122 * std::sin(F) + 0.280602 * std::sin(Mp + F)
               + 0.277693 * std::sin(Mp - F) + 0.173237 * std::sin(2 * D - F)
               + 0.055413 * std::sin(2 * D - Mp + F) + 0.046271 * std::sin(2 * D - Mp - F);
    double dist = 385000.56 - 20905.355 * std::cos(Mp) - 3699.111 * std::cos(2 * D - Mp)
                - 2955.968 * std::cos(2 * D) - 569.925 * std::cos(2 * Mp)
                + 48.888 * std::cos(M) - 3.149 * std::cos(2 * F)
                + 246.158 * std::cos(2 * D - 2 * Mp) - 152.138 * std::cos(2 * D - M - Mp)
                - 170.733 * std::cos(2 * D + Mp) - 204.586 * std::cos(2 * D - M);

    double l = lon * DEG, b = lat * DEG;
    double eps = (23.439291 - 0.0130042 * T) * DEG;  // obliquite

    double xe = dist * std::cos(b) * std::cos(l);
    double ye = dist * std::cos(b) * std::sin(l);
    double ze = dist * std::sin(b);
    return {xe, ye * std::cos(eps) - ze * std::sin(eps), ye * std::sin(eps) + ze * std::cos(eps)};
}

// ---------------------------------------------------------------------------
// Dynamique
// ---------------------------------------------------------------------------
Vec3 acceleration(const Vec3& r, const Vec3& rm) {
    double r2 = dot(r, r), rn = std::sqrt(r2);
    Vec3 a = r * (-MU_EARTH / (r2 * rn));

    // J2 (axe polaire = z de l'ECI)
    double k = 1.5 * J2 * MU_EARTH * R_EARTH * R_EARTH / (r2 * r2 * rn);
    double z2 = 5.0 * r.z * r.z / r2;
    a = a + Vec3{k * r.x * (z2 - 1.0), k * r.y * (z2 - 1.0), k * r.z * (z2 - 3.0)};

    // Lune : attraction directe - terme indirect (repere geocentrique non inertiel)
    Vec3 d = rm - r;
    double dn = norm(d), mn = norm(rm);
    a = a + d * (MU_MOON / (dn * dn * dn)) - rm * (MU_MOON / (mn * mn * mn));
    return a;
}

// RK4 ; la Lune est evaluee au debut, milieu et fin du pas
State rk4(const State& s, double dt, const Vec3& rm0, const Vec3& rm1, const Vec3& rm2) {
    auto f = [](const State& y, const Vec3& rm) { return State{y.v, acceleration(y.r, rm)}; };
    auto add = [](const State& y, const State& k, double h) {
        return State{y.r + k.r * h, y.v + k.v * h};
    };
    State k1 = f(s, rm0);
    State k2 = f(add(s, k1, dt / 2), rm1);
    State k3 = f(add(s, k2, dt / 2), rm1);
    State k4 = f(add(s, k3, dt), rm2);
    return {s.r + (k1.r + k2.r * 2 + k3.r * 2 + k4.r) * (dt / 6),
            s.v + (k1.v + k2.v * 2 + k3.v * 2 + k4.v) * (dt / 6)};
}

// Elements orbitaux (a en km, e, i en deg) a partir du vecteur d'etat
struct Elements { double a, e, inc, perigee_alt, apogee_alt; };
Elements elements(const State& s) {
    double r = norm(s.r), v2 = dot(s.v, s.v);
    double energy = v2 / 2 - MU_EARTH / r;
    double a = -MU_EARTH / (2 * energy);
    Vec3 h = cross(s.r, s.v);
    Vec3 ev = cross(s.v, h) * (1.0 / MU_EARTH) - s.r * (1.0 / r);
    double e = norm(ev);
    return {a, e, std::acos(h.z / norm(h)) / DEG, a * (1 - e) - R_EARTH, a * (1 + e) - R_EARTH};
}

// ---------------------------------------------------------------------------
// Lecture du fichier Space-Track
// ---------------------------------------------------------------------------
std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n"), e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

std::vector<Object> load_objects(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::cerr << "Impossible d'ouvrir " << path << "\n"; std::exit(1); }
    std::vector<Object> objs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> f;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, '|')) f.push_back(trim(tok));
        if (f.size() < 17) continue;
        try {
            Object o;
            o.id = std::stoi(f[0]);
            o.name = f[1]; o.type = f[2]; o.rcs = f[5];
            o.s.r = {std::stod(f[11]), std::stod(f[12]), std::stod(f[13])};
            o.s.v = {std::stod(f[14]), std::stod(f[15]), std::stod(f[16])};
            objs.push_back(std::move(o));
        } catch (...) { continue; }
    }
    return objs;
}

void write_snapshot(const std::string& path, const std::vector<Object>& objs, double t) {
    std::ofstream out(path);
    out << "# t = " << t << " s apres l'epoch\nid,x_km,y_km,z_km,vx,vy,vz,alive\n";
    out.precision(9);
    for (const auto& o : objs)
        out << o.id << ',' << o.s.r.x << ',' << o.s.r.y << ',' << o.s.r.z << ','
            << o.s.v.x << ',' << o.s.v.y << ',' << o.s.v.z << ',' << o.alive << '\n';
}

}  // namespace

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
    std::vector<Object> objs = load_objects(path);
    std::cout << objs.size() << " objets charges. Duree " << duration / 3600.0
              << " h, pas " << dt << " s\n";
    if (objs.empty()) return 1;

    // Reference de conservation : energie specifique (J2/Lune la font deriver un peu)
    std::vector<State> initial;
    for (const auto& o : objs) initial.push_back(o.s);

    Vec3 moon0 = moon_position(JD_EPOCH);
    std::cout << "Lune a l'epoch : distance " << norm(moon0) << " km\n";

    long steps = static_cast<long>(std::ceil(duration / dt));
    long next_out = out_every > 0 ? static_cast<long>(out_every / dt) : 0;
    long snap = 0;

    std::vector<size_t> tracked;
    std::ofstream traj;
    if (!track_ids.empty()) {
        for (size_t i = 0; i < objs.size(); ++i)
            if (std::find(track_ids.begin(), track_ids.end(), objs[i].id) != track_ids.end())
                tracked.push_back(i);
        traj.open(OUT_DIR + "/trajectories.csv");
        traj << "t_s,id,name,x_km,y_km,z_km\n";
        if (next_out == 0) next_out = std::max(1L, static_cast<long>(60.0 / dt));
    }
    auto record = [&](double t) {
        for (size_t i : tracked)
            traj << t << ',' << objs[i].id << ',' << objs[i].name << ',' << objs[i].s.r.x
                 << ',' << objs[i].s.r.y << ',' << objs[i].s.r.z << '\n';
    };
    if (!tracked.empty()) record(0.0);
    std::ofstream moon_out;
    if (!tracked.empty()) {
        moon_out.open(OUT_DIR + "/moon.csv");
        moon_out << "t_s,x_km,y_km,z_km\n";
    }

    for (long n = 0; n < steps; ++n) {
        double t = n * dt;
        Vec3 m0 = moon_position(JD_EPOCH + t / 86400.0);
        Vec3 m1 = moon_position(JD_EPOCH + (t + dt / 2) / 86400.0);
        Vec3 m2 = moon_position(JD_EPOCH + (t + dt) / 86400.0);

        #pragma omp parallel for schedule(static)
        for (long i = 0; i < static_cast<long>(objs.size()); ++i) {
            Object& o = objs[i];
            if (!o.alive) continue;
            o.s = rk4(o.s, dt, m0, m1, m2);
            if (norm(o.s.r) < R_EARTH) o.alive = false;   // rentree / impact
        }

        if (!tracked.empty()) {
            if ((n + 1) % next_out == 0) {
                record((n + 1) * dt);
                moon_out << (n + 1) * dt << ',' << m2.x << ',' << m2.y << ',' << m2.z << '\n';
            }
        } else if (next_out > 0 && (n + 1) % next_out == 0)
            write_snapshot(OUT_DIR + "/snapshot_" + std::to_string(snap++) + ".csv", objs, (n + 1) * dt);
    }

    write_snapshot(OUT_DIR + "/final_state.csv", objs, steps * dt);

    // Bilan : objets perdus et derive de l'energie/de l'altitude
    long lost = 0;
    double max_da = 0;
    int max_id = 0;
    for (size_t i = 0; i < objs.size(); ++i) {
        if (!objs[i].alive) { ++lost; continue; }
        double da = std::fabs(elements(objs[i].s).a - elements(initial[i]).a);
        if (da > max_da) { max_da = da; max_id = objs[i].id; }
    }
    std::cout << "Objets sous la surface terrestre : " << lost << "\n"
              << "Plus forte variation de demi-grand axe : " << max_da << " km (NORAD "
              << max_id << ")\n"
              << "Etat final ecrit dans " << OUT_DIR << "/final_state.csv\n";
    return 0;
}
