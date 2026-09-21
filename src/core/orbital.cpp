#include "orbital.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

namespace kessler {

namespace {

std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n"), e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

// Date civile UTC -> jour julien (algorithme de Meeus, chap. 7).
double to_jd(int y, int m, double day) {
    if (m <= 2) { y -= 1; m += 12; }
    int A = y / 100;
    int B = 2 - A + A / 4;
    return std::floor(365.25 * (y + 4716)) + std::floor(30.6001 * (m + 1))
           + day + B - 1524.5;
}

}  // namespace

// ---------------------------------------------------------------------------
// Categories d'objets
// ---------------------------------------------------------------------------

ObjectType parse_object_type(const std::string& s) {
    if (s == "PAYLOAD") return ObjectType::Payload;
    if (s == "DEBRIS") return ObjectType::Debris;
    if (s == "ROCKET BODY") return ObjectType::RocketBody;
    return ObjectType::Unknown;
}

const char* object_type_name(ObjectType t) {
    switch (t) {
        case ObjectType::Payload: return "PAYLOAD";
        case ObjectType::Debris: return "DEBRIS";
        case ObjectType::RocketBody: return "ROCKET BODY";
        default: return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// Ephemeride lunaire
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

    // Lune : attraction directe moins terme indirect
    Vec3 d = rm - r;
    double dn = norm(d), mn = norm(rm);
    a = a + d * (MU_MOON / (dn * dn * dn)) - rm * (MU_MOON / (mn * mn * mn));
    return a;
}

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

Elements elements(const State& s) {
    double r = norm(s.r), v2 = dot(s.v, s.v);
    double energy = v2 / 2 - MU_EARTH / r;
    double a = -MU_EARTH / (2 * energy);
    Vec3 h = cross(s.r, s.v);
    Vec3 ev = cross(s.v, h) * (1.0 / MU_EARTH) - s.r * (1.0 / r);
    double e = norm(ev);
    double period = a > 0 ? 2 * PI * std::sqrt(a * a * a / MU_EARTH) / 60.0 : 0.0;
    return {a, e, std::acos(h.z / norm(h)) / DEG,
            a * (1 - e) - R_EARTH, a * (1 + e) - R_EARTH, period};
}

// ---------------------------------------------------------------------------
// Entrees / sorties
// ---------------------------------------------------------------------------

std::vector<Object> load_catalog(const std::string& path, int* dropped) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Impossible d'ouvrir " << path << "\n";
        return {};
    }
    std::vector<Object> objs;
    std::string line;
    int bad = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> f;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, '|')) f.push_back(trim(tok));
        if (f.size() < 17) { ++bad; continue; }
        try {
            Object o;
            o.id = std::stoi(f[0]);
            o.name = f[1];
            o.type = parse_object_type(f[2]);
            o.country = f[4];
            o.rcs = f[5];
            o.s.r = {std::stod(f[11]), std::stod(f[12]), std::stod(f[13])};
            o.s.v = {std::stod(f[14]), std::stod(f[15]), std::stod(f[16])};
            objs.push_back(std::move(o));
        } catch (...) { ++bad; }
    }
    if (dropped) *dropped = bad;
    return objs;
}

double read_epoch_jd(const std::string& path) {
    std::ifstream in(path);
    if (!in) return JD_DEFAULT_EPOCH;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] != '#') break;  // l'en-tete est fini
        size_t pos = line.find("epoch commun) :");
        if (pos == std::string::npos) continue;
        // Format attendu : 2026-08-01T10:00:00+00:00
        int y, mo, d, h, mi, s;
        if (std::sscanf(line.c_str() + pos, "epoch commun) : %d-%d-%dT%d:%d:%d",
                        &y, &mo, &d, &h, &mi, &s) == 6) {
            return to_jd(y, mo, d + (h + mi / 60.0 + s / 3600.0) / 24.0);
        }
    }
    return JD_DEFAULT_EPOCH;
}

void write_snapshot(const std::string& path, const std::vector<Object>& objs, double t) {
    std::ofstream out(path);
    out << "# t = " << t << " s apres l'epoch\nid,x_km,y_km,z_km,vx,vy,vz,alive\n";
    out.precision(9);
    for (const auto& o : objs)
        out << o.id << ',' << o.s.r.x << ',' << o.s.r.y << ',' << o.s.r.z << ','
            << o.s.v.x << ',' << o.s.v.y << ',' << o.s.v.z << ',' << o.alive << '\n';
}

bool read_snapshot(const std::string& path, std::vector<Object>& objs, double* t) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    size_t i = 0;
    bool header_done = false;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            size_t pos = line.find("t =");
            if (t && pos != std::string::npos) *t = std::atof(line.c_str() + pos + 3);
            continue;
        }
        if (!header_done) { header_done = true; continue; }  // ligne d'en-tete des colonnes
        if (i >= objs.size()) return false;                  // snapshot plus long que le catalogue

        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> f;
        while (std::getline(ss, tok, ',')) f.push_back(tok);
        if (f.size() < 8) return false;
        try {
            if (std::stoi(f[0]) != objs[i].id) return false;  // ordre different du catalogue
            objs[i].s.r = {std::stod(f[1]), std::stod(f[2]), std::stod(f[3])};
            objs[i].s.v = {std::stod(f[4]), std::stod(f[5]), std::stod(f[6])};
            objs[i].alive = std::stoi(f[7]) != 0;
        } catch (...) { return false; }
        ++i;
    }
    return i == objs.size();
}

std::string jd_to_utc_string(double jd) {
    // Meeus, chap. 7 : jour julien -> date civile.
    double z = std::floor(jd + 0.5);
    double frac = jd + 0.5 - z;
    double a = z;
    if (z >= 2299161) {
        double alpha = std::floor((z - 1867216.25) / 36524.25);
        a = z + 1 + alpha - std::floor(alpha / 4);
    }
    double b = a + 1524;
    double c = std::floor((b - 122.1) / 365.25);
    double d = std::floor(365.25 * c);
    double e = std::floor((b - d) / 30.6001);

    int day = static_cast<int>(b - d - std::floor(30.6001 * e));
    int month = static_cast<int>(e < 14 ? e - 1 : e - 13);
    int year = static_cast<int>(month > 2 ? c - 4716 : c - 4715);

    double hours = frac * 24.0;
    int hh = static_cast<int>(hours);
    int mm = static_cast<int>((hours - hh) * 60.0);
    int sec = static_cast<int>(((hours - hh) * 60.0 - mm) * 60.0 + 0.5);
    if (sec >= 60) { sec -= 60; ++mm; }
    if (mm >= 60) { mm -= 60; ++hh; }

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d UTC",
                  year, month, day, hh, mm, sec);
    return buf;
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

void Simulation::step() {
    Vec3 m0 = moon_position(epoch_jd + t / 86400.0);
    Vec3 m1 = moon_position(epoch_jd + (t + dt / 2) / 86400.0);
    Vec3 m2 = moon_position(epoch_jd + (t + dt) / 86400.0);

    const long n = static_cast<long>(objects.size());
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        Object& o = objects[i];
        if (!o.alive) continue;
        o.s = rk4(o.s, dt, m0, m1, m2);
        if (norm(o.s.r) < R_EARTH) o.alive = false;   // rentree / impact
    }
    t += dt;
}

}  // namespace kessler
