// Coeur orbital partage entre la simulation en ligne de commande (src/main.cpp)
// et le viewer temps reel (src/viewer/viewer.cpp).
//
// Tout est en kilometres et en secondes, dans le repere inertiel geocentrique
// (ECI) du fichier d'entree.

#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace kessler {

// ---------------------------------------------------------------------------
// Constantes
// ---------------------------------------------------------------------------

inline constexpr double PI = 3.14159265358979323846;
inline constexpr double DEG = PI / 180.0;

inline constexpr double MU_EARTH = 398600.4418;   // km^3/s^2
inline constexpr double MU_MOON = 4902.800066;    // km^3/s^2
inline constexpr double R_EARTH = 6378.137;       // km (equatorial)
inline constexpr double R_MOON = 1737.4;          // km
inline constexpr double J2 = 1.08262668e-3;

// Epoch par defaut (2026-08-01 10:00:00 UTC), utilise si l'en-tete du fichier
// de catalogue ne porte pas de date lisible.
inline constexpr double JD_DEFAULT_EPOCH = 2461253.916666667;

inline const std::string OUT_DIR = "output";

// ---------------------------------------------------------------------------
// Algebre vectorielle
// ---------------------------------------------------------------------------

struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
};

inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

struct State { Vec3 r, v; };

// ---------------------------------------------------------------------------
// Objets du catalogue
// ---------------------------------------------------------------------------

// L'ordre sert d'indice pour les couleurs et les filtres du viewer.
enum class ObjectType { Payload = 0, Debris, RocketBody, Unknown, Count };

ObjectType parse_object_type(const std::string& s);
const char* object_type_name(ObjectType t);

struct Object {
    int id = 0;
    std::string name, country, rcs;
    ObjectType type = ObjectType::Unknown;
    State s;
    bool alive = true;   // false une fois passe sous la surface terrestre
};

// ---------------------------------------------------------------------------
// Dynamique
// ---------------------------------------------------------------------------

// Ephemeride lunaire analytique (Meeus, chap. 47, termes principaux),
// exprimee dans le repere equatorial ECI.
Vec3 moon_position(double jd);

// Gravite terrestre ponctuelle + J2 + perturbation lunaire (terme direct et
// terme indirect du repere geocentrique non inertiel).
Vec3 acceleration(const Vec3& r, const Vec3& rm);

// Un pas de Runge-Kutta 4 ; la Lune est evaluee au debut, au milieu et a la
// fin du pas.
State rk4(const State& s, double dt, const Vec3& rm0, const Vec3& rm1, const Vec3& rm2);

struct Elements {
    double a = 0;            // demi-grand axe (km)
    double e = 0;            // excentricite
    double inc = 0;          // inclinaison (deg)
    double perigee_alt = 0;  // altitude au perigee (km)
    double apogee_alt = 0;   // altitude a l'apogee (km)
    double period_min = 0;   // periode orbitale (min)
};

Elements elements(const State& s);

// ---------------------------------------------------------------------------
// Entrees / sorties
// ---------------------------------------------------------------------------

// Lit l'export Space-Track. `dropped`, s'il est fourni, recoit le nombre de
// lignes rejetees (format inattendu) : ce compte doit rester nul.
std::vector<Object> load_catalog(const std::string& path, int* dropped = nullptr);

// Date de reference lue dans l'en-tete du catalogue, en jour julien.
// Retourne JD_DEFAULT_EPOCH si l'en-tete ne porte pas de date exploitable.
double read_epoch_jd(const std::string& path);

void write_snapshot(const std::string& path, const std::vector<Object>& objs, double t);

// Relit un snapshot dans `objs` (meme ordre que l'ecriture). Renvoie false si
// le fichier est illisible ou ne correspond pas au catalogue charge.
// `t`, s'il est fourni, recoit l'instant du snapshot en secondes.
bool read_snapshot(const std::string& path, std::vector<Object>& objs, double* t = nullptr);

// Jour julien -> date civile UTC, pour l'affichage.
std::string jd_to_utc_string(double jd);

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

// Etat d'une propagation en cours. Partage par la CLI et le viewer pour que
// les deux integrent rigoureusement la meme dynamique.
struct Simulation {
    std::vector<Object> objects;
    double epoch_jd = JD_DEFAULT_EPOCH;
    double t = 0.0;    // secondes ecoulees depuis l'epoch
    double dt = 10.0;  // pas d'integration (s)

    // Avance d'un pas. Parallelise par OpenMP quand il est disponible.
    void step();

    // Avance de `n` pas.
    void advance(long n) { for (long i = 0; i < n; ++i) step(); }

    double jd() const { return epoch_jd + t / 86400.0; }
    Vec3 moon() const { return moon_position(jd()); }
};

}  // namespace kessler
