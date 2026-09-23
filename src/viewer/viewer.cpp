// Kessler_Sim - viewer 3D temps reel
//
// Affiche l'integralite du catalogue (~28 000 objets) et laisse filtrer,
// selectionner et suivre des objets en particulier.
//
// Deux modes :
//   - Live    : le viewer propage lui-meme, avec le RK4 de src/core (meme
//               dynamique que la simulation en ligne de commande).
//   - Relecture : rejoue les snapshot_*.csv produits par kessler_sim, donc
//               exactement ce que la simulation a calcule.
//
// Rendu : un seul appel instancie par categorie d'objet (4 au total), et le
// tampon de profondeur du GPU se charge de l'occultation par la Terre — le
// probleme que matplotlib ne savait pas resoudre.
//
// Utilisation, DEPUIS LA RACINE DU DEPOT :
//   kessler_viewer [fichier_catalogue]
//   kessler_viewer --screenshot vue.png [--frames N]   (rendu hors interaction)
//
// Etat de depart, pratique pour scripter : --play (lance la propagation),
// --detect (active aussi la detection), --grid (affiche la grille),
// --radius-scale S.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <cstring>
#include <deque>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

#include "raylib.h"
#include "raymath.h"

#include "imgui.h"
#include "rlImGui.h"

// raylib.h definit PI comme macro, ce qui rendrait kessler::PI inutilisable.
// Les en-tetes raylib sont deja analyses ici, y compris les fonctions inline
// de raymath : liberer le nom maintenant ne leur retire rien.
#undef PI

#include "core/collision.hpp"
#include "core/orbital.hpp"
#include "core/threads.hpp"

using namespace kessler;

namespace {

// 1 unite de rendu = 1000 km. Garde la scene dans une plage ou la precision
// des flottants et les plans de coupe par defaut de raylib restent corrects :
// Terre 6.4, GEO 42, Lune 385.
constexpr float SCALE = 0.001f;

constexpr int TYPE_COUNT = static_cast<int>(ObjectType::Count);

const Color TYPE_COLOR[TYPE_COUNT] = {
    {80, 160, 255, 255},   // PAYLOAD
    {235, 80, 70, 255},    // DEBRIS
    {255, 165, 50, 255},   // ROCKET BODY
    {170, 170, 170, 255},  // UNKNOWN
};

Vector3 to_render(const Vec3& v) {
    // ECI (x, y, z) -> raylib (x, z, y) : raylib a Y vers le haut, on met donc
    // l'axe polaire terrestre sur Y.
    return {static_cast<float>(v.x * SCALE),
            static_cast<float>(v.z * SCALE),
            static_cast<float>(v.y * SCALE)};
}

bool contains_ci(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return true;
    auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
                          [](char a, char b) {
                              return std::tolower(static_cast<unsigned char>(a)) ==
                                     std::tolower(static_cast<unsigned char>(b));
                          });
    return it != haystack.end();
}

// ---------------------------------------------------------------------------
// Camera orbitale
// ---------------------------------------------------------------------------

struct OrbitCamera {
    float distance = 30.0f;    // en unites de rendu
    float azimuth = 0.6f;      // rad
    float elevation = 0.4f;    // rad
    Vector3 target = {0, 0, 0};

    Camera3D to_camera3d() const {
        Camera3D cam = {};
        float ce = std::cos(elevation);
        cam.position = {target.x + distance * ce * std::cos(azimuth),
                        target.y + distance * std::sin(elevation),
                        target.z + distance * ce * std::sin(azimuth)};
        cam.target = target;
        cam.up = {0, 1, 0};
        cam.fovy = 45.0f;
        cam.projection = CAMERA_PERSPECTIVE;
        return cam;
    }

    void handle_input() {
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            Vector2 d = GetMouseDelta();
            // Convention "on attrape le globe" : glisser vers la droite fait
            // tourner la Terre vers la droite, donc la camera part a gauche,
            // ce qui correspond a un azimut croissant.
            azimuth += d.x * 0.005f;
            elevation = Clamp(elevation + d.y * 0.005f, -1.55f, 1.55f);
        }
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) distance = Clamp(distance * std::pow(0.88f, wheel), 7.0f, 900.0f);
    }
};

// ---------------------------------------------------------------------------
// Filtres
// ---------------------------------------------------------------------------

struct Filters {
    bool type_on[TYPE_COUNT] = {true, true, true, true};
    float alt_min = 0.0f;
    // Assez haut pour englober les rares objets en orbite lunaire (DRO-A est
    // a ~529 000 km) : par defaut, tout le catalogue doit etre affiche.
    float alt_max = 1000000.0f;
    float inc_min = 0.0f;
    float inc_max = 180.0f;
    char search[64] = "";
    bool tracked_only = false;
};

// ---------------------------------------------------------------------------
// Rendu instancie
// ---------------------------------------------------------------------------

// Shader minimal d'instanciation : raylib fournit la matrice de chaque
// instance via l'attribut `instanceTransform`, et la couleur via colDiffuse.
// On tient donc tout le catalogue en 4 appels de dessin.
const char* VS_INSTANCED = R"(#version 330
in vec3 vertexPosition;
in mat4 instanceTransform;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * instanceTransform * vec4(vertexPosition, 1.0);
}
)";

const char* FS_INSTANCED = R"(#version 330
uniform vec4 colDiffuse;
out vec4 finalColor;
void main() { finalColor = colDiffuse; }
)";

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

struct App {
    Simulation sim;
    std::vector<State> initial;     // pour la remise a zero
    std::string catalog_path;

    // Grandeurs recalculees a chaque image (filtrage et affichage).
    std::vector<float> alt, inc;
    std::vector<char> visible;

    Filters filters;
    std::vector<int> tracked;
    int selected = -1;

    // Mode relecture
    bool replay_mode = false;
    std::vector<std::string> snapshots;
    int snapshot_index = 0;

    // Controle du temps
    bool playing = false;
    int steps_per_frame = 6;

    // Taille des points, proportionnelle a la distance camera pour garder une
    // taille apparente constante. Volontairement petite : a 28 000 objets, des
    // points trop gros forment une coquille opaque qui masque la Terre.
    float point_scale = 0.0008f;

    OrbitCamera cam;
    bool follow_selected = false;

    int visible_count = 0;
    int type_visible[TYPE_COUNT] = {};

    // --- Detection des rapprochements (mode Live seulement) ---
    bool detect = false;
    bool show_grid = false;          // trace les cellules de la grille de detection
    ScreeningConfig screen_cfg;
    std::vector<State> before;       // etats en debut de pas, pour screen_step
    ScreeningStats last_stats;
    double detect_ms = 0.0;          // moyenne glissante, par pas
    long long total_conj = 0, total_coll = 0;

    // Un rapprochement signale dans la vue : les objets s'eloignent ensuite,
    // donc on memorise leurs positions a l'instant du signalement.
    struct Event {
        int i = 0, j = 0;
        double t = 0, miss_km = 0, v_rel = 0;
        bool collision = false;
        Vec3 ri, rj;
        float age = 0.0f;            // secondes ecoulees a l'ecran
    };
    std::deque<Event> events;        // le plus recent en tete
    float event_lifetime = 4.0f;

    void run_detection(double t0) {
        auto c0 = std::chrono::steady_clock::now();
        std::vector<Conjunction> found =
            screen_step(before, sim.objects, t0, sim.dt, screen_cfg, &last_stats);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - c0).count();
        detect_ms = detect_ms > 0 ? 0.9 * detect_ms + 0.1 * ms : ms;

        for (const Conjunction& c : found) {
            Event e;
            e.i = c.i; e.j = c.j; e.t = c.t;
            e.miss_km = c.miss_km; e.v_rel = c.v_rel_km_s;
            e.collision = c.collision;
            e.ri = sim.objects[c.i].s.r;
            e.rj = sim.objects[c.j].s.r;
            events.push_front(e);
            ++total_conj;
            if (c.collision) ++total_coll;
        }
        while (events.size() > 200) events.pop_back();
    }

    void reset() {
        for (size_t i = 0; i < sim.objects.size(); ++i) {
            sim.objects[i].s = initial[i];
            sim.objects[i].alive = true;
        }
        sim.t = 0.0;
    }

    void scan_snapshots() {
        snapshots.clear();
        std::error_code ec;
        if (!std::filesystem::exists(OUT_DIR, ec)) return;
        for (const auto& e : std::filesystem::directory_iterator(OUT_DIR, ec)) {
            std::string n = e.path().filename().string();
            if (n.rfind("snapshot_", 0) == 0 && e.path().extension() == ".csv")
                snapshots.push_back(e.path().string());
        }
        // snapshot_2 doit venir avant snapshot_10 : tri sur l'indice numerique.
        std::sort(snapshots.begin(), snapshots.end(), [](const std::string& a, const std::string& b) {
            auto num = [](const std::string& s) {
                size_t p = s.find_last_of('_');
                return p == std::string::npos ? 0L : std::atol(s.c_str() + p + 1);
            };
            return num(a) < num(b);
        });
    }

    void load_snapshot_at(int index) {
        if (index < 0 || index >= static_cast<int>(snapshots.size())) return;
        double t = sim.t;
        if (read_snapshot(snapshots[index], sim.objects, &t)) {
            sim.t = t;
            snapshot_index = index;
        }
    }

    // Recalcule altitude, inclinaison et visibilite. Le gros du travail par
    // image, donc parallelise.
    void refresh_derived() {
        const long n = static_cast<long>(sim.objects.size());
        std::string needle(filters.search);

        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; ++i) {
            const Object& o = sim.objects[i];
            double r = norm(o.s.r);
            alt[i] = static_cast<float>(r - R_EARTH);
            Vec3 h = cross(o.s.r, o.s.v);
            double hn = norm(h);
            inc[i] = hn > 0 ? static_cast<float>(std::acos(h.z / hn) / DEG) : 0.0f;
        }

        visible_count = 0;
        for (int t = 0; t < TYPE_COUNT; ++t) type_visible[t] = 0;

        for (long i = 0; i < n; ++i) {
            const Object& o = sim.objects[i];
            int ti = static_cast<int>(o.type);
            bool ok = o.alive && filters.type_on[ti]
                      && alt[i] >= filters.alt_min && alt[i] <= filters.alt_max
                      && inc[i] >= filters.inc_min && inc[i] <= filters.inc_max;
            if (ok && !needle.empty())
                ok = contains_ci(o.name, needle) || contains_ci(std::to_string(o.id), needle);
            if (ok && filters.tracked_only)
                ok = std::find(tracked.begin(), tracked.end(), static_cast<int>(i)) != tracked.end();
            visible[i] = ok ? 1 : 0;
            if (ok) { ++visible_count; ++type_visible[ti]; }
        }
    }

    bool is_tracked(int i) const {
        return std::find(tracked.begin(), tracked.end(), i) != tracked.end();
    }

    void toggle_tracked(int i) {
        auto it = std::find(tracked.begin(), tracked.end(), i);
        if (it == tracked.end()) tracked.push_back(i);
        else tracked.erase(it);
    }
};

// Une orbite complete propagee a partir de l'etat courant, sur une copie :
// la simulation n'est pas affectee.
struct OrbitSample {
    std::vector<Vector3> points;   // pour le trace
    double r_min = 0, r_max = 0;   // rayons extremes effectivement parcourus (km)
    bool valid = false;
};

OrbitSample propagate_orbit(const App& app, const Object& o, int samples) {
    OrbitSample out;
    Elements el = elements(o.s);
    if (el.a <= 0 || el.period_min <= 0 || el.period_min > 60 * 24 * 40) return out;

    double dt = el.period_min * 60.0 / samples;
    State s = o.s;
    double jd = app.sim.jd();
    out.points.reserve(samples + 1);
    out.r_min = out.r_max = norm(s.r);
    for (int k = 0; k <= samples; ++k) {
        out.points.push_back(to_render(s.r));
        double r = norm(s.r);
        out.r_min = std::min(out.r_min, r);
        out.r_max = std::max(out.r_max, r);
        Vec3 m0 = moon_position(jd);
        Vec3 m1 = moon_position(jd + dt / 2 / 86400.0);
        Vec3 m2 = moon_position(jd + dt / 86400.0);
        s = rk4(s, dt, m0, m1, m2);
        jd += dt / 86400.0;
    }
    out.valid = true;
    return out;
}

// ---------------------------------------------------------------------------
// Panneaux ImGui
// ---------------------------------------------------------------------------

// Les trois panneaux sont empiles a gauche au premier lancement ; ImGui
// memorise ensuite ce que l'utilisateur en fait dans son imgui.ini.
void place_panel(float y, float height) {
    ImGui::SetNextWindowPos({10, y}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({360, height}, ImGuiCond_FirstUseEver);
}

void draw_simulation_panel(App& app) {
    place_panel(10, 300);
    ImGui::Begin("Simulation");
    ImGui::PushItemWidth(150);

    int mode = app.replay_mode ? 1 : 0;
    if (ImGui::RadioButton("Live (RK4)", &mode, 0)) app.replay_mode = false;
    ImGui::SameLine();
    if (ImGui::RadioButton("Relecture", &mode, 1)) {
        app.replay_mode = true;
        app.scan_snapshots();
        if (!app.snapshots.empty()) app.load_snapshot_at(app.snapshot_index);
    }

    ImGui::Separator();
    ImGui::Text("%s", jd_to_utc_string(app.sim.jd()).c_str());
    ImGui::Text("t = %.2f h apres l'epoch", app.sim.t / 3600.0);
    ImGui::Text("%d / %d objets affiches", app.visible_count,
                static_cast<int>(app.sim.objects.size()));
    ImGui::Text("%.0f fps", static_cast<double>(GetFPS()));

    ImGui::Separator();
    if (app.replay_mode) {
        if (app.snapshots.empty()) {
            ImGui::TextWrapped("Aucun snapshot dans output/. Lance par exemple :");
            ImGui::TextWrapped("kessler_sim data/... 24 10 600");
            if (ImGui::Button("Rechercher")) app.scan_snapshots();
        } else {
            int idx = app.snapshot_index;
            if (ImGui::SliderInt("Image", &idx, 0, static_cast<int>(app.snapshots.size()) - 1))
                app.load_snapshot_at(idx);
            ImGui::Checkbox("Lecture", &app.playing);
            ImGui::SameLine();
            if (ImGui::Button("Rechercher")) app.scan_snapshots();
        }
    } else {
        if (ImGui::Button(app.playing ? "Pause" : "Lecture")) app.playing = !app.playing;
        ImGui::SameLine();
        if (ImGui::Button("Retour a l'epoch")) { app.reset(); app.playing = false; }

        ImGui::SliderInt("Pas par image", &app.steps_per_frame, 1, 60);
        float dt = static_cast<float>(app.sim.dt);
        // Echelle logarithmique : sur 0.1 - 120 s, un curseur lineaire rendrait
        // les petits pas quasiment impossibles a regler.
        if (ImGui::SliderFloat("Pas dt (s)", &dt, 0.1f, 120.0f, "%.1f",
                               ImGuiSliderFlags_Logarithmic))
            app.sim.dt = dt;
        ImGui::Text("Vitesse : %.0f x temps reel",
                    app.sim.dt * app.steps_per_frame * (GetFPS() > 0 ? GetFPS() : 60));
    }

    ImGui::Separator();
    ImGui::SliderFloat("Taille des points", &app.point_scale, 0.0002f, 0.004f, "%.4f");

    ImGui::PopItemWidth();
    ImGui::End();
}

void draw_filters_panel(App& app) {
    place_panel(320, 270);
    ImGui::Begin("Filtres");
    ImGui::PushItemWidth(150);
    Filters& f = app.filters;

    for (int t = 0; t < TYPE_COUNT; ++t) {
        Color c = TYPE_COLOR[t];
        ImGui::PushStyleColor(ImGuiCol_Text,
                              IM_COL32(c.r, c.g, c.b, 255));
        ImGui::Checkbox(object_type_name(static_cast<ObjectType>(t)), &f.type_on[t]);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextDisabled("(%d)", app.type_visible[t]);
    }

    ImGui::Separator();
    ImGui::DragFloatRange2("Altitude (km)", &f.alt_min, &f.alt_max, 50.0f,
                           0.0f, 1000000.0f, "%.0f", "%.0f");
    ImGui::DragFloatRange2("Inclinaison", &f.inc_min, &f.inc_max, 0.5f,
                           0.0f, 180.0f, "%.1f", "%.1f");
    ImGui::InputText("Nom / NORAD", f.search, sizeof(f.search));
    ImGui::Checkbox("Objets suivis seulement", &f.tracked_only);

    ImGui::Separator();
    if (ImGui::Button("LEO")) { f.alt_min = 0; f.alt_max = 2000; }
    ImGui::SameLine();
    if (ImGui::Button("MEO")) { f.alt_min = 2000; f.alt_max = 35000; }
    ImGui::SameLine();
    if (ImGui::Button("GEO")) { f.alt_min = 35000; f.alt_max = 36500; }
    ImGui::SameLine();
    if (ImGui::Button("Tout")) {
        f = Filters{};
    }

    ImGui::PopItemWidth();
    ImGui::End();
}

// Distance lisible : metres en dessous du kilometre.
std::string format_distance(double km) {
    char buf[32];
    if (km < 1.0) std::snprintf(buf, sizeof(buf), "%.0f m", km * 1000.0);
    else std::snprintf(buf, sizeof(buf), "%.2f km", km);
    return buf;
}

void draw_detection_panel(App& app) {
    ImGui::SetNextWindowPos({GetScreenWidth() - 380.0f, 10}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({370, 520}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Rapprochements");

    if (app.replay_mode) {
        ImGui::TextWrapped("La detection a besoin des etats en debut et en fin de pas : "
                           "elle n'est disponible qu'en mode Live.");
        ImGui::End();
        return;
    }

    ImGui::Checkbox("Detection active", &app.detect);
    ImGui::SameLine();
    ImGui::TextDisabled("(%.1f ms/pas)", app.detect_ms);
    if (app.detect && app.steps_per_frame > 3)
        ImGui::TextColored({1.0f, 0.75f, 0.3f, 1.0f},
                           "%d pas par image : ~%.0f ms d'analyse par image.",
                           app.steps_per_frame, app.detect_ms * app.steps_per_frame);

    ImGui::PushItemWidth(140);
    float seuil = static_cast<float>(app.screen_cfg.threshold_km);
    if (ImGui::SliderFloat("Seuil (km)", &seuil, 0.1f, 50.0f, "%.1f",
                           ImGuiSliderFlags_Logarithmic))
        app.screen_cfg.threshold_km = seuil;

    float scale = static_cast<float>(app.screen_cfg.radius_scale);
    if (ImGui::SliderFloat("Facteur de rayon", &scale, 1.0f, 10000.0f, "x%.0f",
                           ImGuiSliderFlags_Logarithmic))
        app.screen_cfg.radius_scale = scale;
    ImGui::PopItemWidth();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Multiplie les rayons de collision (0.1 / 0.4 / 2 m selon\n"
                          "la taille radar). Le taux de collision croit comme le\n"
                          "carre du facteur : x100 donne ~15 collisions par jour.");

    ImGui::Checkbox("Debug : grille de detection", &app.show_grid);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Trace les cellules de la grille. Avec un objet selectionne,\n"
                          "montre sa cellule et les 26 voisines reellement examinees.");

    ImGui::Separator();
    ImGui::Text("Total : %lld rapprochements, %lld collisions",
                app.total_conj, app.total_coll);
    ImGui::Text("Cellule %.0f km, %lld paires candidates/pas",
                app.last_stats.cell_km, app.last_stats.candidate_pairs);
    ImGui::TextDisabled("%lld paires volant de concert, ecartees",
                        app.last_stats.co_orbiting);

    ImGui::Separator();
    if (ImGui::Button("Effacer")) {
        app.events.clear();
        app.total_conj = app.total_coll = 0;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("clic : selectionner et centrer");

    ImGui::BeginChild("liste");
    for (size_t k = 0; k < app.events.size(); ++k) {
        const App::Event& e = app.events[k];
        ImGui::PushID(static_cast<int>(k));
        if (e.collision) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 90, 90, 255));
        bool clicked = ImGui::Selectable(
            (format_distance(e.miss_km) + "  " + app.sim.objects[e.i].name +
             "  /  " + app.sim.objects[e.j].name).c_str());
        if (e.collision) ImGui::PopStyleColor();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("t = %.3f h\nvitesse relative %.2f km/s\n%s",
                              e.t / 3600.0, e.v_rel,
                              e.collision ? "COLLISION" : "passage");
        if (clicked) {
            app.selected = e.i;
            app.follow_selected = true;
        }
        ImGui::PopID();
    }
    ImGui::EndChild();
    ImGui::End();
}

// Trace la grille de detection : la cellule de l'objet selectionne et ses 26
// voisines — exactement le voisinage examine — ou, a defaut de selection, les
// cellules occupees autour du point vise.
void draw_grid_debug(const App& app) {
    double cell = app.last_stats.cell_km;
    if (cell <= 0) return;
    float side = static_cast<float>(cell * SCALE);

    auto cube = [&](int64_t ix, int64_t iy, int64_t iz, Color col) {
        Vec3 c{(ix + 0.5) * cell, (iy + 0.5) * cell, (iz + 0.5) * cell};
        DrawCubeWires(to_render(c), side, side, side, col);
    };
    auto cell_of = [&](const Vec3& r) {
        return std::array<int64_t, 3>{{static_cast<int64_t>(std::floor(r.x / cell)),
                                       static_cast<int64_t>(std::floor(r.y / cell)),
                                       static_cast<int64_t>(std::floor(r.z / cell))}};
    };

    if (app.selected >= 0) {
        auto c = cell_of(app.sim.objects[app.selected].s.r);
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz)
                    cube(c[0] + dx, c[1] + dy, c[2] + dz,
                         (dx || dy || dz) ? Color{70, 130, 180, 90} : Color{120, 230, 160, 220});
        return;
    }

    // Sans selection : les cellules occupees les plus proches de la camera,
    // plafonnees pour ne pas noyer la vue. Centrer sur le point vise ne
    // montrerait rien, celui-ci etant au centre de la Terre par defaut.
    Camera3D c3d = app.cam.to_camera3d();
    Vec3 eye{c3d.position.x / SCALE, c3d.position.z / SCALE, c3d.position.y / SCALE};

    std::unordered_set<uint64_t> seen;
    std::vector<std::pair<double, std::array<int64_t, 3>>> occupied;
    for (size_t i = 0; i < app.sim.objects.size(); ++i) {
        if (!app.visible[i]) continue;
        auto c = cell_of(app.sim.objects[i].s.r);
        uint64_t key = (uint64_t(c[0] + 1048576) << 42) | (uint64_t(c[1] + 1048576) << 21)
                     | uint64_t(c[2] + 1048576);
        if (!seen.insert(key).second) continue;
        Vec3 centre{(c[0] + 0.5) * cell, (c[1] + 0.5) * cell, (c[2] + 0.5) * cell};
        occupied.push_back({norm(centre - eye), c});
    }

    const size_t cap = 300;
    if (occupied.size() > cap)
        std::nth_element(occupied.begin(), occupied.begin() + cap, occupied.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
    size_t count = std::min(cap, occupied.size());
    for (size_t k = 0; k < count; ++k)
        cube(occupied[k].second[0], occupied[k].second[1], occupied[k].second[2],
             {70, 130, 180, 80});
}

void draw_selection_panel(App& app) {
    place_panel(600, 290);
    ImGui::Begin("Selection");

    if (app.selected >= 0 && app.selected < static_cast<int>(app.sim.objects.size())) {
        const Object& o = app.sim.objects[app.selected];
        Elements el = elements(o.s);
        ImGui::TextWrapped("%s", o.name.c_str());
        ImGui::Text("NORAD %d  -  %s", o.id, object_type_name(o.type));
        if (!o.country.empty()) ImGui::Text("Pays : %s", o.country.c_str());
        if (!o.rcs.empty()) ImGui::Text("Taille radar : %s", o.rcs.c_str());
        ImGui::Separator();
        ImGui::Text("Altitude    %.1f km", norm(o.s.r) - R_EARTH);
        ImGui::Text("Vitesse     %.3f km/s", norm(o.s.v));
        ImGui::Text("Inclinaison %.2f deg", el.inc);

        // Perigee et apogee effectivement parcourus, mesures sur une orbite
        // propagee avec la dynamique complete : stables d'une image a l'autre.
        OrbitSample orb = propagate_orbit(app, o, 720);
        if (orb.valid) {
            ImGui::SeparatorText("Sur une orbite");
            ImGui::Text("Perigee     %.1f km", orb.r_min - R_EARTH);
            ImGui::Text("Apogee      %.1f km", orb.r_max - R_EARTH);
            ImGui::Text("Excentricite %.5f", (orb.r_max - orb.r_min) / (orb.r_max + orb.r_min));
        }

        // Elements de l'ellipse keplerienne tangente a l'instant present.
        // J2 deforme la trajectoire en continu : ils oscillent au cours de
        // l'orbite (~20 km pour l'ISS), quel que soit le pas d'integration.
        ImGui::SeparatorText("Osculateurs (instantanes)");
        ImGui::TextDisabled("Perigee     %.1f km", el.perigee_alt);
        ImGui::TextDisabled("Apogee      %.1f km", el.apogee_alt);
        ImGui::TextDisabled("Excentricite %.5f", el.e);
        ImGui::TextDisabled("Periode     %.1f min", el.period_min);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Ellipse keplerienne tangente a l'instant present.\n"
                              "L'aplatissement terrestre (J2) la deforme en continu :\n"
                              "ces valeurs oscillent au fil de l'orbite, c'est normal.");
        ImGui::Separator();
        bool tracked = app.is_tracked(app.selected);
        if (ImGui::Button(tracked ? "Ne plus suivre" : "Suivre")) app.toggle_tracked(app.selected);
        ImGui::SameLine();
        ImGui::Checkbox("Camera liee", &app.follow_selected);
    } else {
        ImGui::TextWrapped("Clic droit sur un objet pour le selectionner.");
    }

    ImGui::Separator();
    ImGui::Text("Objets suivis (%d)", static_cast<int>(app.tracked.size()));
    int to_remove = -1;
    for (size_t k = 0; k < app.tracked.size(); ++k) {
        int i = app.tracked[k];
        ImGui::PushID(i);
        if (ImGui::SmallButton("x")) to_remove = i;
        ImGui::SameLine();
        if (ImGui::Selectable(app.sim.objects[i].name.c_str(), app.selected == i))
            app.selected = i;
        ImGui::PopID();
    }
    if (to_remove >= 0) app.toggle_tracked(to_remove);

    ImGui::End();
}

// ---------------------------------------------------------------------------
// Selection a la souris
// ---------------------------------------------------------------------------

int pick_object(const App& app, const Camera3D& cam, float point_radius) {
    Ray ray = GetScreenToWorldRay(GetMousePosition(), cam);
    int best = -1;
    float best_dist = 1e30f;
    // Rayon de test elargi : viser un point de quelques pixels reste difficile.
    float radius = std::max(point_radius * 3.0f, 0.05f);
    for (size_t i = 0; i < app.sim.objects.size(); ++i) {
        if (!app.visible[i]) continue;
        RayCollision hit = GetRayCollisionSphere(ray, to_render(app.sim.objects[i].s.r), radius);
        if (hit.hit && hit.distance < best_dist) { best_dist = hit.distance; best = static_cast<int>(i); }
    }
    return best;
}

}  // namespace

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::string catalog = "data/satellites_20260801_1000Z.txt";
    std::string screenshot;
    int screenshot_frames = 45;

    bool start_play = false, start_detect = false, start_grid = false;
    double start_scale = 1.0;
    int threads = 0;   // 0 = defaut (coeurs physiques)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--screenshot" && i + 1 < argc) screenshot = argv[++i];
        else if (a == "--frames" && i + 1 < argc) screenshot_frames = std::atoi(argv[++i]);
        else if (a == "--play") start_play = true;
        else if (a == "--detect") { start_detect = true; start_play = true; }
        else if (a == "--grid") start_grid = true;
        else if (a == "--radius-scale" && i + 1 < argc) start_scale = std::atof(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (!a.empty() && a[0] != '-') catalog = a;
    }

    int used_threads = configure_threads(threads);

    App app;
    app.playing = start_play;
    app.detect = start_detect;
    app.show_grid = start_grid;
    app.screen_cfg.radius_scale = start_scale;
    app.catalog_path = catalog;
    int dropped = 0;
    app.sim.objects = load_catalog(catalog, &dropped);
    if (app.sim.objects.empty()) {
        std::printf("Aucun objet charge depuis %s\n", catalog.c_str());
        std::printf("Lance le viewer depuis la racine du depot.\n");
        return 1;
    }
    app.sim.epoch_jd = read_epoch_jd(catalog);
    for (const auto& o : app.sim.objects) app.initial.push_back(o.s);
    app.alt.resize(app.sim.objects.size());
    app.inc.resize(app.sim.objects.size());
    app.visible.assign(app.sim.objects.size(), 1);
    app.scan_snapshots();
    std::printf("%zu objets charges (%d lignes ignorees), %d threads.\n",
                app.sim.objects.size(), dropped, used_threads);

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(1600, 900, "Kessler_Sim - viewer");
    SetTargetFPS(60);
    rlImGuiSetup(true);

    // Un cube minuscule par objet : 4 appels instancies suffisent pour tout
    // le catalogue, et le tampon de profondeur gere l'occultation.
    Mesh point_mesh = GenMeshCube(1.0f, 1.0f, 1.0f);   // GenMesh* televerse deja le maillage

    Shader instanced = LoadShaderFromMemory(VS_INSTANCED, FS_INSTANCED);
    instanced.locs[SHADER_LOC_MATRIX_MVP] = GetShaderLocation(instanced, "mvp");
    instanced.locs[SHADER_LOC_COLOR_DIFFUSE] = GetShaderLocation(instanced, "colDiffuse");
    instanced.locs[SHADER_LOC_MATRIX_MODEL] = GetShaderLocationAttrib(instanced, "instanceTransform");

    Material point_mat[TYPE_COUNT];
    for (int t = 0; t < TYPE_COUNT; ++t) {
        point_mat[t] = LoadMaterialDefault();
        point_mat[t].shader = instanced;
        point_mat[t].maps[MATERIAL_MAP_DIFFUSE].color = TYPE_COLOR[t];
    }

    Model earth = LoadModelFromMesh(GenMeshSphere(static_cast<float>(R_EARTH) * SCALE, 24, 48));

    std::vector<Matrix> transforms[TYPE_COUNT];
    int frame = 0;

    while (!WindowShouldClose()) {
        ImGuiIO& io = ImGui::GetIO();

        // --- Entrees ---
        if (!io.WantCaptureMouse) {
            app.cam.handle_input();
            if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
                int hit = pick_object(app, app.cam.to_camera3d(),
                                      app.cam.distance * app.point_scale);
                if (hit >= 0) app.selected = hit;
            }
        }
        if (!io.WantCaptureKeyboard && IsKeyPressed(KEY_SPACE)) app.playing = !app.playing;

        // --- Avancement du temps ---
        if (app.playing) {
            if (app.replay_mode) {
                if (!app.snapshots.empty())
                    app.load_snapshot_at((app.snapshot_index + 1) %
                                         static_cast<int>(app.snapshots.size()));
            } else if (app.detect) {
                // Detection a chaque pas : c'est la condition pour ne rien
                // rater, le TCA etant cherche a l'interieur du pas.
                for (int k = 0; k < app.steps_per_frame; ++k) {
                    app.before.resize(app.sim.objects.size());
                    for (size_t i = 0; i < app.sim.objects.size(); ++i)
                        app.before[i] = app.sim.objects[i].s;
                    double t0 = app.sim.t;
                    app.sim.step();
                    app.run_detection(t0);
                }
            } else {
                app.sim.advance(app.steps_per_frame);
            }
        }

        float dt_frame = GetFrameTime();
        for (App::Event& e : app.events) e.age += dt_frame;

        app.refresh_derived();

        if (app.follow_selected && app.selected >= 0)
            app.cam.target = to_render(app.sim.objects[app.selected].s.r);
        else
            app.cam.target = {0, 0, 0};

        // --- Matrices d'instances ---
        float pr = app.cam.distance * app.point_scale;
        for (int t = 0; t < TYPE_COUNT; ++t) transforms[t].clear();
        for (size_t i = 0; i < app.sim.objects.size(); ++i) {
            if (!app.visible[i]) continue;
            Vector3 p = to_render(app.sim.objects[i].s.r);
            transforms[static_cast<int>(app.sim.objects[i].type)].push_back(
                MatrixMultiply(MatrixScale(pr, pr, pr), MatrixTranslate(p.x, p.y, p.z)));
        }

        // --- Rendu ---
        Camera3D cam3d = app.cam.to_camera3d();
        BeginDrawing();
        ClearBackground({8, 10, 18, 255});

        BeginMode3D(cam3d);
        DrawModel(earth, {0, 0, 0}, 1.0f, {28, 62, 112, 255});
        DrawSphereWires({0, 0, 0}, static_cast<float>(R_EARTH) * SCALE * 1.002f, 12, 24,
                        {60, 110, 170, 110});

        for (int t = 0; t < TYPE_COUNT; ++t)
            if (!transforms[t].empty())
                DrawMeshInstanced(point_mesh, point_mat[t], transforms[t].data(),
                                  static_cast<int>(transforms[t].size()));

        // Traces des objets suivis + mise en evidence
        for (int i : app.tracked) {
            std::vector<Vector3> trail = propagate_orbit(app, app.sim.objects[i], 160).points;
            for (size_t k = 1; k < trail.size(); ++k)
                DrawLine3D(trail[k - 1], trail[k], {255, 255, 120, 180});
        }
        if (app.selected >= 0) {
            Vector3 p = to_render(app.sim.objects[app.selected].s.r);
            DrawSphereWires(p, pr * 4.0f, 6, 8, WHITE);
        }

        // Rapprochements recents : un segment entre les deux objets, qui
        // s'efface en quelques secondes. Rouge pour une collision.
        for (const App::Event& e : app.events) {
            if (e.age > app.event_lifetime) continue;
            float f = 1.0f - e.age / app.event_lifetime;
            Color col = e.collision
                ? Color{255, 70, 70, static_cast<unsigned char>(255 * f)}
                : Color{255, 235, 130, static_cast<unsigned char>(210 * f)};
            Vector3 a = to_render(e.ri), b = to_render(e.rj);
            DrawLine3D(a, b, col);
            DrawSphereWires({(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2},
                            pr * (e.collision ? 6.0f : 3.0f), 5, 6, col);
        }

        if (app.show_grid) draw_grid_debug(app);

        // Lune, a l'echelle et a sa vraie position
        Vector3 moon = to_render(app.sim.moon());
        DrawSphere(moon, static_cast<float>(R_MOON) * SCALE, {190, 190, 185, 255});

        EndMode3D();

        // Etiquettes des objets suivis, projetees en 2D
        for (int i : app.tracked) {
            Vector3 p = to_render(app.sim.objects[i].s.r);
            Vector2 sp = GetWorldToScreen(p, cam3d);
            if (sp.x > 0 && sp.y > 0 && sp.x < GetScreenWidth() && sp.y < GetScreenHeight())
                DrawText(app.sim.objects[i].name.c_str(), static_cast<int>(sp.x) + 8,
                         static_cast<int>(sp.y) - 6, 14, {255, 255, 150, 220});
        }

        DrawText("Glisser : tourner  |  Molette : zoom  |  Clic droit : selectionner  |  Espace : lecture",
                 12, GetScreenHeight() - 24, 16, {150, 160, 180, 200});

        rlImGuiBegin();
        draw_simulation_panel(app);
        draw_filters_panel(app);
        draw_selection_panel(app);
        draw_detection_panel(app);
        rlImGuiEnd();

        EndDrawing();

        ++frame;
        if (!screenshot.empty() && frame >= screenshot_frames) {
            // TakeScreenshot ne retient que le nom de fichier et ecrit dans le
            // repertoire courant : on deplace ensuite vers le chemin demande.
            std::filesystem::path want(screenshot);
            TakeScreenshot(want.filename().string().c_str());
            if (want.has_parent_path()) {
                std::error_code ec;
                std::filesystem::create_directories(want.parent_path(), ec);
                std::filesystem::rename(want.filename(), want, ec);
            }
            break;
        }
    }

    UnloadModel(earth);
    UnloadShader(instanced);
    UnloadMesh(point_mesh);
    rlImGuiShutdown();
    CloseWindow();
    return 0;
}
