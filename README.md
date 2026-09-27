# Kessler_Sim

Propagation orbitale de l'ensemble du catalogue Space-Track (~28 000 objets)
dans le système Terre-Lune, en C++.

Le projet part d'un état initial réel — tous les objets catalogués ramenés à un
**instant commun** par SGP4 — et les propage ensuite avec un intégrateur
numérique. L'objectif à terme est de simuler le syndrome de Kessler : détection
de collisions, fragmentation, cascade.

## État actuel

| Brique | État |
|---|---|
| Export du catalogue Space-Track à un epoch commun | fait |
| Propagation RK4 (Terre + J2 + Lune) | fait |
| Viewer 3D temps réel, filtres et suivi d'objets | fait |
| Tracés matplotlib (trajectoires, instantanés) | fait |
| Détection des rapprochements et collisions (déterministe) | fait |
| Fragmentation (NASA Standard Breakup Model), impacts provoqués | fait |
| Traînée atmosphérique | à faire — indispensable pour un seuil de Kessler |
| Collisions probabilistes (méthode CUBE), pour les runs longs | à faire |

## Structure

```
.
├── src/
│   ├── core/orbital.*      dynamique partagée : RK4, J2, Lune, entrées/sorties
│   ├── core/collision.*    détection des rapprochements : grille + instant de rapprochement
│   ├── core/breakup.*      fragmentation (NASA Standard Breakup Model), masses
│   ├── core/threads.*      choix du nombre de threads
│   ├── main.cpp            simulation en ligne de commande
│   └── viewer/viewer.cpp   viewer 3D temps réel (raylib + Dear ImGui)
├── scripts/
│   ├── spacetrack_export.py   télécharge les TLE et les propage à un epoch commun
│   ├── plot_orbits.py         trace les trajectoires de quelques objets
│   ├── plot_snapshot.py       trace un instantané du catalogue (nuage de points)
│   ├── plot_common.py         briques partagées par les deux scripts de tracé
│   └── check_duplicates.py    vérifie l'absence de doublons dans un export
├── config/masses.csv       masses estimées du catalogue, par taille radar et type
├── data/                   états initiaux (versionnés)
├── output/                 sorties de simulation (ignorées par git)
└── CMakeLists.txt
```

## Démarrage rapide

```bash
cmake -B build
cmake --build build
```

Deux binaires apparaissent à la racine : `kessler_sim` (simulation) et
`kessler_viewer` (viewer 3D). La configuration télécharge raylib, Dear ImGui et
rlImGui — comptez quelques minutes la première fois, et une connexion.

> Les commandes de ce README sont écrites une par ligne exprès : Windows
> PowerShell 5.1 ne connaît pas l'opérateur `&&`. Pour enchaîner sous
> PowerShell, utilisez `;` ou `; if ($?) { ... }`. PowerShell 7 et les shells
> POSIX acceptent `&&` sans problème.

Pour ne compiler que la simulation, sans dépendance ni réseau :

```bash
cmake -B build -DKESSLER_BUILD_VIEWER=OFF
cmake --build build
```

Sans CMake du tout, la simulation seule se compile en une ligne :

```bash
g++ -O2 -std=c++17 -fopenmp -Isrc src/core/orbital.cpp src/core/collision.cpp src/main.cpp -o kessler_sim
```

Sur MSYS2, CMake s'installe avec :

```bash
pacman -S mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
```

Puis, **depuis la racine du dépôt** (les chemins sont relatifs au répertoire
courant). Sous Windows, remplacez `./kessler_sim` par `.\kessler_sim.exe` :

```bash
./kessler_sim data/satellites_20260801_1000Z.txt 24 10
```

Sur les 28 340 objets, 24 h simulées prennent environ 4 s (8 cœurs, pas de 10 s).

### Arguments

```
kessler_sim [fichier] [durée_h] [pas_s] [période_sortie_s] [ids_norad]
```

| Argument | Défaut | Rôle |
|---|---|---|
| `fichier` | `data/satellites_20260801_1000Z.txt` | état initial |
| `durée_h` | 24 | durée simulée, en heures |
| `pas_s` | 10 | pas d'intégration, en secondes |
| `période_sortie_s` | 0 | intervalle entre deux sorties CSV (0 = état final seulement) |
| `ids_norad` | — | liste séparée par des virgules ; produit `trajectories.csv` au lieu des snapshots |

Tout est écrit dans `output/` : `final_state.csv`, les `snapshot_N.csv`, et
`trajectories.csv` + `moon.csv` quand des identifiants sont donnés.

## Rapprochements et collisions

```bash
./kessler_sim data/satellites_20260801_1000Z.txt 24 10 --conj-km 5
```

Détecte, pendant la propagation, tous les passages à moins de `--conj-km`
kilomètres, et signale une collision quand la distance descend sous la somme
des rayons des deux objets. Le détail va dans `output/conjunctions.csv` :
instant, identifiants, distance de passage, vitesse relative.

| Option | Défaut | Rôle |
|---|---|---|
| `--screen` | — | active la détection avec le seuil par défaut |
| `--conj-km X` | 5 | seuil de rapprochement retenu (active la détection) |
| `--radius-scale S` | 1 | multiplie tous les rayons de collision |
| `--min-vrel V` | 10 | vitesse relative minimale d'une rencontre, en m/s |
| `--threads N` | tous | nombre de threads OpenMP |

**Principe.** Une grille de hachage ne retient que les paires assez proches en
début de pas pour pouvoir se rencontrer pendant le pas — la taille de cellule
suit la vitesse relative maximale fois le pas. Pour ces paires, l'instant de
rapprochement maximal est calculé analytiquement, puis affiné par
interpolation d'Hermite entre le début et la fin du pas.

Le point clé : deux objets proches subissent presque la même gravité, donc
leur mouvement *relatif* est quasi rectiligne sur un pas, même si leurs
trajectoires absolues sont courbes. Une collision est détectée même si les
deux objets se sont traversés entre deux pas — **la détection ne dépend pas du
pas d'intégration**. Vérifié : sur 2 h, les 2 969 rapprochements trouvés à
10 s de pas le sont tous à 1 s, appariés un pour un, avec au plus 9 mm d'écart
sur la distance et 5 µs sur l'instant.

**Rayons de collision.** Le catalogue ne donne qu'une catégorie de section
radar : 0,1 m (SMALL), 0,4 m (MEDIUM, et objets sans catégorie), 2 m (LARGE).
Valeurs indicatives, à calibrer ; `--radius-scale` les multiplie toutes.

**Paires volant de concert.** Deux objets lancés ensemble peuvent voler à
quelques centaines de mètres l'un de l'autre, à quelques m/s relatifs. Leur
distance varie à peine d'un pas à l'autre, l'instant de rapprochement n'est pas
défini, et un contact ne serait de toute façon pas une fragmentation
hypervéloce. Sous `--min-vrel`, ces paires sont comptées à part au lieu d'être
traitées comme des rencontres.

### Premier bilan sur 24 h

| Distance de passage | Rapprochements |
|---|---|
| < 5 km | 38 367 |
| < 1 km | 1 493 |
| < 100 m | 14 |
| collision (rayons réels) | 0 |

Le plus serré : 7,2 m entre STARLINK-3051 et STARLINK-34756. Vitesse relative
médiane : 11,5 km/s.

Le nombre de passages à moins de *d* croît comme *d²* — ~1 500 par km² et par
jour, constant de 100 m à 5 km. C'est ce qu'on attend si les distances de
passage se répartissent au hasard dans le plan de rencontre, et cela donne un
ordre de grandeur direct du taux de collision :

> taux ≈ 1 500 × (R₁ + R₂)² par jour, avec R en km

Multiplier les rayons par *S* (`--radius-scale`) multiplie ce taux par *S²*.
Attention à la somme des rayons à retenir : le catalogue est dominé par des
objets LARGE (rayon 2 m), elle vaut donc plutôt 2 à 4 m qu'un mètre. Mesuré à
×100 : **100 à 130 collisions par jour** entre objets du catalogue, soit ~4 à 5
par an aux rayons réels. C'est plus que la réalité, où les collisions
accidentelles entre objets catalogués se comptent sur une décennie : les
rayons LARGE sont généreux, et les satellites manœuvrables — Starlink en tête —
évitent activement les rapprochements, ce que la simulation ignore.

**Limite importante.** Les états initiaux viennent de TLE propagés par SGP4,
dont l'erreur de position est de l'ordre du kilomètre. Chaque rapprochement
individuel n'a donc rien d'une prédiction réelle : ce sont les statistiques
qui sont représentatives, pas les événements.

### Coût et parallélisation

Mesures sur i7-11800H (8 cœurs, 16 threads logiques), 1 h simulée à dt = 10 s,
configurations alternées pour annuler la dérive thermique :

| Threads | Propagation | Grille | Paires | Total |
|---|---|---|---|---|
| 1 | 0,587 s | 0,470 s | 2,609 s | 3,67 s |
| 8 | 0,156 s | 0,509 s | 0,437 s | 1,10 s |
| 16 | 0,120 s | 0,511 s | 0,301 s | 0,93 s |

La propagation accélère 4,9 fois, le parcours des paires 8,7 fois.
**La construction de la grille est séquentielle et ne gagne rien** : à
16 threads elle représente plus de la moitié du temps total, c'est le facteur
limitant au sens d'Amdahl.

**Le pas optimal n'est pas le plus petit.** Le coût de la propagation varie en
1/dt, mais les cellules doivent grandir avec le pas, donc les paires candidates
croissent en dt³ et le parcours en dt². Sur 6 h simulées :

| Pas | Total | Paires candidates/pas | Écart de distance vs dt = 10 s |
|---|---|---|---|
| 10 s | 6,6 s | 124 000 | référence |
| **30 s** | **3,4 s** | 1 690 000 | médian 1 cm, max 5,9 m |
| 60 s | 5,5 s | 7 500 000 | médian 16 cm, max 131 m |
| 120 s | 25,6 s | 31 800 000 | — |

Les 9 033 rapprochements sont les mêmes à 10, 30 et 60 s : seule la précision
des distances se dégrade. Donc **dt = 30 s pour les runs longs**, dt = 10 s
quand la distance exacte compte.

À dt = 30 s : environ 14 s par jour simulé, soit ~1 h 25 par année simulée.
Attention, le coût du parcours croît en N², pas en N : une cascade qui
multiplierait le nombre d'objets par 4 multiplierait ce terme par 16.

## Fragmentation

```bash
./kessler_sim data/satellites_20260801_1000Z.txt 24 10 --impact 49157 --impact-mass 10
./kessler_sim data/satellites_20260801_1000Z.txt 48 30 --breakup --radius-scale 100
```

La première commande provoque l'impact d'un projectile fictif de 10 kg sur
STARLINK-3051 à 10 km/s, puis suit le nuage. La seconde laisse les collisions
survenir d'elles-mêmes, avec une section efficace gonflée pour déclencher une
cascade en quelques heures au lieu de siècles.

| Option | Défaut | Rôle |
|---|---|---|
| `--breakup` | — | les collisions détectées fragmentent les objets |
| `--impact NORAD` | — | impact d'un projectile fictif sur cet objet (active `--breakup`) |
| `--impact-mass KG` | 10 | masse du projectile |
| `--impact-vrel KMS` | 10 | vitesse relative de l'impact |
| `--impact-at H` | 0 | instant de l'impact, en heures |
| `--lc L` | 0,1 | plus petit fragment suivi, en mètres |
| `--masses FICHIER` | `config/masses.csv` | masses estimées du catalogue |
| `--seed N` | 1 | graine du tirage : un run est reproductible |

Sorties : `output/breakups.csv` (une ligne par fragmentation : parents,
énergie, fragments) et `output/population.csv` (population heure par heure :
objets du catalogue, fragments, fragmentations cumulées) — de quoi tracer la
vitesse d'une cascade.

### Le modèle

C'est le *NASA Standard Breakup Model* (Johnson et al., 2001), référence des
modèles d'évolution des débris.

1. **Catastrophique ou non.** L'énergie cinétique du projectile rapportée à la
   masse de la cible, E = ½ m_p v² / m_t, est comparée à **40 J/g**. Au-dessus,
   les deux objets sont pulvérisés ; en dessous, le projectile est détruit et la
   cible écornée. À 10 km/s, un projectile de 0,08 % de la masse de la cible
   suffit : un débris de 800 g détruit un satellite d'une tonne.
2. **Combien de fragments.** N(> L) = 0,1 · M^0,75 · L^−1,71, avec M la masse
   mise en jeu. Seuls les fragments au-dessus de `--lc` sont suivis : pour une
   collision catastrophique entre deux objets d'une tonne, ~1 500 au-dessus de
   10 cm, ~80 000 au-dessus de 1 cm.
3. **Quels fragments.** Taille tirée dans cette loi de puissance ; rapport
   surface/masse tiré dans la distribution du SBM ; masse déduite des deux.
   Les fragments sont acceptés tant que la masse de leur parent le permet.
4. **À quelle vitesse.** Celle de *leur* parent plus un Δv isotrope de norme
   log-normale, de l'ordre de 100 m/s. Les fragments restent donc groupés
   autour de l'orbite de leur parent : deux nuages distincts, comme observé
   après la collision Iridium 33 / Cosmos 2251 en 2009.

Pourquoi pas un cône autour de la direction moyenne des deux objets : pour un
croisement typique à 100°, la vitesse moyenne ne vaut que ~4,8 km/s, et tous les
fragments retomberaient en moins d'une orbite. Ce modèle ferait disparaître la
cascade qu'on cherche à étudier.

**Masses.** Space-Track ne publie pas de masses : `config/masses.csv` en donne
une estimation par catégorie de taille radar et par type d'objet, à régler. Le
fichier indique la répartition du catalogue entre les cases, pour savoir
lesquelles comptent — les grosses charges utiles (13 578, dont 8 838 Starlink)
et les débris moyens (6 736) dominent.

### Vérifications

Sur un cas type Iridium 33 / Cosmos 2251 (950 kg + 560 kg à 11,7 km/s) :

| Grandeur | Résultat | Attendu |
|---|---|---|
| Énergie spécifique | 40 347 J/g, catastrophique | idem (calcul direct) |
| Fragments > 10 cm | 1 128 | 1 242 par la loi ; > 2 000 catalogués en réalité |
| N(>20 cm)/N(>10 cm) | 0,298 | 2^−1,71 = 0,306 |
| Δv médian | 105 m/s | ~100 m/s |
| Isotropie (moyenne des directions, 13 382 fragments) | 0,0071 | ~0,008 (bruit statistique) |

Le SBM sous-estime ce cas précis d'environ un facteur 2 : l'ordre de grandeur
est bon. Même graine, mêmes fragments : un run est exactement reproductible.

**La cascade entière est indépendante du pas.** Sur 6 h à rayons ×100, les
mêmes 51 fragmentations — mêmes paires, à la même seconde — à dt = 10 s et à
dt = 30 s.

### Une cascade, heure par heure

`--breakup --radius-scale 100`, dt = 30 s, sans impact provoqué — les
collisions surviennent d'elles-mêmes (run arrêté à 28 h) :

| Tranche | Fragmentations | Cumul | Fragments créés (cumul) |
|---|---|---|---|
| 0–4 h | 22 | 22 | 10 879 |
| 4–8 h | 59 | 81 | ~30 000 |
| 8–12 h | 97 | 178 | 57 865 |
| 12–16 h | 144 | 322 | 92 151 |
| 16–20 h | 187 | 509 | 125 699 |
| 20–24 h | 312 | 821 | 175 464 |
| 24–28 h | 325 | 1 146 | 237 253 |

Le rythme s'accélère, et la nature des collisions change : au début les
objets du catalogue se percutent entre eux ; à 28 h, 761 fragmentations
opposent un objet du catalogue à un fragment et 232 deux fragments, contre 153
entre objets du catalogue. **Les débris deviennent les projectiles** : c'est le
mécanisme de Kessler.

Le coût suit : la population passe de 28 000 à plus de 260 000 objets, et le
parcours des paires croît en N². Les premières heures se simulent en quelques
secondes ; au-delà de 24 h, chaque heure simulée prend plusieurs minutes.

### Deux réglages de performance

- **Seuil de détection.** En mode fragmentation, sauf `--conj-km` explicite,
  le seuil se cale sur la plus grande somme de rayons possible. Un seuil de
  5 km faisait affiner des centaines de milliers de passages entre fragments
  frères d'un même nuage, sans aucune conséquence.
- **Fragments échappés.** Un fragment qui reçoit assez de Δv pour une
  trajectoire hyperbolique quitte l'attraction terrestre : il est écarté à la
  création et compté à part. La grille dimensionne ses cellules sur l'objet le
  plus rapide ; 13 fragments à plus de 15 km/s suffisaient à en doubler la
  taille, et à multiplier par 8 les paires candidates.

### Limites

- **Sans traînée, pas de seuil de Kessler.** Le taux de collision croît comme
  N² et rien ne retire d'objets : toute situation finit en cascade, il suffit
  d'attendre. Le seuil est un équilibre entre création de débris et retombées
  dans l'atmosphère — la traînée est la prochaine brique.
- La quantité de mouvement n'est conservée qu'en moyenne (Δv isotropes),
  limite connue du SBM.
- Les coefficients de surface/masse sont ceux des fragments de satellite ; le
  SBM en a d'autres pour les étages de fusée, non distingués ici.
- Les masses du catalogue sont des estimations par catégorie.

## Viewer 3D

```bash
./kessler_viewer
```

Affiche l'intégralité du catalogue en temps réel. Le rendu passe par quatre
appels instanciés — un par catégorie d'objet — et tient les 28 340 objets à
60 fps. L'occultation par la Terre est gérée par le tampon de profondeur du
GPU, donc correctement, contrairement aux tracés matplotlib.

| Commande | Action |
|---|---|
| Glisser | tourner autour de la Terre |
| Molette | zoomer, de l'orbite basse à l'orbite lunaire |
| Clic droit | sélectionner l'objet sous le curseur |
| Espace | lancer ou suspendre la propagation |

**Deux modes.** *Live* propage avec le même RK4 que la simulation en ligne de
commande — le code de `src/core` est partagé, il n'y a pas deux dynamiques à
maintenir. Le pas et le nombre de pas par image se règlent en cours de route :
à 10 s de pas et 6 pas par image, le temps défile 3 600 fois plus vite que le
temps réel. *Relecture* rejoue les `snapshot_*.csv` produits par `kessler_sim`,
donc exactement ce que la simulation a calculé :

```bash
./kessler_sim data/satellites_20260801_1000Z.txt 6 10 1800   # 12 snapshots
./kessler_viewer                                             # puis mode Relecture
```

**Filtres.** Cases par catégorie, plages d'altitude et d'inclinaison,
recherche par nom ou par NORAD, et raccourcis LEO / MEO / GEO. Le compte
d'objets affichés se met à jour en direct.

**Suivi.** Un objet sélectionné affiche ses éléments orbitaux (périgée, apogée,
inclinaison, excentricité, période, vitesse). Le bouton *Suivre* trace son
orbite complète et l'étiquette dans la vue ; *Caméra liée* centre la vue
dessus. Plusieurs objets peuvent être suivis en même temps.

**Rapprochements en direct.** En mode Live, le panneau *Rapprochements* active
la détection pendant que la simulation tourne. Chaque passage sous le seuil
trace un segment entre les deux objets, qui s'efface en quelques secondes —
rouge s'il s'agit d'une collision. La liste donne distance, noms, instant et
vitesse relative ; un clic sélectionne l'objet et y accroche la caméra. Le
seuil et le facteur de rayon se règlent sans interrompre la simulation.

La détection coûte ~2 ms par pas, et elle tourne à *chaque* pas — c'est la
condition pour ne rien rater. Avec 6 pas par image, comptez ~15 ms par image :
le panneau affiche le coût et prévient quand il devient sensible.

**Débogage de la grille.** La case *Debug : grille de détection* trace les
cellules. Avec un objet sélectionné, elle montre sa cellule et les 26 voisines,
c'est-à-dire exactement le voisinage examiné ; sans sélection, les cellules
occupées les plus proches de la caméra.

**Fragmentation.** La case *Fragmentation (SBM)* du panneau *Rapprochements*
fait fragmenter chaque collision détectée ; les fragments forment une
catégorie à part, en magenta, filtrable comme les autres. Chaque événement
s'affiche comme une sphère qui s'étend puis s'efface, rouge s'il est
catastrophique, et la liste des fragmentations récentes en donne le détail.

**Provoquer un impact.** Dans *Sélection*, choisis une masse de projectile et
une vitesse relative : le panneau indique l'énergie et si l'impact sera
catastrophique. Le bouton *Impact* frappe l'objet sélectionné, active détection
et fragmentation, puis place la caméra au-dessus du nuage et la fait suivre un
fragment — la cible détruite, elle, n'est plus propagée. *Retour à l'epoch*
retire tous les fragments.

```bash
./kessler_viewer --impact 49157 --impact-mass 10
```

**État de départ**, pratique pour scripter : `--play` lance la propagation,
`--detect` active aussi la détection, `--breakup` la fragmentation, `--grid`
affiche la grille, `--radius-scale S` multiplie les rayons de collision, et
`--impact NORAD` (avec `--impact-mass`, `--impact-vrel`) frappe un objet au
démarrage.

Le viewer écrit un `imgui.ini` à la racine pour mémoriser la disposition des
panneaux. Il est dans le `.gitignore` ; le supprimer rétablit la disposition
d'origine.

Pour une image sans interaction (utile en capture ou en script) :

```bash
./kessler_viewer --screenshot output/vue.png --frames 60
```

## Tracés matplotlib

Plus limités que le viewer, mais pratiques pour produire une figure fixe.

```bash
pip install -r requirements.txt
```

Deux scripts, qui lancent la simulation puis tracent le résultat. `--no-run`
réutilise le CSV existant, `--save FICHIER` enregistre au lieu d'afficher.

**Trajectoires de quelques objets** — `plot_orbits.py`

```bash
python scripts/plot_orbits.py                       # 6 h, sélection par défaut
python scripts/plot_orbits.py 25544 20580 --hours 3 # ISS + Hubble
python scripts/plot_orbits.py 36411 --hours 48 --moon
```

Sans argument : ISS, Hubble, Aqua, NOAA 18, NOAA 20 et EWS-G2 (géostationnaire).
`--moon` ajoute la Lune, ce qui étend l'échelle à ~400 000 km et réduit les
orbites basses à un point.

**Instantané du catalogue** — `plot_snapshot.py`

Nuage de points d'un échantillon d'objets, coloré par catégorie (charge utile,
débris, étage de fusée).

```bash
python scripts/plot_snapshot.py                     # 1000 objets, vue 3D
python scripts/plot_snapshot.py -n 5000 --hours 72
python scripts/plot_snapshot.py --max-alt 2000      # LEO seulement
python scripts/plot_snapshot.py --view alt-inc -n 28340 --size 2
python scripts/plot_snapshot.py --file output/snapshot_3.csv --no-run
```

| Option | Rôle |
|---|---|
| `-n` | taille de l'échantillon (défaut 1000) |
| `--view` | `3d` (défaut) ou `alt-inc` |
| `--max-alt` | ne garder que les objets sous cette altitude, en km |
| `--file` | CSV à tracer (défaut `output/final_state.csv`) |
| `--seed` | graine de l'échantillonnage, pour un tirage reproductible |
| `--earth-alpha` | opacité du globe en vue 3D (défaut 1) |
| `--show-hidden` | ne pas masquer les objets situés derrière la Terre |

La vue `alt-inc` place l'altitude en abscisse (échelle log) et l'inclinaison en
ordonnée. C'est le tracé classique en analyse de débris : chaque amas y
correspond à un régime orbital — la bande héliosynchrone vers 98°, les
constellations autour de 53°, les GNSS vers 20 000 km, et le mur géostationnaire
à 35 786 km.

En vue 3D, les quelques objets très hauts écrasent l'échelle et compriment la
couche basse en une coquille ; `--max-alt 2000` donne une vue LEO lisible.

#### Occultation par la Terre

`mplot3d` ne dispose pas de tampon de profondeur : il trie les artistes entre
eux, pas fragment par fragment. Un nuage de points passe donc *entièrement*
devant ou derrière le globe, et augmenter l'opacité de la Terre n'y change
rien — les objets de l'autre côté restent visibles au travers.

Le script corrige ça en testant lui-même, point par point, si l'objet tombe
dans la silhouette du globe du côté opposé à la caméra ; les points concernés
sont rendus transparents. Le test est refait à chaque rotation de la vue. Il
est exact en projection orthographique, et très légèrement approché au ras du
limbe en projection perspective (celle par défaut).

`--show-hidden` rétablit l'ancien comportement. Ce traitement ne s'applique
qu'au nuage de points : les trajectoires de `plot_orbits.py`, étant des lignes
continues, traversent toujours le globe.

## Modèle physique

Implémenté une seule fois dans `src/core/orbital.cpp`, et utilisé tel quel par
la simulation comme par le viewer.

Intégrateur Runge-Kutta d'ordre 4, en repère inertiel géocentrique (ECI), avec :

- gravité terrestre ponctuelle (`μ = 398600.4418 km³/s²`) ;
- aplatissement terrestre, terme `J2` ;
- perturbation lunaire (terme direct et terme indirect), la Lune suivant une
  éphéméride analytique de Meeus plutôt qu'une intégration.

Un objet est marqué perdu dès que son rayon passe sous `R_terre`.

**Non modélisés** : traînée atmosphérique, Soleil, pression de radiation,
harmoniques au-delà de `J2`. Sans traînée, les orbites basses ne décroissent
jamais — ce qui compte pour un Kessler à long terme reste donc à ajouter.

Un pas de 10 s et un pas de 2 s donnent des positions qui diffèrent d'environ
0,5 m au maximum sur 24 h, ce qui valide le pas par défaut. Aucune comparaison
avec SGP4 au-delà de l'instant initial n'a été faite.

## Régénérer les données

`data/satellites_20260801_1000Z.txt` est fourni. Pour produire un autre epoch,
il faut un compte [Space-Track](https://www.space-track.org/auth/createAccount)
(gratuit) :

```bash
cp .env.example .env    # puis remplir SPACETRACK_USER et SPACETRACK_PASS
python scripts/spacetrack_export.py
```

Le script récupère la liste des objets non désorbités, télécharge leur TLE le
plus récent avant la date cible, puis les propage tous au même instant par
SGP4. Compter une vingtaine de minutes : Space-Track limite le débit à 30
requêtes par minute, et le catalogue est interrogé par lots.

La date cible se règle par `TARGET_DATE` en tête du script. Elle doit être
assez ancienne — quelques semaines — car la classe `gp_history` n'est pas
peuplée pour les dates récentes.

Le `.env` est dans le `.gitignore` et ne doit jamais être commité.

## Licence

MIT — voir [LICENSE](LICENSE).
