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
| Visualisation 3D des trajectoires | fait |
| Détection de collisions | à faire |
| Modèle de fragmentation | à faire |
| Traînée atmosphérique | à faire |

## Structure

```
.
├── src/main.cpp            simulateur (lecture, RK4, sorties CSV)
├── scripts/
│   ├── spacetrack_export.py   télécharge les TLE et les propage à un epoch commun
│   ├── plot_orbits.py         trace les trajectoires de quelques objets
│   ├── plot_snapshot.py       trace un instantané du catalogue (nuage de points)
│   ├── plot_common.py         briques partagées par les deux scripts de tracé
│   └── check_duplicates.py    vérifie l'absence de doublons dans un export
├── data/                   états initiaux (versionnés)
├── output/                 sorties de simulation (ignorées par git)
└── CMakeLists.txt
```

## Démarrage rapide

```bash
cmake -B build && cmake --build build --config Release
```

Sans CMake, la compilation directe fonctionne aussi :

```bash
g++ -O2 -std=c++17 -fopenmp src/main.cpp -o kessler_sim
```

Puis, **depuis la racine du dépôt** (les chemins sont relatifs au répertoire courant) :

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

### Visualisation

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
