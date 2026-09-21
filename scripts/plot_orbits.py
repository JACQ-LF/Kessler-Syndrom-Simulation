#!/usr/bin/env python3
"""
plot_orbits.py
--------------
Lance kessler_sim sur une poignee de satellites et trace leurs trajectoires
(repere inertiel ECI) en 3D autour de la Terre.

Utilisation (depuis n'importe ou, les chemins sont resolus par rapport au depot) :
    python scripts/plot_orbits.py                       # selection par defaut, 6 h
    python scripts/plot_orbits.py 25544 20580 --hours 3 # ISS + Hubble
    python scripts/plot_orbits.py 36411 --hours 48 --moon
    python scripts/plot_orbits.py --no-run              # re-trace output/trajectories.csv
    python scripts/plot_orbits.py --save orbites.png    # enregistre au lieu d'afficher

Prerequis : kessler_sim compile (voir README), numpy, matplotlib.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "satellites_20260801_1000Z.txt"
OUT_DIR = ROOT / "output"

R_EARTH = 6378.137  # km

# ISS, Hubble, Aqua, NOAA 18, NOAA 20, EWS-G2 (GEO)
DEFAULT_IDS = [25544, 20580, 27424, 28654, 43013, 36411]


def find_executable():
    """Cherche le binaire a la racine puis dans les repertoires de build usuels."""
    name = "kessler_sim.exe" if os.name == "nt" else "kessler_sim"
    candidates = [ROOT / name, ROOT / "build" / name,
                  ROOT / "build" / "Release" / name, ROOT / "build" / "Debug" / name]
    for path in candidates:
        if path.exists():
            return path
    sys.exit(
        f"{name} introuvable. Compile d'abord :\n"
        f"    cmake -B build && cmake --build build --config Release"
    )


def run_sim(ids, hours, dt, every):
    if not DATA_FILE.exists():
        sys.exit(f"{DATA_FILE} introuvable : lance d'abord scripts/spacetrack_export.py.")
    # cwd = racine du depot : le binaire ecrit dans output/ relativement au cwd.
    subprocess.run(
        [str(find_executable()), str(DATA_FILE), str(hours), str(dt), str(every),
         ",".join(map(str, ids))],
        cwd=ROOT, check=True,
    )


def load_trajectories(path):
    """Retourne {(id, nom): tableau (N, 3) de positions ECI en km}."""
    if not path.exists():
        sys.exit(f"{path} introuvable : relance sans --no-run.")
    trajs = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["id"], row["name"])
            trajs.setdefault(key, []).append(
                (float(row["x_km"]), float(row["y_km"]), float(row["z_km"]))
            )
    return {k: np.array(v) for k, v in trajs.items()}


def draw_earth(ax):
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(
        R_EARTH * np.cos(u) * np.sin(v),
        R_EARTH * np.sin(u) * np.sin(v),
        R_EARTH * np.cos(v),
        color="tab:blue", alpha=0.25, linewidth=0,
    )


def set_equal_axes(ax, extent):
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_box_aspect((1, 1, 1))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("ids", nargs="*", type=int,
                   help="NORAD_ID a tracer (defaut : selection variee)")
    p.add_argument("--hours", type=float, default=6.0, help="duree simulee (defaut 6)")
    p.add_argument("--dt", type=float, default=10.0, help="pas d'integration en s")
    p.add_argument("--every", type=float, default=60.0, help="echantillonnage en s")
    p.add_argument("--moon", action="store_true", help="tracer aussi la Lune")
    p.add_argument("--no-run", action="store_true", help="ne pas relancer la simulation")
    p.add_argument("--save", metavar="FICHIER", help="enregistrer l'image au lieu de l'afficher")
    args = p.parse_args()

    if not args.no_run:
        run_sim(args.ids or DEFAULT_IDS, args.hours, args.dt, args.every)

    trajs = load_trajectories(OUT_DIR / "trajectories.csv")
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection="3d")
    draw_earth(ax)

    extent = 0.0
    for (nid, name), pts in trajs.items():
        ax.plot(*pts.T, lw=1.2, label=f"{name} ({nid})")
        ax.scatter(*pts[0], s=15)  # position initiale
        extent = max(extent, np.abs(pts).max())

    moon_file = OUT_DIR / "moon.csv"
    if args.moon and moon_file.exists():
        moon = np.loadtxt(moon_file, delimiter=",", skiprows=1)[:, 1:4]
        ax.plot(*moon.T, color="gray", lw=2, label="Lune")
        ax.scatter(*moon[-1], color="gray", s=60)
        extent = max(extent, np.abs(moon).max())

    set_equal_axes(ax, extent * 1.05)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title(f"Trajectoires ECI — {args.hours:g} h")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"Image enregistree dans {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
