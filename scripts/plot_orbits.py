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
import sys

import matplotlib.pyplot as plt
import numpy as np

from plot_common import OUT_DIR, draw_earth, finish, run_sim, set_equal_axes

# ISS, Hubble, Aqua, NOAA 18, NOAA 20, EWS-G2 (GEO)
DEFAULT_IDS = [25544, 20580, 27424, 28654, 43013, 36411]


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
        run_sim(args.hours, args.dt, args.every, args.ids or DEFAULT_IDS)

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
    finish(fig, args.save)


if __name__ == "__main__":
    main()
