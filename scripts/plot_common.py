#!/usr/bin/env python3
"""
plot_common.py
--------------
Briques partagees par plot_orbits.py et plot_snapshot.py : resolution des
chemins du depot, lancement du binaire, lecture des CSV de sortie et
elements communs des figures 3D.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "satellites_20260801_1000Z.txt"
OUT_DIR = ROOT / "output"

R_EARTH = 6378.137  # km

# Couleurs par categorie d'objet (colonne OBJECT_TYPE de l'export Space-Track).
TYPE_COLORS = {
    "PAYLOAD": "tab:blue",
    "DEBRIS": "tab:red",
    "ROCKET BODY": "tab:orange",
    "UNKNOWN": "gray",
}


# ---------------------------------------------------------------------------
# Binaire de simulation
# ---------------------------------------------------------------------------

def find_executable():
    """Cherche kessler_sim a la racine puis dans les repertoires de build usuels."""
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


def run_sim(hours, dt, every=0, ids=None):
    """Lance la simulation depuis la racine du depot (les sorties vont dans output/)."""
    if not DATA_FILE.exists():
        sys.exit(f"{DATA_FILE} introuvable : lance d'abord scripts/spacetrack_export.py.")
    cmd = [str(find_executable()), str(DATA_FILE), str(hours), str(dt), str(every)]
    if ids:
        cmd.append(",".join(map(str, ids)))
    subprocess.run(cmd, cwd=ROOT, check=True)


# ---------------------------------------------------------------------------
# Lecture des sorties
# ---------------------------------------------------------------------------

def read_csv_rows(path):
    """Lit un CSV produit par kessler_sim en ignorant les lignes de commentaire.

    Retourne (liste de dict, temps en secondes lu dans l'en-tete ou None).
    """
    if not path.exists():
        sys.exit(f"{path} introuvable.")
    t = None
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                if "t =" in line:
                    try:
                        t = float(line.split("t =")[1].split("s")[0])
                    except (IndexError, ValueError):
                        pass
                continue
            lines.append(line)
    return list(csv.DictReader(lines)), t


def load_object_types():
    """Retourne {norad_id (str): OBJECT_TYPE} depuis l'export Space-Track."""
    types = {}
    if not DATA_FILE.exists():
        return types
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split("|")
            if len(fields) > 2:
                types[fields[0].strip()] = fields[2].strip()
    return types


# ---------------------------------------------------------------------------
# Elements de figure
# ---------------------------------------------------------------------------

def draw_earth(ax, alpha=0.25):
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(
        R_EARTH * np.cos(u) * np.sin(v),
        R_EARTH * np.sin(u) * np.sin(v),
        R_EARTH * np.cos(v),
        color="tab:blue", alpha=alpha, linewidth=0,
    )


def set_equal_axes(ax, extent):
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_box_aspect((1, 1, 1))


def finish(fig, save):
    """Enregistre ou affiche la figure selon l'option --save."""
    import matplotlib.pyplot as plt
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
        print(f"Image enregistree dans {save}")
    else:
        plt.show()
