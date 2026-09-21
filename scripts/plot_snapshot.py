#!/usr/bin/env python3
"""
plot_snapshot.py
----------------
Trace un instantane du catalogue : un nuage de points (scatter) d'un
echantillon d'objets, colore par categorie (charge utile / debris / etage).

Deux vues :
    --view 3d        positions ECI autour de la Terre (defaut)
    --view alt-inc   altitude vs inclinaison, la vue classique en analyse
                     de debris : chaque amas correspond a un regime orbital

Utilisation :
    python scripts/plot_snapshot.py                        # simule 24 h, 1000 objets
    python scripts/plot_snapshot.py -n 5000 --hours 72
    python scripts/plot_snapshot.py --no-run               # utilise le CSV existant
    python scripts/plot_snapshot.py --file output/snapshot_3.csv --no-run
    python scripts/plot_snapshot.py --max-alt 2000         # LEO seulement
    python scripts/plot_snapshot.py --view alt-inc -n 28340
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import to_rgba

from plot_common import (
    OUT_DIR, R_EARTH, TYPE_COLORS,
    apply_occlusion, draw_earth, finish, load_object_types, read_csv_rows,
    run_sim, set_equal_axes, watch_rotation,
)


def load_snapshot(path):
    """Retourne (ids, positions (N,3), vitesses (N,3), temps) pour les objets vivants."""
    rows, t = read_csv_rows(path)
    alive = [r for r in rows if r.get("alive", "1") == "1"]
    dead = len(rows) - len(alive)
    if dead:
        print(f"{dead} objets sous la surface terrestre, exclus du trace.")
    if not alive:
        sys.exit("Aucun objet a tracer.")
    ids = np.array([r["id"] for r in alive])
    pos = np.array([[float(r["x_km"]), float(r["y_km"]), float(r["z_km"])] for r in alive])
    vel = np.array([[float(r["vx"]), float(r["vy"]), float(r["vz"])] for r in alive])
    return ids, pos, vel, t


def inclination_deg(pos, vel):
    """Inclinaison instantanee, depuis le moment cinetique h = r x v."""
    h = np.cross(pos, vel)
    return np.degrees(np.arccos(np.clip(h[:, 2] / np.linalg.norm(h, axis=1), -1, 1)))


def sample(n, total, seed):
    """Indices d'un echantillon reproductible, ou tout si n >= total."""
    if n >= total:
        return np.arange(total)
    return np.random.default_rng(seed).choice(total, size=n, replace=False)


def plot_3d(ax, pos, types, sizes, earth_alpha=1.0, occlude=True, point_alpha=0.75):
    # computed_zorder=False : on impose l'ordre de trace (globe puis points).
    # Les points restants sont tous devant le globe, l'occultation etant
    # geree en amont point par point — voir plot_common.occluded_by_earth.
    ax.computed_zorder = False
    draw_earth(ax, earth_alpha, zorder=0)

    scatters = []
    for name, color in TYPE_COLORS.items():
        m = types == name
        if not m.any():
            continue
        rgba = np.array(to_rgba(color, point_alpha))
        # Pas d'argument `alpha` scalaire ici : mplot3d l'appliquerait a tous
        # les points a chaque trace et ecraserait l'alpha par point.
        scatter = ax.scatter(*pos[m].T, s=sizes, color=rgba, edgecolors="none",
                             depthshade=False, zorder=2, label=f"{name} ({m.sum()})")
        scatters.append((scatter, pos[m], rgba))

    if occlude:
        apply_occlusion(scatters, ax)
        watch_rotation(ax.figure, ax, scatters)

    # Cadrage sur l'ensemble des points, pas seulement les visibles : la vue
    # ne doit pas sauter quand on la fait tourner.
    set_equal_axes(ax, np.abs(pos).max() * 1.05)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")


def plot_alt_inc(ax, alt, inc, types, sizes):
    for name, color in TYPE_COLORS.items():
        m = types == name
        if m.any():
            ax.scatter(alt[m], inc[m], s=sizes, c=color, alpha=0.95,
                       edgecolors="none", label=f"{name} ({m.sum()})")
    ax.set_xscale("log")
    ax.set_xlabel("Altitude (km, echelle log)")
    ax.set_ylabel("Inclinaison (deg)")
    ax.set_ylim(0, 180)
    ax.grid(alpha=0.25, which="both")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("-n", "--count", type=int, default=1000,
                   help="nombre d'objets echantillonnes (defaut 1000)")
    p.add_argument("--view", choices=["3d", "alt-inc"], default="3d",
                   help="type de trace (defaut 3d)")
    p.add_argument("--file", help="CSV a tracer (defaut output/final_state.csv)")
    p.add_argument("--hours", type=float, default=24.0, help="duree simulee (defaut 24)")
    p.add_argument("--dt", type=float, default=10.0, help="pas d'integration en s")
    p.add_argument("--max-alt", type=float, help="n'afficher que sous cette altitude (km)")
    p.add_argument("--seed", type=int, default=0, help="graine de l'echantillonnage")
    p.add_argument("--size", type=float, default=5.0, help="taille des points")
    p.add_argument("--earth-alpha", type=float, default=1.0,
                   help="opacite du globe en vue 3d (defaut 1)")
    p.add_argument("--show-hidden", action="store_true",
                   help="ne pas masquer les objets situes derriere la Terre")
    p.add_argument("--no-run", action="store_true", help="ne pas relancer la simulation")
    p.add_argument("--save", metavar="FICHIER", help="enregistrer au lieu d'afficher")
    args = p.parse_args()

    path = Path(args.file) if args.file else OUT_DIR / "final_state.csv"
    if not args.no_run:
        run_sim(args.hours, args.dt)

    ids, pos, vel, t = load_snapshot(path)
    alt = np.linalg.norm(pos, axis=1) - R_EARTH
    inc = inclination_deg(pos, vel)

    if args.max_alt is not None:
        keep = alt <= args.max_alt
        print(f"{keep.sum()} objets sous {args.max_alt:g} km (sur {len(alt)}).")
        ids, pos, alt, inc = ids[keep], pos[keep], alt[keep], inc[keep]
        if not len(ids):
            sys.exit("Aucun objet dans cette tranche d'altitude.")

    idx = sample(args.count, len(ids), args.seed)
    ids, pos, alt, inc = ids[idx], pos[idx], alt[idx], inc[idx]

    type_map = load_object_types()
    types = np.array([type_map.get(i, "UNKNOWN") or "UNKNOWN" for i in ids])

    fig = plt.figure(figsize=(10, 9) if args.view == "3d" else (11, 6))
    if args.view == "3d":
        ax = fig.add_subplot(projection="3d")
        plot_3d(ax, pos, types, args.size, args.earth_alpha, not args.show_hidden)
    else:
        ax = fig.add_subplot()
        plot_alt_inc(ax, alt, inc, types, args.size)

    when = f"{t / 3600:.1f} h apres l'epoch" if t is not None else "instant inconnu"
    ax.set_title(f"{len(ids)} objets sur {len(type_map)} — {when}")
    ax.legend(loc="upper right", fontsize=8, markerscale=2.5)
    finish(fig, args.save)


if __name__ == "__main__":
    main()
