#!/usr/bin/env python3
"""
check_duplicates.py
-------------------
Verifie qu'un export Space-Track ne contient pas deux fois le meme NORAD_ID
(un doublon fausserait la simulation : le meme objet serait propage deux fois).

Utilisation :
    python scripts/check_duplicates.py [fichier]
"""

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = ROOT / "data" / "satellites_20260801_1000Z.txt"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    if not path.exists():
        sys.exit(f"{path} introuvable.")

    with open(path, encoding="utf-8") as f:
        ids = [line.split("|")[0].strip() for line in f if not line.startswith("#")]

    dupes = [i for i, n in collections.Counter(ids).items() if n > 1]
    print(f"{len(ids)} objets, {len(set(ids))} NORAD_ID uniques.")
    if dupes:
        print(f"ATTENTION — {len(dupes)} doublons : {dupes[:20]}")
        sys.exit(1)
    print("Aucun doublon.")


if __name__ == "__main__":
    main()
