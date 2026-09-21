#!/usr/bin/env python3
"""
spacetrack_export.py
---------------------
Récupère, pour TOUS les objets catalogués par Space-Track (satellites actifs,
débris, étages de fusée...), le TLE le plus proche d'une date de référence
COMMUNE à tous les objets, propage l'état orbital (SGP4) à cet epoch exact,
et écrit le résultat dans un fichier texte — prêt à servir d'état initial
pour une simulation orbitale.

Prérequis :
    pip install -r requirements.txt

Identifiants Space-Track :
    Copier ".env.example" en ".env" (à la racine du dépôt) et le remplir :
        SPACETRACK_USER=ton_email
        SPACETRACK_PASS=ton_mot_de_passe
    Le script charge automatiquement ce fichier au démarrage.
    Si le .env est absent ou incomplet, les identifiants sont demandés
    de façon interactive (via getpass).
    IMPORTANT : ne jamais commit/partager le fichier .env (il est déjà
    listé dans le .gitignore).

Utilisation :
    python scripts/spacetrack_export.py

    Le fichier produit est écrit dans data/ et sert d'entrée à la
    simulation C++ (voir src/main.cpp). Compter ~20 min de requêtes.

Notes importantes :
- La classe "gp_history" n'est PAS peuplée pour les dates trop récentes
  (délai de traitement de plusieurs jours côté Space-Track).
- Interroger l'historique de TOUT le catalogue (~30 000 objets) sur une
  plage de dates EN UNE SEULE requête fait planter le serveur Space-Track
  (erreur 500 : le tri est trop coûteux avant même la pagination). On
  contourne ça en récupérant d'abord la liste des NORAD_CAT_ID (requête
  légère sur "satcat"), puis en interrogeant "gp_history" PAR LOTS de
  quelques centaines d'objets à la fois.
- Respecte les limites de débit de Space-Track (< 30 requêtes/min) : une
  pause est insérée entre chaque requête.
- Sortie : pour chaque objet, le TLE brut (pour re-propager toi-même dans
  ta simu), la position/vitesse ECI (repère inertiel, ce dont un
  propagateur a besoin), et la position géodésique (lat/lon/alt) pour
  vérification visuelle rapide.
"""

import os
import sys
import time
import getpass
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Racine du dépôt : ce script vit dans scripts/, le .env et data/ sont au-dessus.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")  # charge les variables depuis le .env s'il existe
except ImportError:
    print("Astuce : installe python-dotenv (pip install python-dotenv) pour charger un .env automatiquement.")

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

# Date de référence COMMUNE à tous les objets. Doit être assez ancienne
# pour que gp_history soit peuplé (quelques semaines dans le passé est
# une valeur sûre). Change-la librement — seule l'unicité compte pour toi.
TARGET_DATE = datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc)

# Fenêtre de recherche de TLE AVANT la date cible. Certains objets (GEO,
# par ex.) ne sont mis à jour que toutes les 1-2 semaines : une fenêtre
# trop courte les ferait disparaître du résultat.
WINDOW_DAYS = 21

# Taille des lots de NORAD_CAT_ID envoyés par requête gp_history. Plus
# petit = requêtes plus fiables mais plus nombreuses ; plus grand = risque
# de retomber sur une erreur 500.
BATCH_SIZE = 300

# Taille de page pour la pagination de la requête satcat.
PAGE_SIZE = 1000

# Pause entre deux requêtes (secondes) pour rester sous la limite de
# débit Space-Track (30 req/min max).
REQUEST_DELAY = 2.2

OUTPUT_FILE = str(DATA_DIR / f"satellites_{TARGET_DATE.strftime('%Y%m%d_%H%M')}Z.txt")

SPACETRACK_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
SPACETRACK_LOGOUT_URL = "https://www.space-track.org/ajaxauth/logout"

# Liste légère de tous les objets actuellement en orbite (pas encore décayés).
# NOTE : sur la classe "satcat", le champ s'appelle DECAY (pas DECAY_DATE,
# qui n'existe que sur gp/gp_history). Requête paginée (limit/offset) car
# Space-Track renvoie une erreur 500 sur un résultat complet non paginé
# d'environ 30 000 lignes, même sans filtre de date.
SATCAT_URL = (
    "https://www.space-track.org/basicspacedata/query/class/satcat/"
    "CURRENT/Y/"
    "DECAY/null-val/"
    "orderby/NORAD_CAT_ID/"
    "limit/{limit},{offset}/"
    "format/json"
)

# gp_history pour un lot précis de NORAD_CAT_ID, sur la fenêtre de dates.
GP_HISTORY_BATCH_URL = (
    "https://www.space-track.org/basicspacedata/query/class/gp_history/"
    "NORAD_CAT_ID/{ids}/"
    "EPOCH/{start}--{end}/"
    "orderby/NORAD_CAT_ID,EPOCH%20desc/"
    "format/json"
)

# ----------------------------------------------------------------------
# AUTHENTIFICATION
# ----------------------------------------------------------------------

def get_credentials():
    user = os.environ.get("SPACETRACK_USER")
    pwd = os.environ.get("SPACETRACK_PASS")
    if not user:
        user = input("Identifiant Space-Track (email) : ").strip()
    if not pwd:
        pwd = getpass.getpass("Mot de passe Space-Track : ")
    return user, pwd


def login(session: requests.Session, user: str, pwd: str):
    resp = session.post(
        SPACETRACK_LOGIN_URL,
        data={"identity": user, "password": pwd},
    )
    resp.raise_for_status()
    if "Login Failure" in resp.text or resp.status_code != 200:
        raise RuntimeError("Échec de connexion à Space-Track. Vérifie tes identifiants.")
    print("Connecté à Space-Track avec succès.")


def safe_get_json(session: requests.Session, url: str, context: str):
    """GET + parsing JSON avec messages d'erreur clairs (au lieu de crasher)."""
    resp = session.get(url)
    if resp.status_code != 200:
        # On affiche la réponse complète (les erreurs Space-Track contiennent
        # parfois un message utile plus loin dans le HTML/texte renvoyé).
        raise RuntimeError(
            f"Erreur HTTP {resp.status_code} pour {context}.\n"
            f"URL : {url}\n"
            f"Réponse complète :\n{resp.text}"
        )
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(
            f"Réponse non-JSON pour {context}. Extrait : {resp.text[:300]}"
        )
    if isinstance(data, dict):
        raise RuntimeError(f"Erreur renvoyée par Space-Track pour {context} : {data}")
    return data


# ----------------------------------------------------------------------
# ÉTAPE 1 : liste de tous les NORAD_CAT_ID actifs
# ----------------------------------------------------------------------

def fetch_all_norad_ids(session: requests.Session):
    """
    Récupère la liste des objets en orbite ET leurs métadonnées physiques
    (taille via RCS, type d'objet, désignation internationale) depuis satcat.
    Retourne (liste_des_ids, dict norad_id -> métadonnées).
    """
    print("Récupération de la liste complète des objets (satcat)...")
    seen = set()
    all_ids = []
    metadata = {}
    offset = 0
    page_num = 1
    while True:
        url = SATCAT_URL.format(limit=PAGE_SIZE, offset=offset)  # -> limit/PAGE_SIZE,offset
        print(f"  satcat page {page_num} (offset={offset})...")
        data = safe_get_json(session, url, f"satcat page {page_num}")
        if not data:
            break
        new_on_this_page = 0
        for rec in data:
            raw_id = rec.get("NORAD_CAT_ID")
            if raw_id is None:
                continue
            norad_id = str(raw_id).strip()  # type cohérent partout (évite les doublons "25544" vs 25544)
            if norad_id not in seen:
                seen.add(norad_id)
                all_ids.append(norad_id)
                new_on_this_page += 1
                metadata[norad_id] = {
                    "OBJECT_TYPE": rec.get("OBJECT_TYPE", ""),
                    "RCS_SIZE": rec.get("RCS_SIZE", ""),        # categorie : SMALL / MEDIUM / LARGE
                    "RCSVALUE": rec.get("RCSVALUE", ""),        # section radar estimee en m^2, si connue
                    "OBJECT_ID": rec.get("OBJECT_ID", ""),      # designation internationale, ex 1960-009C
                    "COUNTRY": rec.get("COUNTRY", ""),
                }
        if new_on_this_page == 0 and len(data) > 0:
            # Toute la page est déjà connue -> pagination qui boucle, on arrête
            # pour éviter une boucle infinie.
            print("  -> page entièrement redondante avec les précédentes, arrêt de la pagination satcat.")
            break
        if len(data) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        page_num += 1
        time.sleep(REQUEST_DELAY)
    print(f"{len(all_ids)} objets uniques actuellement en orbite (non décayés).")
    return all_ids, metadata


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ----------------------------------------------------------------------
# ÉTAPE 2 : gp_history par lots de NORAD_CAT_ID
# ----------------------------------------------------------------------

def fetch_gp_history_batched(session: requests.Session, norad_ids: list,
                              target: datetime, window_days: int):
    start = (target - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = target.strftime("%Y-%m-%d")

    batches = list(chunk_list(norad_ids, BATCH_SIZE))
    all_records = []

    for i, batch in enumerate(batches, start=1):
        ids_str = ",".join(str(x) for x in batch)
        url = GP_HISTORY_BATCH_URL.format(ids=ids_str, start=start, end=end)
        print(f"Lot {i}/{len(batches)} ({len(batch)} objets)...")
        try:
            data = safe_get_json(session, url, f"lot {i}")
        except RuntimeError as e:
            # Un lot en échec ne doit pas faire planter tout le run :
            # on log et on continue avec les autres lots.
            print(f"  -> ÉCHEC du lot {i}, on continue : {e}")
            time.sleep(REQUEST_DELAY)
            continue

        all_records.extend(data)
        print(f"  -> {len(data)} enregistrements (total cumulé : {len(all_records)})")
        time.sleep(REQUEST_DELAY)  # respect du rate limit Space-Track

    print(f"Total : {len(all_records)} enregistrements TLE reçus (toutes mises à jour confondues).")
    return all_records


def keep_latest_before_target(records: list, target: datetime):
    """
    Pour chaque NORAD_CAT_ID, garde le TLE dont l'EPOCH est <= target
    et le plus proche possible de target (le premier rencontré, car
    la requête est triée EPOCH desc).
    """
    latest = {}
    skipped_future, skipped_bad = 0, 0
    for rec in records:
        raw_id = rec.get("NORAD_CAT_ID")
        if raw_id is None:
            skipped_bad += 1
            continue
        norad_id = str(raw_id).strip()  # meme normalisation que dans fetch_all_norad_ids
        try:
            epoch = datetime.strptime(
                rec["EPOCH"].split(".")[0], "%Y-%m-%dT%H:%M:%S"
            ).replace(tzinfo=timezone.utc)
        except (ValueError, KeyError):
            skipped_bad += 1
            continue
        if epoch > target:
            skipped_future += 1
            continue
        if norad_id not in latest:
            latest[norad_id] = rec  # premier rencontré = le plus récent <= target

    print(f"{len(latest)} objets uniques avec un TLE valide avant la date cible "
          f"({skipped_future} ignorés car postérieurs, {skipped_bad} malformés).")
    return latest


# ----------------------------------------------------------------------
# PROPAGATION SGP4 -> état orbital complet à la date cible
# ----------------------------------------------------------------------

def propagate_all(latest: dict, metadata: dict, target: datetime, output_path: str):
    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale()
    t = ts.from_datetime(target)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Etat orbital de tous les objets Space-Track a un epoch commun\n")
        f.write(f"# Date de reference (epoch commun) : {target.isoformat()}\n")
        f.write(f"# Fenetre de recherche TLE : {WINDOW_DAYS} jours avant la date cible\n")
        f.write(
            "# RCS_SIZE = categorie de taille (SMALL/MEDIUM/LARGE) basee sur la "
            "section radar equivalente. RCS_VALUE_m2 = estimation numerique en m^2 "
            "quand disponible (vide sinon). Ce n'est PAS une dimension physique "
            "(longueur/largeur), mais la meilleure approximation de taille fournie "
            "publiquement par Space-Track.\n"
        )
        f.write(
            "# Colonnes : NORAD_ID | NOM | OBJECT_TYPE | INTL_DESIGNATOR | COUNTRY | "
            "RCS_SIZE | RCS_VALUE_m2 | "
            "TLE_SOURCE_EPOCH (date d'emission du TLE utilise, INFORMATIF SEULEMENT) | "
            "LAT_deg | LON_deg | ALT_km | "
            "X_km | Y_km | Z_km | VX_km_s | VY_km_s | VZ_km_s | "
            "(-- toutes les valeurs ci-dessus sont calculees au MEME instant pour "
            "TOUS les objets, a savoir la date de reference ci-dessus --) | "
            "INCLINAISON_deg | PERIODE_min | TLE_LINE1 | TLE_LINE2\n"
        )
        f.write("#" + "-" * 100 + "\n")

        ok, failed = 0, 0
        for norad_id, rec in latest.items():
            name = rec.get("OBJECT_NAME", "UNKNOWN").strip()
            tle_line1 = rec.get("TLE_LINE1")
            tle_line2 = rec.get("TLE_LINE2")
            epoch_str = rec.get("EPOCH", "")

            meta = metadata.get(norad_id, {})
            object_type = meta.get("OBJECT_TYPE", "") or ""
            intl_designator = meta.get("OBJECT_ID", "") or ""
            country = meta.get("COUNTRY", "") or ""
            rcs_size = meta.get("RCS_SIZE", "") or ""
            rcs_value = meta.get("RCSVALUE", "")
            rcs_value_str = str(rcs_value) if rcs_value not in (None, "") else ""

            if not tle_line1 or not tle_line2:
                failed += 1
                continue

            try:
                sat = EarthSatellite(tle_line1, tle_line2, name, ts)
                geocentric = sat.at(t)

                # Position geodesique (verif visuelle)
                subpoint = wgs84.subpoint(geocentric)
                lat = subpoint.latitude.degrees
                lon = subpoint.longitude.degrees
                alt_km = subpoint.elevation.km

                # Vecteur d'etat ECI (ce dont un propagateur numerique a besoin)
                x, y, z = geocentric.position.km
                vx, vy, vz = geocentric.velocity.km_per_s

                incl = float(rec.get("INCLINATION", "nan"))
                period = float(rec.get("PERIOD", "nan"))

                f.write(
                    f"{norad_id} | {name} | {object_type} | {intl_designator} | {country} | "
                    f"{rcs_size} | {rcs_value_str} | "
                    f"{epoch_str} | "
                    f"{lat:.4f} | {lon:.4f} | {alt_km:.2f} | "
                    f"{x:.3f} | {y:.3f} | {z:.3f} | "
                    f"{vx:.5f} | {vy:.5f} | {vz:.5f} | "
                    f"{incl:.2f} | {period:.2f} | "
                    f"{tle_line1} | {tle_line2}\n"
                )
                ok += 1
            except Exception:
                failed += 1
                continue

        f.write("#" + "-" * 100 + "\n")
        f.write(f"# Total propages avec succes : {ok}\n")
        f.write(f"# Total echecs (TLE invalide/decaye) : {failed}\n")

    print(f"Terminé. {ok} objets écrits dans {output_path} ({failed} échecs).")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    user, pwd = get_credentials()
    session = requests.Session()
    try:
        login(session, user, pwd)
    except Exception as e:
        print(f"Erreur de connexion : {e}", file=sys.stderr)
        sys.exit(1)

    try:
        norad_ids, metadata = fetch_all_norad_ids(session)
        time.sleep(REQUEST_DELAY)
        records = fetch_gp_history_batched(session, norad_ids, TARGET_DATE, WINDOW_DAYS)
    except Exception as e:
        print(f"Erreur lors de la requête : {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Toujours se déconnecter proprement
        session.get(SPACETRACK_LOGOUT_URL)

    if not records:
        print("Aucun enregistrement reçu. Vérifie la fenêtre de dates ou tes droits d'accès.")
        sys.exit(1)

    latest = keep_latest_before_target(records, TARGET_DATE)
    if not latest:
        print("Aucun objet avec un TLE valide avant la date cible. Augmente WINDOW_DAYS.")
        sys.exit(1)

    propagate_all(latest, metadata, TARGET_DATE, OUTPUT_FILE)


if __name__ == "__main__":
    main()