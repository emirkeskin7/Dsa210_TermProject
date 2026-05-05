"""
F1 Mechanical DNF — data-collection script.

This script is the data-collection step for the term project. It pulls
race results for every Formula 1 race from 2014 through 2025 using
FastF1, enriches each row with the circuit's altitude and circuit type,
labels mechanical DNFs, and writes the final table to
``f1_term_project_data.csv``.

All hypothesis testing, statistical analysis, plotting, and machine
learning live in the companion notebook
``F1_Mechanical_DNF_Analysis.ipynb``, which reads this CSV. To
reproduce the project end-to-end:

    1. Run this script once to (re)build the CSV.
    2. Open the notebook and "Run All".

Re-running this script when new races happen is the only reason to
touch it again — the analysis itself is now in the notebook.
"""

import os
import fastf1
import pandas as pd

# --- 0. CACHE SETUP -----------------------------------------------------
# FastF1 caches the (large) session payloads to disk so subsequent runs
# are fast.
if not os.path.exists("f1_cache"):
    os.makedirs("f1_cache")
fastf1.Cache.enable_cache("f1_cache")


# --- 1. RACE-RESULT COLLECTION ------------------------------------------
all_race_data = []

for year in range(2014, 2026):
    print(f"Fetching season: {year}")
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule["RoundNumber"] > 0]

    for _, event in races.iterrows():
        round_num = event["RoundNumber"]
        circuit_ref = event["Location"].lower().replace(" ", "_")

        try:
            session = fastf1.get_session(year, round_num, "R")
            session.load(laps=False, telemetry=False,
                         weather=False, messages=False)
            results = session.results

            for _, result in results.iterrows():
                all_race_data.append({
                    "Year":        year,
                    "Round":       round_num,
                    "Circuit":     circuit_ref,
                    "Driver":      result["Abbreviation"],
                    "Constructor": result["TeamName"],
                    "Status":      result["Status"],
                })
        except Exception:
            # Some sessions occasionally fail to load (network blips,
            # missing data on the upstream API). Skip and continue —
            # the cache will retry on the next run.
            continue

df = pd.DataFrame(all_race_data)
print(f"Data collection complete. Rows: {len(df):,}")


# --- 2. CIRCUIT-LEVEL ENRICHMENT ---------------------------------------
# Altitude (in metres) and a Street/Permanent classification for every
# circuit visited during the 2014-2025 hybrid era.
circuit_info = [
    {"Circuit": "melbourne",    "Altitude":    5, "Type": "Street"},
    {"Circuit": "kuala_lumpur", "Altitude":   18, "Type": "Permanent"},
    {"Circuit": "shanghai",     "Altitude":    5, "Type": "Permanent"},
    {"Circuit": "sakhir",       "Altitude":    2, "Type": "Permanent"},
    {"Circuit": "barcelona",    "Altitude":  155, "Type": "Permanent"},
    {"Circuit": "monte_carlo",  "Altitude":   69, "Type": "Street"},
    {"Circuit": "montreal",     "Altitude":   13, "Type": "Street"},
    {"Circuit": "spielberg",    "Altitude":  677, "Type": "Permanent"},
    {"Circuit": "silverstone",  "Altitude":  153, "Type": "Permanent"},
    {"Circuit": "hockenheim",   "Altitude":  103, "Type": "Permanent"},
    {"Circuit": "budapest",     "Altitude":  236, "Type": "Permanent"},
    {"Circuit": "spa",          "Altitude":  418, "Type": "Permanent"},
    {"Circuit": "monza",        "Altitude":  162, "Type": "Permanent"},
    {"Circuit": "marina_bay",   "Altitude":   15, "Type": "Street"},
    {"Circuit": "suzuka",       "Altitude":   45, "Type": "Permanent"},
    {"Circuit": "sochi",        "Altitude":    2, "Type": "Street"},
    {"Circuit": "austin",       "Altitude":  161, "Type": "Permanent"},
    {"Circuit": "são_paulo",    "Altitude":  785, "Type": "Permanent"},
    {"Circuit": "abu_dhabi",    "Altitude":   10, "Type": "Permanent"},
    {"Circuit": "mexico_city",  "Altitude": 2285, "Type": "Permanent"},
    {"Circuit": "baku",         "Altitude":  -28, "Type": "Street"},
    {"Circuit": "le_castellet", "Altitude":  432, "Type": "Permanent"},
    {"Circuit": "zandvoort",    "Altitude":   15, "Type": "Permanent"},
    {"Circuit": "imola",        "Altitude":   37, "Type": "Permanent"},
    {"Circuit": "jeddah",       "Altitude":   12, "Type": "Street"},
    {"Circuit": "miami",        "Altitude":    2, "Type": "Street"},
    {"Circuit": "lusail",       "Altitude":   15, "Type": "Permanent"},
    {"Circuit": "las_vegas",    "Altitude":  620, "Type": "Street"},
    {"Circuit": "istanbul",     "Altitude":  130, "Type": "Permanent"},
    {"Circuit": "mugello",      "Altitude":  242, "Type": "Permanent"},
    {"Circuit": "nürburg",      "Altitude":  500, "Type": "Permanent"},
    {"Circuit": "portimão",     "Altitude":   60, "Type": "Permanent"},
]

enrichment_df = pd.DataFrame(circuit_info)
df = df.merge(enrichment_df, on="Circuit", how="left")

# Fill in any unseen circuits with sensible defaults so the analysis
# notebook does not have to deal with NaNs in these two columns.
df["Type"]     = df["Type"].fillna("Permanent")
df["Altitude"] = df["Altitude"].fillna(150)


# --- 3. MECHANICAL-DNF LABELLING ---------------------------------------
MECHANICAL_KEYWORDS = (
    "Engine", "Gearbox", "Electrical", "Hydraulics",
    "Overheating", "Power Unit", "Mechanical",
)


def is_mechanical_dnf(status: str) -> int:
    """Return 1 if the FIA finishing status names a mechanical cause."""
    s = str(status)
    return int(any(kw in s for kw in MECHANICAL_KEYWORDS))


df["is_mechanical_dnf"] = df["Status"].apply(is_mechanical_dnf)

# Driver-error retirements are not reliability events — strip them out.
df = df[~df["Status"].str.contains("Accident|Collision", na=False)]


# --- 4. EXPORT ----------------------------------------------------------
OUT_PATH = "f1_term_project_data.csv"
df.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(df):,} rows to {OUT_PATH}")
print("All analysis (hypothesis tests, plots, ML) lives in the notebook:")
print("    F1_Mechanical_DNF_Analysis.ipynb")
