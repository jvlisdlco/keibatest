import streamlit as st
import pandas as pd
import json
import os
import requests
from bs4 import BeautifulSoup
import re
import lightgbm as lgb

PIPELINE_DIR = "pipeline"
HEADERS = {"User-Agent": "Mozilla/5.0"}

@st.cache_resource
def load_models():
    models = {"venue": {}, "national": {}}
    with open(f"{PIPELINE_DIR}/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    turf_cols = config["turf_cols"]
    dirt_cols = config["dirt_cols"]
    for filename in os.listdir(PIPELINE_DIR):
        if not filename.endswith(".txt"):
            continue
        path = f"{PIPELINE_DIR}/{filename}"
        model = lgb.Booster(model_file=path)
        if "national" in filename:
            surface = "turf" if "turf" in filename else "dirt"
            cols = turf_cols if surface == "turf" else dirt_cols
            models["national"][surface] = {"model": model, "cols": cols}
        else:
            parts = filename.replace("model_", "").replace(".txt", "").rsplit("_", 1)
            if len(parts) == 2:
                venue, surface = parts
                cols = turf_cols if surface == "turf" else dirt_cols
                models["venue"][(venue, surface)] = {"model": model, "cols": cols}
    return models

@st.cache_data
def load_stats():
    stats = {}
    for name in ["jockey_stats", "trainer_stats"]:
        path = f"{PIPELINE_DIR}/{name}.csv"
        if os.path.exists(path):
            stats[name] = pd.read_csv(path)
        else:
            stats[name] = pd.DataFrame()
    return stats

def scrape_race(race_id, stats):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.encoding = resp.apparent_encoding or "euc-jp"
    soup = BeautifulSoup(resp.text, "html.parser")
    race_name_elem = soup.select_one(".RaceName")
    race_name = race_name_elem.get_text(strip=True) if race_name_elem else "不明"
    race_data = soup.select_one(".RaceData01")
    race_text = race_data.get_text() if race_data else ""
    surface = "dirt" if "ダ" in race_text else "turf"
    dist_match = re.search(r"(\d+)m", race_text)
    distance = int(dist_match.group(1)) if dist_match else 1600
    venue_map = {"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}
    venue = venue_map.get(race_id[4:6], "不明")
    horses = []
    for row in soup.select("tr.HorseList"):
        name_elem = row.select_one(".HorseName a")
        jockey_elem = row.select_one(".Jockey a")
        trainer_elem = row.select_one(".Trainer a")
        tds = row.select("td")
        if name_elem:
            horses.append({
                "horse_name": name_elem.get_text(strip=True),
                "jockey": jockey_elem.get_text(strip=True) if jockey_elem else "",
                "trainer": trainer_elem.get_text(strip=True) if trainer_elem else "",
                "frame": tds[0].get_text(strip=True) if tds else "0",
                "umaban": tds[1].get_text(strip=True) if len(tds) > 1 else "",
                "distance": distance
            })
    if not horses:
        return None, "出走馬なし"
    df = pd.DataFrame(horses)
    df["frame_num"] = pd.to_numeric(df["frame"], errors="coerce").fillna(0)
    df["frame_rank"] = df["frame_num"].rank()
    df["jockey_avg_rank"] = 8.0
    df["jockey_win_rate"] = 0.08
    df["trainer_avg_rank"] = 8.0
    df["trainer_win_rate"] = 0.08
    js = stats.get("jockey_stats", pd.DataFrame())
    if len(js) > 0:
        for idx, row in df.iterrows():
            jname = str(row["jockey"])
            if jname:
                match = js[js["jockey"].astype(str).str.contains(jname, na=False, regex=False)]
                if len(match) > 0:
                    df.loc[idx, "jockey_avg_rank"] = match.iloc[0]["jockey_avg_rank"]
                    df.loc[idx, "jockey_win_rate"] = match.iloc[0]["
