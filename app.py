import streamlit as st
import pandas as pd
import json
import os
import requests
from bs4 import BeautifulSoup
import re
import lightgbm as lgb
import time

PIPELINE_DIR = "pipeline"
HEADERS = {"User-Agent": "Mozilla/5.0"}

st.set_page_config(page_title="競馬予想", page_icon="🏇", layout="wide")

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    background: linear-gradient(90deg, #e94560, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.sub-title {
    text-align: center;
    color: #888;
    font-size: 16px;
    margin-bottom: 24px;
}
.stButton > button {
    background: linear-gradient(90deg, #e94560, #ff6b6b) !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 40px !important;
    box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4) !important;
}
.horse-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    border-left: 4px solid #e94560;
}
.horse-card-star {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    border-left: 4px solid #ffd700;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.3);
}
.umaban {
    font-size: 28px;
    font-weight: bold;
    color: #e94560;
}
.umaban-star {
    font-size: 28px;
    font-weight: bold;
    color: #ffd700;
}
.horse-name {
    font-size: 18px;
    color: #eee;
    margin-left: 12px;
}
.section-title {
    font-size: 20px;
    font-weight: bold;
    color: #e94560;
    margin: 20px 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #e94560;
}
.bet-box {
    background: #0f3460;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #eee;
}
</style>
""", unsafe_allow_html=True)

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
        try:
            model = lgb.Booster(model_file=path)
            if "national" in filename:
                surface = "turf" if "turf" in filename else "dirt"
                cols = turf_cols if surface == "turf" else dirt_cols
                models["national"][surface] = {"model": model, "cols": cols}
            else:
                parts = filename.replace("model_", "").replace(".txt", "").replace("_v3", "").rsplit("_", 1)
                if len(parts) == 2:
                    venue, surface = parts
                    cols = turf_cols if surface == "turf" else dirt_cols
                    models["venue"][(venue, surface)] = {"model": model, "cols": cols}
        except:
            pass
    return models

@st.cache_data
def load_stats():
    stats = {}
    for name in ["jockey_stats", "trainer_stats", "sire_stats", "damsire_stats"]:
        path = f"{PIPELINE_DIR}/{name}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace("\xa0", " ").str.strip()
            stats[name] = df
        else:
            stats[name] = pd.DataFrame()
    return stats

def get_recent_stats(horse_id):
    if not horse_id:
        return 8.0, 0.08, 0.25
    url = f"https://db.netkeiba.com/horse/result/{horse_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.encoding = "euc-jp"
        soup = BeautifulSoup(resp.content, "html.parser", from_encoding="euc-jp")
        table = soup.select_one("table.db_h_race_results")
        if not table:
            return 8.0, 0.08, 0.25
        ranks = []
        for row in table.select("tr")[1:6]:
            tds = row.select("td")
            if len(tds) < 12:
                continue
            rank_text = tds[11].get_text(strip=True)
            try:
                rank = int(rank_text)
                if 1 <= rank <= 18:
                    ranks.append(rank)
            except:
                pass
        if not ranks:
            return 8.0, 0.08, 0.25
        avg_rank = sum(ranks) / len(ranks)
        win_rate = sum(1 for r in ranks if r == 1) / len(ranks)
        top3_rate = sum(1 for r in ranks if r <= 3) / len(ranks)
        return round(avg_rank, 2), round(win_rate, 3), round(top3_rate, 3)
    except:
        return 8.0, 0.08, 0.25

def get_pedigree(horse_id):
    if not horse_id:
        return "", ""
    url = f"https://db.netkeiba.com/horse/ped/{horse_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.encoding = "euc-jp"
        soup = BeautifulSoup(resp.content, "html.parser", from_encoding="euc-jp")
        blood = soup.select_one(".blood_table")
        if blood:
            links = blood.select("a")
            sire = links[0].get_text(strip=True) if len(links) > 0 else ""
            damsire = links[4].get_text(strip=True) if len(links) > 4 else ""
            return sire.replace("\xa0", " ").strip(), damsire.replace("\xa0", " ").strip()
    except:
        pass
    return "", ""

def find_match(df_stats, col_name, search_name):
    if len(df_stats) == 0 or not search_name:
        return None
    search_clean = str(search_name).replace("\xa0", " ").replace(" ", "").strip()
    for idx, row in df_stats.iterrows():
        stat_name = str(row[col_name]).replace("\xa0", " ").replace(" ", "").strip()
        if search_clean in stat_name or stat_name in search_clean:
            return row
    return None

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
            href = name_elem.get("href", "")
            horse_id = ""
            if "/horse/" in href:
                parts = href.split("/horse/")
                if len(parts) > 1:
                    horse_id = parts[1].split("/")[0]
            horses.append({
                "horse_name": name_elem.get_text(strip=True),
                "horse_id": horse_id,
                "jockey": jockey_elem.get_text(strip=True) if jockey_elem else "",
                "trainer": trainer_elem.get_text(strip=True) if trainer_elem else "",
                "frame": tds[0].get_text(strip=True) if tds else "0",
                "umaban": tds[1].get_text(strip=True
