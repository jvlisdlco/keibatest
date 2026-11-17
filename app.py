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
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

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
        
        if filename.startswith("model_national_"):
            surface = filename.replace("model_national_", "").replace(".txt", "")
            cols = turf_cols if surface == "turf" else dirt_cols
            models["national"][surface] = {"model": model, "cols": cols}
        elif filename.startswith("model_"):
            parts = filename.replace("model_", "").replace(".txt", "").rsplit("_", 1)
            if len(parts) == 2:
                venue, surface = parts
                cols = turf_cols if surface == "turf" else dirt_cols
                models["venue"][(venue, surface)] = {"model": model, "cols": cols}
    
    return models

@st.cache_data
def load_stats():
    stats = {}
    for name in ["jockey_stats", "trainer_stats", "sire_stats", "damsire_stats"]:
        path = f"{PIPELINE_DIR}/{name}.csv"
        if os.path.exists(path):
            stats[name] = pd.read_csv(path)
        else:
            stats[name] = pd.DataFrame()
    return stats

def get_pedigree(horse_id):
    """馬の父・母父を取得"""
    url = f"https://db.netkeiba.com/horse/ped/{horse_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.encoding = "euc-jp"
        soup = BeautifulSoup(resp.text, "html.parser")
        blood = soup.select_one(".blood_table")
        if blood:
            links = blood.select("a")
            sire = links[0].get_text(strip=True) if len(links) > 0 else ""
            damsire = links[4].get_text(strip=True) if len(links) > 4 else ""
            return sire, damsire
    except:
        pass
    return "", ""

def scrape_race(race_id, stats):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = resp.apparent_encoding or "euc-jp"
        soup = BeautifulSoup(resp.text, "html.parser")
        
        race_name = soup.select_one(".RaceName")
        race_name = race_name.get_text(strip=True) if race_name else "不明"
        
        race_data = soup.select_one(".RaceData01")
        race_text = race_data.get_text() if race_data else ""
        
        surface = "dirt" if "ダ" in race_text else "turf"
        dist = re.search(r"(\d+)m", race_text)
        distance = int(dist.group(1)) if dist else 1600
        
        venues = {"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京",
                  "06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}
        venue = venues.get(race_id[4:6], "不明")
        
        horses = []
        for row in soup.select("tr.HorseList"):
            name_elem = row.select_one(".HorseName a")
            jockey_elem = row.select_one(".Jockey a")
            trainer_elem = row.select_one(".Trainer a")
            tds = row.select("td")
            
            if name_elem:
                href = name_elem.get("href", "")
                horse_id = href.split("/")[-2] if "/horse/" in href else ""
                
                horses.append({
                    "horse_name": name_elem.get_text(strip=True),
                    "horse_id": horse_id,
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
        
        # 血統取得
        sires, damsires = [], []
        for hid in df["horse_id"]:
            s, d = get_pedigree(hid) if hid else ("", "")
            sires.append(s)
            damsires.append(d)
            time.sleep(0.3)
        df["sire"] = sires
        df["damsire"] = damsires
        
        # 成績マージ
        if len(stats["jockey_stats"]) > 0:
            df = df.merge(stats["jockey_stats"], on="jockey", how="left")
        if len(stats["trainer_stats"]) > 0:
            df = df.merge(stats["trainer_stats"], on="trainer", how="left")
        if len(stats["sire_stats"]) > 0:
            df = df.merge(stats["sire_stats"], on="sire", how="left")
        if len(stats["damsire_stats"]) > 0:
            df = df.merge(stats["damsire_stats"], on="damsire", how="left")
        
        # 欠損埋め
        for col, default in [("jockey_avg_rank", 8.0), ("jockey_win_rate", 0.08),
                             ("trainer_avg_rank", 8.0), ("trainer_win_rate", 0.08),
                             ("sire_win_rate", 0.08), ("damsire_win_rate", 0.08)]:
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)
        
        # 追加特徴量（簡易）
        df["avg_agari"] = 37.0
        df["avg_last_corner"] = 7.0
        df["sire_venue_win"] = df["sire_win_rate"]
        df["sire_dist_win"] = df["sire_win_rate"]
        df["damsire_venue_win"] = df["damsire_win_rate"]
        df["damsire_dist_win"] = df["damsire_win_rate"]
        df["venue_win_rate"] = 0.08
        df["track_win_rate"] = 0.08
        df["dist_win_rate"] = 0.08
        
        return {"name": race_name, "venue": venue, "surface": surface,
                "distance": distance, "horses": df}, None
    except Exception as e:
        return None, str(e)

def predict_race(models, race_df, venue, surface):
    if (venue, surface) in models["venue"]:
        info = models["venue"][(venue, surface)]
        model_type = "会場"
    elif surface in models["national"]:
        info = models["national"][surface]
        model_type = "全国"
    else:
        return None, "モデルなし"
    
    model = info["model"]
    cols = info["cols"]
    
    for c in cols:
        if c not in race_df.columns:
            race_df[c] = 0
    
    X = race_df[cols].fillna(0).values
    scores = model.predict(X)
    
    result = race_df[["horse_name", "jockey", "umaban"]].copy()
    result["score"] = scores
    result["rank"] = result["score"].rank(ascending=False).astype(int)
    return result.sort_values("rank"), model_type

# UI
st.set_page_config(page_title="競馬予想", page_icon="🏇")
st.title("競馬予想")

try:
    models = load_models()
    stats = load_stats()
    st.sidebar.success(f"モデル読込完了")
except Exception as e:
    st.error(f"エラー: {e}")
    st.stop()

race_id = st.text_input("レースID（12桁）", placeholder="202406010101")

if st.button("予想する", type="primary"):
    if len(race_id) != 12:
        st.error("12桁で入力")
    else:
        with st.spinner("データ取得中（血統含む）..."):
            info, err = scrape_race(race_id, stats)
        
        if err:
            st.error(err)
        else:
            label = "芝" if info["surface"] == "turf" else "ダート"
            st.info(f"**{info['name']}** / {info['venue']} {label} {info['distance']}m")
            
            result, mtype = predict_race(models, info["horses"], info["venue"], info["surface"])
            
            if result is not None:
                st.subheader(f"予想結果（{mtype}モデル）")
                for _, row in result.iterrows():
                    r = int(row["rank"])
                    text = f"{r}位: {row['umaban']}番 {row['horse_name']}（{row['jockey']}）"
                    if r <= 3:
                        st.markdown(f"### {text}")
                    else:
                        st.markdown(text)
            else:
                st.error(mtype)
