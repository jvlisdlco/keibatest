import streamlit as st
import pandas as pd
import json
import os
import requests
from bs4 import BeautifulSoup
import re
import lightgbm as lgb

PIPELINE_DIR = "pipeline"

@st.cache_resource
def load_models():
    """モデルを読み込み"""
    models = {"venue": {}, "national": {}}
    
    # config読み込み
    with open(f"{PIPELINE_DIR}/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    turf_cols = config["turf_cols"]
    dirt_cols = config["dirt_cols"]
    
    # .txtファイルを探す
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
            # model_福島_turf.txt
            parts = filename.replace("model_", "").replace(".txt", "").rsplit("_", 1)
            if len(parts) == 2:
                venue, surface = parts
                cols = turf_cols if surface == "turf" else dirt_cols
                models["venue"][(venue, surface)] = {"model": model, "cols": cols}
    
    return models, turf_cols, dirt_cols

def predict_race(models, race_df, venue, surface):
    """予測"""
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
    
    result = race_df[["horse_name", "jockey", "frame"]].copy()
    result["score"] = scores
    result["rank"] = result["score"].rank(ascending=False).astype(int)
    return result.sort_values("rank"), model_type

def scrape_race(race_id):
    """レース情報取得"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
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
    name = row.select_one(".HorseName a")
    jockey = row.select_one(".Jockey a")
    tds = row.select("td")
    if name:
        # 馬番は通常2番目のtd
        umaban = tds[1].get_text(strip=True) if len(tds) > 1 else ""
        wakuban = tds[0].get_text(strip=True) if tds else ""
        horses.append({
            "horse_name": name.get_text(strip=True),
            "jockey": jockey.get_text(strip=True) if jockey else "",
            "frame": wakuban,      # 枠番
            "umaban": umaban,      # 馬番
            "distance": distance
        })
        
        if not horses:
            return None, "出走馬なし"
        
        df = pd.DataFrame(horses)
        df["frame_num"] = pd.to_numeric(df["frame"], errors="coerce").fillna(0)
        df["frame_rank"] = df["frame_num"].rank()
        
        return {"name": race_name, "venue": venue, "surface": surface, 
                "distance": distance, "horses": df}, None
    except Exception as e:
        return None, str(e)

# ===== UI =====
st.set_page_config(page_title="競馬予想", page_icon="🏇")
st.title("🏇 競馬予想")

# モデル読み込み
try:
    models, _, _ = load_models()
    n_venue = len(models["venue"])
    n_national = len(models["national"])
    st.sidebar.success(f"モデル: 会場{n_venue} / 全国{n_national}")
except Exception as e:
    st.error(f"読み込みエラー: {e}")
    st.stop()

# 入力
race_id = st.text_input("レースID（12桁）", placeholder="202406010101")

if st.button("取得して予想", type="primary"):
    if len(race_id) != 12:
        st.error("12桁で入力してください")
    else:
        with st.spinner("取得中..."):
            info, err = scrape_race(race_id)
        
        if err:
            st.error(err)
        else:
            surface_label = "芝" if info["surface"] == "turf" else "ダート"
            st.info(f"**{info['name']}** / {info['venue']} {surface_label} {info['distance']}m")
            
            result, model_type = predict_race(models, info["horses"], info["venue"], info["surface"])
            
            if result is not None:
                st.subheader(f"予想（{model_type}）")
                for _, row in result.iterrows():
                    r = int(row["rank"])
                    if r <= 3:
                        st.markdown(f"### {r}位: {row['horse_name']}")
                    else:
                        st.markdown(f"{r}位: {row['horse_name']}")
            else:
                st.error(model_type)
