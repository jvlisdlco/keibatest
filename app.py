import streamlit as st
import pandas as pd
import json
import os
import requests
from bs4 import BeautifulSoup
import re
import lightgbm as lgb

PIPELINE_DIR = "pipeline"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

@st.cache_resource
def load_predictor():
    with open(f"{PIPELINE_DIR}/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    class Predictor:
        def __init__(self):
            self.venue_models = {}
            self.national_models = {}
            self.turf_cols = config["turf_cols"]
            self.dirt_cols = config["dirt_cols"]
            
            # 会場モデル読み込み（.txt形式）
            for key in config["models"]["venue_models"].keys():
                venue, surface = key.rsplit("_", 1)
                filename = f"model_{venue}_{surface}.txt"
                full_path = f"{PIPELINE_DIR}/{filename}"
                if os.path.exists(full_path):
                    model = lgb.Booster(model_file=full_path)
                    cols = self.turf_cols if surface == "turf" else self.dirt_cols
                    self.venue_models[(venue, surface)] = {"model": model, "cols": cols}
            
            # 全国モデル読み込み
            for surface in config["models"]["national_models"].keys():
                filename = f"model_national_{surface}.txt"
                full_path = f"{PIPELINE_DIR}/{filename}"
                if os.path.exists(full_path):
                    model = lgb.Booster(model_file=full_path)
                    cols = self.turf_cols if surface == "turf" else self.dirt_cols
                    self.national_models[surface] = {"model": model, "cols": cols}
        
        def predict(self, race_df, venue, surface):
            if (venue, surface) in self.venue_models:
                info = self.venue_models[(venue, surface)]
                model_type = "会場"
            else:
                info = self.national_models.get(surface)
                model_type = "全国"
            
            if info is None:
                return None, "モデルなし"
            
            model = info["model"]
            cols = info["cols"]
            
            for c in cols:
                if c not in race_df.columns:
                    race_df[c] = 0
            
            X = race_df[cols].fillna(0)
            scores = model.predict(X.values)
            
            result = race_df[["horse_name", "jockey", "frame"]].copy()
            result["score"] = scores
            result["rank"] = result["score"].rank(ascending=False).astype(int)
            return result.sort_values("rank"), model_type
    
    return Predictor()

def scrape_race(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = resp.apparent_encoding or "euc-jp"
        soup = BeautifulSoup(resp.text, "html.parser")
        
        race_name_elem = soup.select_one(".RaceName")
        race_name = race_name_elem.get_text(strip=True) if race_name_elem else "不明"
        
        race_data = soup.select_one(".RaceData01")
        race_data_text = race_data.get_text() if race_data else ""
        
        surface = "dirt" if ("ダート" in race_data_text or "ダ" in race_data_text) else "turf"
        
        dist_match = re.search(r"(\d+)m", race_data_text)
        distance = int(dist_match.group(1)) if dist_match else 1600
        
        venue_map = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
                     "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
        venue = venue_map.get(race_id[4:6], "不明")
        
        horses = []
        for row in soup.select("tr.HorseList"):
            try:
                name_elem = row.select_one(".HorseName a")
                horse_name = name_elem.get_text(strip=True) if name_elem else ""
                jockey_elem = row.select_one(".Jockey a")
                jockey = jockey_elem.get_text(strip=True) if jockey_elem else ""
                tds = row.select("td")
                frame = tds[0].get_text(strip=True) if tds else "0"
                if horse_name:
                    horses.append({"horse_name": horse_name, "jockey": jockey, "frame": frame, "distance": distance})
            except:
                continue
        
        if not horses:
            return None, "出走馬が取得できません"
        
        df = pd.DataFrame(horses)
        df["frame_num"] = pd.to_numeric(df["frame"], errors="coerce").fillna(0)
        df["frame_rank"] = df["frame_num"].rank(method="average")
        
        return {"race_name": race_name, "venue": venue, "surface": surface, "distance": distance, "horses": df}, None
    except Exception as e:
        return None, str(e)

# UI
st.set_page_config(page_title="競馬予想", page_icon="🏇")
st.title("🏇 競馬予想")

try:
    predictor = load_predictor()
    st.sidebar.success("モデル読み込み完了")
except Exception as e:
    st.error(f"モデル読み込みエラー: {e}")
    st.stop()

st.subheader("レースIDを入力")
st.caption("netkeibaのURL末尾12桁")

race_id = st.text_input("レースID", placeholder="202406010101")

if st.button("レース情報を取得", type="primary"):
    if len(race_id) != 12 or not race_id.isdigit():
        st.error("レースIDは12桁の数字です")
    else:
        with st.spinner("取得中..."):
            race_info, error = scrape_race(race_id)
        if error:
            st.error(f"エラー: {error}")
        else:
            st.session_state["race_info"] = race_info
            st.success("取得完了")

if "race_info" in st.session_state:
    info = st.session_state["race_info"]
    surface_label = "芝" if info["surface"] == "turf" else "ダート"
    st.info(f"**{info['race_name']}** / {info['venue']} {surface_label} {info['distance']}m")
    st.dataframe(info["horses"][["horse_name", "jockey", "frame"]], hide_index=True)
    
    if st.button("予想する", type="primary"):
        result, model_type = predictor.predict(info["horses"], info["venue"], info["surface"])
        if result is not None:
            st.subheader(f"予想結果（{model_type}モデル）")
            for _, row in result.iterrows():
                rank = int(row["rank"])
                if rank <= 3:
                    st.markdown(f"### {rank}位: {row['horse_name']}（{row['jockey']}）")
                else:
                    st.markdown(f"{rank}位: {row['horse_name']}（{row['jockey']}）")
        else:
            st.error(model_type)

st.caption("レースIDは netkeiba のURL末尾12桁です")
