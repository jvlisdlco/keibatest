
import streamlit as st
import pandas as pd
import pickle
import json
import os

PIPELINE_DIR = "pipeline"

@st.cache_resource
def load_predictor():
    with open(f"{PIPELINE_DIR}/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    class Predictor:
        def __init__(self):
            self.venue_models = {}
            self.national_models = {}
            self.strong_venues = config["strong_venues"]
            self.turf_cols = config["turf_cols"]
            self.dirt_cols = config["dirt_cols"]
            
            for key, path in config["models"]["venue_models"].items():
                venue, surface = key.rsplit("_", 1)
                # パスを修正
                filename = os.path.basename(path)
                full_path = f"{PIPELINE_DIR}/{filename}"
                if os.path.exists(full_path):
                    with open(full_path, "rb") as f:
                        model = pickle.load(f)
                    cols = self.turf_cols if surface == "turf" else self.dirt_cols
                    self.venue_models[(venue, surface)] = {"model": model, "cols": cols}
            
            for surface, path in config["models"]["national_models"].items():
                filename = os.path.basename(path)
                full_path = f"{PIPELINE_DIR}/{filename}"
                if os.path.exists(full_path):
                    with open(full_path, "rb") as f:
                        model = pickle.load(f)
                    cols = self.turf_cols if surface == "turf" else self.dirt_cols
                    self.national_models[surface] = {"model": model, "cols": cols}
        
        def predict(self, race_df):
            venue = race_df["venue"].iloc[0]
            surface = race_df["surface"].iloc[0]
            
            if (venue, surface) in self.venue_models:
                info = self.venue_models[(venue, surface)]
                model_type = "会場"
            else:
                info = self.national_models.get(surface)
                model_type = "全国"
            
            if info is None:
                return None, "モデルが見つかりません"
            
            model = info["model"]
            cols = info["cols"]
            
            for c in cols:
                if c not in race_df.columns:
                    race_df[c] = 0
            
            X = race_df[cols].fillna(0)
            scores = model.predict(X)
            
            result = race_df[["horse_name"]].copy()
            result["score"] = scores
            result["rank"] = result["score"].rank(ascending=False).astype(int)
            return result.sort_values("rank"), model_type
    
    return Predictor()

# UI
st.set_page_config(page_title="競馬予想", page_icon="🏇", layout="wide")
st.title("🏇 競馬予想")

try:
    predictor = load_predictor()
    st.sidebar.success("モデル読み込み完了")
except Exception as e:
    st.error(f"モデル読み込みエラー: {e}")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("レース設定")
    venue = st.selectbox("会場", ["東京", "中山", "阪神", "京都", "福島", "新潟", "中京", "小倉", "札幌", "函館"])
    surface = st.radio("サーフェス", ["turf", "dirt"], format_func=lambda x: "芝" if x == "turf" else "ダート")
    
    st.subheader("出走馬")
    horse_input = st.text_area("馬名（1行1頭）", height=300, placeholder="サンプル馬1\nサンプル馬2\nサンプル馬3")
    
    predict_btn = st.button("予想する", type="primary", use_container_width=True)

with col2:
    st.subheader("予想結果")
    
    if predict_btn:
        horses = [h.strip() for h in horse_input.strip().split("\n") if h.strip()]
        
        if len(horses) < 2:
            st.warning("2頭以上入力してください")
        else:
            race_df = pd.DataFrame({
                "horse_name": horses,
                "venue": venue,
                "surface": surface,
            })
            
            result, model_type = predictor.predict(race_df)
            
            if result is not None:
                surface_label = "芝" if surface == "turf" else "ダート"
                st.info(f"**{venue} {surface_label}** / {model_type}モデル")
                
                for _, row in result.iterrows():
                    rank = int(row["rank"])
                    name = row["horse_name"]
                    if rank <= 3:
                        st.markdown(f"### {rank}位: {name}")
                    else:
                        st.markdown(f"{rank}位: {name}")
                
                st.caption("上位3頭に1着が入る目安: 芝≈87% / ダート≈59%")
            else:
                st.error(model_type)
    else:
        st.info("左のフォームに入力して「予想する」を押してください")

st.markdown("---")
st.caption("※ 簡易版です。実際の予想には過去成績等の特徴量が必要です。")
