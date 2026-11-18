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

st.set_page_config(page_title="馬券おみくじ", page_icon="🏇", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #1a0a0a 0%, #2d1515 50%, #1a0a0a 100%);
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ffd700;
    text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
    margin-bottom: 4px;
}
.sub-title {
    text-align: center;
    color: #c9a227;
    font-size: 14px;
    letter-spacing: 8px;
    margin-bottom: 8px;
}
.omikuji-box {
    text-align: center;
    padding: 24px;
    margin: 20px auto;
    max-width: 450px;
    background: linear-gradient(180deg, #8B0000 0%, #5c0000 100%);
    border: 3px solid #ffd700;
    border-radius: 8px;
    box-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
}
.stButton > button {
    background: linear-gradient(180deg, #ffd700 0%, #c9a227 100%) !important;
    color: #1a0a0a !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 12px 40px !important;
}
.horse-card {
    background: linear-gradient(135deg, #2d1515 0%, #1a0a0a 100%);
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    border-left: 4px solid #ffd700;
}
.horse-card-star {
    background: linear-gradient(135deg, #3d2020 0%, #2d1515 100%);
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    border: 2px solid #ffd700;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.4);
}
.umaban { font-size: 28px; font-weight: bold; color: #ff6b6b; }
.umaban-star { font-size: 28px; font-weight: bold; color: #ffd700; }
.horse-name { font-size: 18px; color: #eee; margin-left: 12px; }
.section-title {
    font-size: 18px;
    font-weight: bold;
    color: #ffd700;
    margin: 20px 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #ffd700;
}
.bet-box {
    background: #2d1515;
    border: 1px solid #ffd700;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #eee;
}
.race-info {
    text-align: center;
    padding: 16px;
    background: #2d1515;
    border: 1px solid #ffd700;
    border-radius: 8px;
    margin: 16px 0;
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
            try:
                rank = int(tds[11].get_text(strip=True))
                if 1 <= rank <= 18:
                    ranks.append(rank)
            except:
                pass
        if not ranks:
            return 8.0, 0.08, 0.25
        return round(sum(ranks)/len(ranks), 2), round(sum(1 for r in ranks if r == 1)/len(ranks), 3), round(sum(1 for r in ranks if r <= 3)/len(ranks), 3)
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
            umaban = tds[1].get_text(strip=True) if len(tds) > 1 else ""
            horses.append({
                "horse_name": name_elem.get_text(strip=True),
                "horse_id": horse_id,
                "jockey": jockey_elem.get_text(strip=True) if jockey_elem else "",
                "trainer": trainer_elem.get_text(strip=True) if trainer_elem else "",
                "frame": tds[0].get_text(strip=True) if tds else "0",
                "umaban": umaban,
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
    df["sire_win_rate"] = 0.08
    df["damsire_win_rate"] = 0.08
    df["recent_avg_rank"] = 8.0
    df["recent_win_rate"] = 0.08
    df["recent_top3_rate"] = 0.25
    for idx, row in df.iterrows():
        match = find_match(stats.get("jockey_stats", pd.DataFrame()), "jockey", row["jockey"])
        if match is not None:
            df.loc[idx, "jockey_avg_rank"] = match.get("jockey_avg_rank", 8.0)
            df.loc[idx, "jockey_win_rate"] = match.get("jockey_win_rate", 0.08)
        match = find_match(stats.get("trainer_stats", pd.DataFrame()), "trainer", row["trainer"])
        if match is not None:
            df.loc[idx, "trainer_avg_rank"] = match.get("trainer_avg_rank", 8.0)
            df.loc[idx, "trainer_win_rate"] = match.get("trainer_win_rate", 0.08)
        recent_rank, recent_win, recent_top3 = get_recent_stats(row["horse_id"])
        df.loc[idx, "recent_avg_rank"] = recent_rank
        df.loc[idx, "recent_win_rate"] = recent_win
        df.loc[idx, "recent_top3_rate"] = recent_top3
        time.sleep(0.15)
        sire, damsire = get_pedigree(row["horse_id"])
        time.sleep(0.1)
        match = find_match(stats.get("sire_stats", pd.DataFrame()), "sire", sire)
        if match is not None:
            df.loc[idx, "sire_win_rate"] = match.get("sire_win_rate", 0.08)
        match = find_match(stats.get("damsire_stats", pd.DataFrame()), "damsire", damsire)
        if match is not None:
            df.loc[idx, "damsire_win_rate"] = match.get("damsire_win_rate", 0.08)
    df["place_tendency"] = df["recent_top3_rate"] - df["recent_win_rate"]
    return {"name": race_name, "venue": venue, "surface": surface, "distance": distance, "horses": df}, None

def predict_race(models, race_df, venue, surface):
    score_venue = None
    score_national = None
    if (venue, surface) in models["venue"]:
        try:
            info = models["venue"][(venue, surface)]
            cols = info["cols"]
            for c in cols:
                if c not in race_df.columns:
                    race_df[c] = 0
            X = race_df[cols].fillna(0).values
            score_venue = info["model"].predict(X)
        except:
            pass
    if surface in models["national"]:
        try:
            info = models["national"][surface]
            cols = info["cols"]
            for c in cols:
                if c not in race_df.columns:
                    race_df[c] = 0
            X = race_df[cols].fillna(0).values
            score_national = info["model"].predict(X)
        except:
            pass
    if score_venue is not None and score_national is not None:
        scores = (score_venue + score_national) / 2
        model_type = "アンサンブル"
    elif score_venue is not None:
        scores = score_venue
        model_type = "会場"
    elif score_national is not None:
        scores = score_national
        model_type = "全国"
    else:
        return None, "モデルなし"
    result = race_df.copy()
    result["score"] = scores
    result["軸スコア"] = result["score"] + result["recent_win_rate"] * 10 + result["recent_top3_rate"] * 2.5
    result["軸順位"] = result["軸スコア"].rank(ascending=False).astype(int)
    result["相手順位"] = result["place_tendency"].rank(ascending=False).astype(int)
    return result, model_type

def render_horse_card(umaban, name, is_star=False):
    if is_star:
        st.markdown(f'<div class="horse-card-star"><span class="umaban-star">⭐ {umaban}</span><span class="horse-name">{name}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="horse-card"><span class="umaban">{umaban}</span><span class="horse-name">{name}</span></div>', unsafe_allow_html=True)

st.markdown("""
<div class='omikuji-box'>
    <div class='main-title'>🏇 馬券おみくじ</div>
    <div class='sub-title'>〜 今日の勝ち馬 〜</div>
</div>
""", unsafe_allow_html=True)

try:
    models = load_models()
    stats = load_stats()
    st.sidebar.success("準備完了")
except Exception as e:
    st.error(f"エラー: {e}")
    st.stop()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    race_id = st.text_input("", placeholder="レースID（12桁）", label_visibility="collapsed")
    predict_btn = st.button("🎋 おみくじを引く", type="primary", use_container_width=True)

if predict_btn:
    if len(race_id) != 12:
        st.error("12桁のレースIDを入力してください")
    else:
        with st.spinner("占い中..."):
            info, err = scrape_race(race_id, stats)
        if err:
            st.error(err)
        else:
            label = "芝" if info["surface"] == "turf" else "ダート"
            st.markdown(f"<div class='race-info'><span style='font-size:22px; color:#ffd700;'>{info['name']}</span><br><span style='color:#aaa;'>{info['venue']} / {label} / {info['distance']}m</span></div>", unsafe_allow_html=True)
            result, mtype = predict_race(models, info["horses"], info["venue"], info["surface"])
            if result is not None:
                axis_candidates = result.nsmallest(3, "軸順位")
                partners = result.nlargest(4, "place_tendency")
                partner_names = set(partners["horse_name"])
                axis_names = set(axis_candidates["horse_name"])
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("<div class='section-title'>🎯 本命</div>", unsafe_allow_html=True)
                    for _, row in axis_candidates.iterrows():
                        render_horse_card(row["umaban"], row["horse_name"], row["horse_name"] in partner_names)
                with col_right:
                    st.markdown("<div class='section-title'>🤝 相手</div>", unsafe_allow_html=True)
                    for _, row in partners.iterrows():
                        render_horse_card(row["umaban"], row["horse_name"], row["horse_name"] in axis_names)
                st.markdown("<div class='section-title'>💰 買い目</div>", unsafe_allow_html=True)
                axis_nums = axis_candidates["umaban"].tolist()
                partner_nums = partners["umaban"].tolist()
                overlap = list(set(axis_nums) & set(partner_nums))
                unique_partners = [p for p in partner_nums if p not in axis_nums]
                col_a, col_b = st.columns(2)
                with col_a:
                    if len(overlap) >= 2:
                        box_nums = sorted(overlap)[:4]
                        points = len(box_nums) * (len(box_nums) - 1) // 2
                        st.markdown(f"<div class='bet-box'><b>ワイド BOX</b><br>{' - '.join(map(str, box_nums))}（{points}点）</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='bet-box'><b>ワイド</b><br>軸 {axis_nums[0]} → {unique_partners[:3]}</div>", unsafe_allow_html=True)
                with col_b:
                    if len(overlap) >= 2:
                        box_nums = sorted(overlap)[:4]
                        points = len(box_nums) * (len(box_nums) - 1) // 2
                        st.markdown(f"<div class='bet-box'><b>馬連 BOX</b><br>{' - '.join(map(str, box_nums))}（{points}点）</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='bet-box'><b>馬連</b><br>軸 {axis_nums[0]} → {unique_partners[:3]}</div>", unsafe_allow_html=True)
                with st.expander("全馬一覧"):
                    display = result[["umaban", "horse_name", "jockey", "軸順位", "相手順位"]].sort_values("軸順位")
                    display.columns = ["馬番", "馬名", "騎手", "軸順位", "相手順位"]
                    st.dataframe(display, hide_index=True, use_container_width=True)
            else:
                st.error(mtype)
