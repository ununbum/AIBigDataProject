"""
이직 위험 분석 서비스 (Streamlit).

train_model.ipynb에서 정립한 파이프라인을 그대로 재현한다.
  - 전처리: 근속년수 구간화, 직급/학력 매핑, 소속 원-핫, 설문/업무몰입도 피처 합성
  - 탭 1: RandomForest 학습 후 feature_importances_로 Major Feature Top 3 컬럼 출력
  - 탭 2: 결합 LDA(2D) 시각화 위에 Isolation Forest 고위험군(이상치) 표시 + 명단 출력

실행:  streamlit run src/streamlit_service.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# --- train_model.ipynb과 동일한 매핑/피처 정의 ---------------------------------
CL_MAP = {"CL1": 1, "CL2": 2, "CL3": 3, "CL4": 4}
DEGREE_MAP = {"고졸": 1, "학사": 2, "석사": 2, "박사": 3}

TRAIN_FEATURE_SURVEY = ["복지", "회의", "교육", "업무", "상사", "동료"]
GROUP_ITER = ["소속_A그룹", "소속_B그룹", "소속_C그룹", "소속_D그룹",
              "소속_E그룹", "소속_F그룹", "소속_G그룹", "소속_H그룹"]
TRAIN_FEATURE_IDENTITY = ["직급", "학력", "근속년수_구분"] + GROUP_ITER
FEATURES = TRAIN_FEATURE_SURVEY + TRAIN_FEATURE_IDENTITY

TARGET = "이직의사"
SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "survey_data.csv"


def preprocess(df_raw: pd.DataFrame):
    """raw 설문 CSV를 학습용 X(17피처), y(이직의사), 표시용 meta(소속/직급/학력)로 변환."""
    df = df_raw.copy()
    df.fillna(0, inplace=True)

    # 매핑 전 원본 값은 명단 표시에 그대로 쓰기 위해 따로 보관한다.
    meta = pd.DataFrame({
        "소속": df_raw["소속"].astype(str).values,
        "직급": df_raw["직급"].astype(str).values,
        "학력": df_raw["학력"].astype(str).values,
    })

    # 근속년수 → 4구간(신입/주니어/미들/시니어)
    I_fresh = ((df["근속년수"] >= 0) & (df["근속년수"] < 3)).astype(int)
    I_junior = ((df["근속년수"] >= 3) & (df["근속년수"] < 10)).astype(int)
    I_middle = ((df["근속년수"] >= 10) & (df["근속년수"] < 20)).astype(int)
    I_senior = (df["근속년수"] >= 20).astype(int)
    df["근속년수_구분"] = 1 * I_fresh + 2 * I_junior + 3 * I_middle + 4 * I_senior

    df["직급"] = df["직급"].map(CL_MAP)
    df["학력"] = df["학력"].map(DEGREE_MAP)
    df[["직급", "학력"]] = df[["직급", "학력"]].fillna(0)

    df = pd.get_dummies(df, columns=["소속"], drop_first=False)
    # 업로드 데이터에 8개 그룹이 모두 없을 수 있으므로 누락 더미는 False로 채운다.
    for col in GROUP_ITER:
        if col not in df.columns:
            df[col] = False

    df_selection = pd.DataFrame()
    for item in TRAIN_FEATURE_SURVEY:
        df_selection[item] = df[f"조직문화평가_{item}"]
    for item in TRAIN_FEATURE_IDENTITY:
        df_selection[item] = df[item]

    # 설문 문항 일부는 1Q/2Q 업무몰입도평가 합으로 대체(노트북과 동일)
    df_selection["회의"] = df["1Q_업무몰입도평가_회의"] + df["2Q_업무몰입도평가_회의"]
    df_selection["업무"] = (df["1Q_업무몰입도평가_고객"] + df["2Q_업무몰입도평가_고객"]
                          + df["1Q_업무몰입도평가_기술력"] + df["2Q_업무몰입도평가_기술력"])
    df_selection["상사"] = (df["1Q_업무몰입도평가_보고"] + df["2Q_업무몰입도평가_보고"]
                          + df["1Q_업무몰입도평가_소통"] + df["2Q_업무몰입도평가_소통"])

    X = df_selection[FEATURES].astype(float)
    y = df[TARGET] if TARGET in df.columns else None
    return X, y, meta


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


st.set_page_config(page_title="이직 위험 분석 서비스", layout="wide")
st.title("이직 위험 분석 서비스")
st.caption("train_model.ipynb의 RandomForest / LDA / Isolation Forest 파이프라인 기반")

# --- 데이터 입력 --------------------------------------------------------------
uploaded = st.sidebar.file_uploader("설문 CSV 업로드", type=["csv"])
if uploaded is not None:
    df_raw = load_csv(uploaded)
    st.sidebar.success(f"업로드 데이터 사용 ({len(df_raw)}행)")
elif SAMPLE_CSV.exists():
    df_raw = load_csv(str(SAMPLE_CSV))
    st.sidebar.info(f"샘플 데이터 사용: {SAMPLE_CSV.name} ({len(df_raw)}행)")
else:
    st.warning("CSV를 업로드해 주세요.")
    st.stop()

missing = [c for c in ["직급", "소속", "학력", "근속년수"] if c not in df_raw.columns]
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}")
    st.stop()

X, y, meta = preprocess(df_raw)
if y is None:
    st.error(f"타깃 컬럼 '{TARGET}'이(가) 없어 학습을 진행할 수 없습니다.")
    st.stop()

std = StandardScaler()
X_std = std.fit_transform(X)

tab1, tab2 = st.tabs(["① Major Feature (RandomForest)", "② 고위험군 (LDA 2D + Isolation Forest)"])

# --- 탭 1: RandomForest feature importance ------------------------------------
with tab1:
    st.subheader("RandomForest 기반 Major Feature Top 3")

    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(X_std, y)

    importance = (pd.Series(rfc.feature_importances_, index=FEATURES)
                  .sort_values(ascending=False))
    top3 = importance.head(3)

    cols = st.columns(3)
    for rank, (feat, score) in enumerate(top3.items(), start=1):
        cols[rank - 1].metric(f"Major Feature {rank}", feat, f"{score:.3f}")

    st.markdown("**Major Feature 컬럼:** " + ", ".join(f"`{c}`" for c in top3.index))

    st.divider()
    st.markdown("전체 Feature Importance")
    st.bar_chart(importance)
    st.dataframe(
        importance.rename_axis("feature").reset_index(name="importance"),
        width="stretch", hide_index=True,
    )

# --- 탭 2: LDA 2D + Isolation Forest 고위험군 ---------------------------------
with tab2:
    st.subheader("결합 LDA(2D) 위 Isolation Forest 고위험군")

    n_classes = y.nunique()
    if n_classes < 3:
        st.error(f"결합 LDA(2D)에는 3개 이상의 이직의사 등급이 필요합니다 (현재 {n_classes}개).")
        st.stop()

    detect_space = st.sidebar.radio(
        "이상치 탐지 공간",
        ["원본 17차원 (권장)", "LDA 2D"],
        help="노트북은 밀도가 희석되지 않는 원본 17차원에서 이상치를 탐지하도록 논증했다.",
    )
    contamination = st.sidebar.select_slider(
        "Isolation Forest contamination",
        options=["auto", 0.05, 0.1, 0.15, 0.2, 0.3], value="auto",
    )

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_std, y)
    lda_df = pd.DataFrame(X_lda, columns=["lda_comp1", "lda_comp2"])

    iso = IsolationForest(contamination=contamination, random_state=0)
    fit_space = X_std if detect_space.startswith("원본") else X_lda
    iso_predict = iso.fit_predict(fit_space)
    outlier_mask = iso_predict == -1

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(lda_df["lda_comp1"], lda_df["lda_comp2"],
               c="lightgray", alpha=0.5, label="정상")
    ax.scatter(lda_df.loc[outlier_mask, "lda_comp1"],
               lda_df.loc[outlier_mask, "lda_comp2"],
               c="red", marker="X", s=120, edgecolors="black", linewidths=1.0,
               label="고위험군(이상치)")
    ax.set_title("결합 LDA (2D) - Isolation Forest 고위험군")
    ax.set_xlabel("lda_comp1")
    ax.set_ylabel("lda_comp2")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"**고위험군 인원: {int(outlier_mask.sum())}명 / 전체 {len(outlier_mask)}명 "
                f"({outlier_mask.mean():.1%})**")

    roster = meta.loc[outlier_mask, ["소속", "직급", "학력"]].copy()
    roster.insert(0, "이직의사", y[outlier_mask].values)
    roster = roster.sort_values(["이직의사", "소속"], ascending=[False, True])
    st.markdown("고위험군 명단")
    roster_view = roster.reset_index().rename(columns={"index": "행번호"})
    st.dataframe(roster_view, width="stretch", hide_index=True)
    st.download_button("명단 CSV 다운로드",
                       roster.to_csv(index=False).encode("utf-8-sig"),
                       file_name="high_risk_roster.csv", mime="text/csv")
