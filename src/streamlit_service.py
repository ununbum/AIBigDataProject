import streamlit as st
import pandas as pd
import joblib

import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

MODEL_RANDOM_STATE = 0
CL_MAP = {"CL1": 1, "CL2": 2, "CL3": 3, "CL4": 4}
DEGREE_MAP = {"고졸": 1, "학사": 2, "석사": 2, "박사": 3}

TRAIN_FEATURE_SURVEY = ["복지", "회의", "교육", "업무", "상사", "동료"]
GROUP_ITER = ["소속_A그룹", "소속_B그룹", "소속_C그룹", "소속_D그룹",
              "소속_E그룹", "소속_F그룹", "소속_G그룹", "소속_H그룹"]
TRAIN_FEATURE_IDENTITY = ["직급", "학력", "근속년수_구분"] + GROUP_ITER
FEATURES = TRAIN_FEATURE_SURVEY + TRAIN_FEATURE_IDENTITY
TARGET = "이직의사"

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def preprocess(df_raw: pd.DataFrame):
    """raw 설문 CSV를 학습용 X(17피처), y(이직의사), 표시용 Data(소속/직급/학력)로 변환."""
    df = df_raw.copy()
    df.fillna(0, inplace=True)

    # 매핑 전 원본 값은 명단 표시에 그대로 쓰기 위해 따로 보관한다.
    table_raw = pd.DataFrame({
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

    df_selection["회의"] = df["1Q_업무몰입도평가_회의"] + df["2Q_업무몰입도평가_회의"]
    df_selection["업무"] = (df["1Q_업무몰입도평가_고객"] + df["2Q_업무몰입도평가_고객"]
                          + df["1Q_업무몰입도평가_기술력"] + df["2Q_업무몰입도평가_기술력"])
    df_selection["상사"] = (df["1Q_업무몰입도평가_보고"] + df["2Q_업무몰입도평가_보고"]
                          + df["1Q_업무몰입도평가_소통"] + df["2Q_업무몰입도평가_소통"])

    X = df_selection[FEATURES]
    y = df[TARGET]

    X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state=MODEL_RANDOM_STATE)
    return X, y, X_tn, X_te, y_tn, y_te, table_raw

st.set_page_config(page_title="조직문화 평가 기반 위험군 분류",layout='wide')
st.title("조직문화 평가 기반 위험군 분류")
st.caption("RandomForest기반의 주요 Feature분석 / LDA를 통한 데이터 시각화 / Isolation Forest 기반의 이상치 탐지")

# --- 데이터 입력 --------------------------------------------------------------


with st.sidebar:
    st.title("↪ 사이드바 메뉴")

    uploaded = st.file_uploader("📂 설문 CSV 업로드", type="csv")
    if uploaded is not None:
        df_raw = load_csv(uploaded)
        st.success("CSV 업로드 완료")
        X, y, X_tn, X_te,y_tn, y_te, table_raw = preprocess(df_raw)

        std = StandardScaler()
        std.fit(X_tn)
        X_tn_std = std.transform(X_tn)
        X_te_std = std.transform(X_te)
        X_std = std.transform(X)        
        
        st.session_state["data"] = "uploaded"
    else:
        st.warning("CSV를 업로드해 주세요.")
        st.stop()

    st.divider()

    if st.button("📝 위험 요소 분석", use_container_width=True):
        st.session_state["page"] = "importance_feature"
    if st.button("👀 이직 고위험군 탐색", use_container_width=True):
        st.session_state["page"] = "visualize"

if "data" in st.session_state:
    if "page" not in st.session_state:
        st.title("CSV를 입력후 메뉴를 선택해주세요")
    elif st.session_state["page"] == "importance_feature":
        st.header("위험 요소 분석")
        st.subheader("RandomForest 기반 이직 결정 요소 Top 3")

        rfc = RandomForestClassifier(random_state=0)
        rfc.fit(X_te_std, y_te)

        importance = (pd.Series(rfc.feature_importances_, index=FEATURES)
                    .sort_values(ascending=False))
        top3 = importance.head(3)

        st.markdown("**Major Feature 컬럼:** " + ", ".join(f"`{c}`" for c in top3.index))

        st.divider()
        st.markdown("전체 Feature Importance")
        st.bar_chart(importance)
        st.dataframe(
            importance.rename_axis("feature").reset_index(name="importance"),
            width="stretch", hide_index=True,
        )
# --- 탭 2: LDA 2D + Isolation Forest 고위험군 ---------------------------------
    elif st.session_state["page"] == "visualize":
        st.header("위험 요소 분석")
        st.subheader("LDA(2D) 공간상의 Isolation Forest 고위험군 시각화")

        estimator = st.sidebar.select_slider(
            "Isolation Forest estimator",
            options=[1, 5, 20, 100, 200, 300], value=100,
        )
        contamination = st.sidebar.select_slider(
            "Isolation Forest contamination",
            options=["auto", 0.05, 0.1, 0.15, 0.2, 0.3], value="auto",
        )

        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(X_te_std, y_te)
        X_lda = lda.transform(X_std)

        lda_df = pd.DataFrame(X_lda, columns=["lda_comp1", "lda_comp2"], index=X.index)
        lda_df['target'] = df_raw.loc[X.index, '이직의사'].values

        iso = IsolationForest(n_estimators=estimator, contamination=contamination, random_state=0)
        iso.fit(X_te_std)
        iso_predict = iso.predict(X_std)



        outlier_mask = iso_predict == -1
        high_risk_mask = lda_df['target'] == 4
        high_risk_outlier_mask = high_risk_mask & outlier_mask

        ext_high_risk_mask = lda_df['target'] == 5
        ext_high_risk_outlier_mask = ext_high_risk_mask & outlier_mask


        fig = plt.figure(figsize=(8, 4))
        plt.scatter(lda_df.loc[high_risk_mask, 'lda_comp1'],
                lda_df.loc[high_risk_mask, 'lda_comp2'],
                marker='^',
                c='gray', label='위험군 4점')
        plt.scatter(lda_df.loc[high_risk_outlier_mask, 'lda_comp1'],
                lda_df.loc[high_risk_outlier_mask, 'lda_comp2'],
                c='red', marker='^', label='위험군 이상치')

        plt.scatter(lda_df.loc[ext_high_risk_mask, 'lda_comp1'],
                lda_df.loc[ext_high_risk_mask, 'lda_comp2'],
                c='gray', marker='X', label='고위험군 5점')
        plt.scatter(lda_df.loc[ext_high_risk_outlier_mask, 'lda_comp1'],
                lda_df.loc[ext_high_risk_outlier_mask, 'lda_comp2'],
                c='red', marker='X', label='고위험군 이상치')
        plt.xlabel("lda_comp1")
        plt.ylabel("lda_comp2")
        plt.legend()

        st.pyplot(fig)

        st.markdown(f"**고위험군 인원: {int(outlier_mask.sum())}명 / 전체 {len(outlier_mask)}명 "
                    f"({outlier_mask.mean():.1%})**")

        table = table_raw.loc[ext_high_risk_mask, ["소속", "직급", "학력"]].copy()
        table.add(table_raw.loc[high_risk_mask, ["소속", "직급", "학력"]].copy())

        st.markdown("고위험군 명단")
        roster_view = table.reset_index().rename(columns={"index": "행번호"})
        st.dataframe(roster_view, width="stretch", hide_index=True)       
