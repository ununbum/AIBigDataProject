"""
조직문화평가 및 업무몰입도평가 Raw 설문 데이터 생성기.

README.md의 프로젝트 제안서에 정의된 Feature 스펙을 기반으로
비식별 처리된 가상의 설문 원본(raw) 데이터를 생성한다.
결측치 처리 등 전처리 작업은 별도로 진행하므로 이 스크립트는 raw 응답 데이터
생성에 집중하되, Target인 '재직상태'(재직중/변동)는 미응답 처리 전 원점수를
기준으로 함께 생성한다.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

RANDOM_SEED = 42
N_TEAM_MEMBERS = 450

POSITIONS = ["CL1", "CL2", "CL3", "CL4"]
POSITION_PROB = [0.05, 0.25, 0.30, 0.40]

GROUPS = [f"{c}그룹" for c in "ABCDEFGH"]

EDUCATIONS = ["고졸", "학사", "석사", "박사"]
EDUCATION_PROB = [0.05, 0.45, 0.35, 0.15]

# 조직문화평가 항목 (1~5점 리커트 척도)
ORG_CULTURE_PREFIX = "조직문화평가_"
ORG_CULTURE_ITEMS = ["복지", "회의", "교육", "업무", "상사", "부서", "C레벨", "동료"]

# 분기별 업무몰입도평가 항목 (1~5점 리커트 척도)
ENGAGEMENT_QUARTERS = ["1Q", "2Q"]
ENGAGEMENT_PREFIX_TEMPLATE = "{quarter}_업무몰입도평가_"
ENGAGEMENT_ITEMS = ["고객", "회의", "기술력", "소통", "보고"]

# 설문별 응답률. 설문(prefix) 단위로 응답 여부가 결정되며, 응답하지 않으면
# 해당 설문에 속한 문항 전체가 결측(NA) 처리된다.
ORG_CULTURE_RESPONSE_RATE = 0.8
ENGAGEMENT_RESPONSE_RATE = 0.5

# 통합점수(100점 환산 평균)의 전년대비 증감값 분포. 평균을 음수로 두어
# 전반적인 하락 추세를 반영하되, 개인별로는 랜덤하게 오르내리도록 한다.
YOY_CHANGE_MEAN = -3.0
YOY_CHANGE_STD = 6.0

# Target: 이직의사. 조직문화평가/1Q/2Q 업무몰입도평가 점수의 변동성이 크거나
# 지속적으로 낮은 사람일수록 이직 위험도(risk_score)가 높아지도록 생성한다.
# 노이즈를 더해 raw 점수만으로 완전히 역산되지 않도록 한다.
# 사내 분위기를 반영해 소수만 뚜렷한 고위험군으로 갈리도록, 상위 위험도 10%는 5점,
# 다음 20%(누적 30%)는 4점으로 분류하고, 나머지 70%는 정규분포를 가정해
# risk_score의 z-score 표준편차 구간(3분위)에 따라 1~3점으로 배분한다.
TIER5_RATE = 0.10
TIER4_RATE = 0.30
AT_RISK_NOISE_STD = 1.0

# 위험군(상위 TIER4_RATE 비율, 4~5점)은 결정적이진 않지만 설문별로 약 40% 확률로
# 미응답하도록, 응답률 자체를 낮춰서 결측 패턴과 이직의사가 약하게 연동되도록 한다.
AT_RISK_NON_RESPONSE_RATE = 0.4


def _generate_positions(n, rng):
    return rng.choice(POSITIONS, size=n, p=POSITION_PROB)


def _generate_groups(n, rng):
    return rng.choice(GROUPS, size=n)


def _generate_education(n, rng):
    return rng.choice(EDUCATIONS, size=n, p=EDUCATION_PROB)


def _generate_tenure(positions, rng):
    """직급이 높을수록 근속연수가 길어지도록 상관관계를 부여하여 근속년수(정수)를 생성."""
    position_base_years = {"CL1": 2, "CL2": 6, "CL3": 12, "CL4": 20}
    base = np.array([position_base_years[p] for p in positions])
    years = np.clip(rng.normal(loc=base, scale=4), 0, 35)
    return np.round(years).astype(int)


def _generate_likert_block(columns, n, rng, base_corr=0.35, extra_corr=None):
    """
    공통 잠재요인 + 항목별 노이즈로 상관관계를 가진 1~5점 리커트 응답을 생성.
    extra_corr: {(col_a, col_b): corr} 형태로 특정 항목 쌍에 추가 상관관계를 부여할 때 사용.
    (예: README에 언급된 상사-업무 만족도 간 다중공선성 재현)
    """
    n_items = len(columns)
    corr = np.full((n_items, n_items), base_corr)
    np.fill_diagonal(corr, 1.0)

    if extra_corr:
        for (col_a, col_b), value in extra_corr.items():
            i, j = columns.index(col_a), columns.index(col_b)
            corr[i, j] = corr[j, i] = value

    latent = rng.multivariate_normal(mean=np.zeros(n_items), cov=corr, size=n)
    scores = np.clip(np.round(latent * 1.1 + 3.3), 1, 5).astype(int)
    return pd.DataFrame(scores, columns=columns).astype("Int64")


def _apply_missing_responses(df, columns, response_rate, rng):
    """
    설문(prefix) 단위로 응답 여부를 결정한다.
    응답하지 않은 사람은 해당 설문에 속한 문항 전체가 결측(NA) 처리된다.
    response_rate는 스칼라 또는 사람별 응답률 배열(예: 위험군만 낮춘 배열) 모두 가능하다.
    """
    n = len(df)
    responded = rng.random(n) < response_rate
    df.loc[~responded, columns] = pd.NA
    return df


def _to_100_scale(df, columns):
    """1~5점 문항들의 응답자별 평균을 100점 만점으로 환산 (결측 문항은 제외하고 평균)."""
    return df[columns].astype("Float64").mean(axis=1) / 5 * 100


def _generate_yoy_change(n, rng):
    """통합점수(100점 환산 평균)의 전년대비 증감값을 랜덤하게 생성."""
    return np.round(rng.normal(YOY_CHANGE_MEAN, YOY_CHANGE_STD, size=n), 1)


def _generate_employment_status(raw_100_scores, rng, tier5_rate=TIER5_RATE, tier4_rate=TIER4_RATE,
                                 noise_std=AT_RISK_NOISE_STD):
    """
    설문 응답(미응답 처리 전) raw 100점 환산 점수를 기준으로 이직의사를 생성한다.
    점수 변동성(표준편차)이 크거나 평균 점수가 지속적으로 낮을수록 위험도(risk_score)가 높아지며,
    여기에 랜덤 노이즈를 더해 raw 점수만으로 완전히 역산되지 않도록 한다.

    사내 분위기를 반영해 소수만 뚜렷한 고위험군으로 갈리도록 상위 위험도 tier5_rate(10%)는
    5점, 다음 구간(누적 tier4_rate=30%까지)은 4점으로 분류한다. 나머지 다수(70%)는
    정규분포를 가정하여 risk_score를 재표준화한 z-score의 표준편차 3분위 구간에 따라
    1(이직 의사 없음)~3점으로 배분한다.
    """
    def _zscore(s):
        return (s - s.mean()) / s.std(ddof=0)

    volatility = raw_100_scores.std(axis=1, ddof=0)
    mean_score = raw_100_scores.mean(axis=1)

    risk_score = _zscore(volatility) - _zscore(mean_score)
    risk_score = risk_score + rng.normal(0, noise_std, size=len(raw_100_scores))

    tier5_cutoff = risk_score.quantile(1 - tier5_rate)
    tier4_cutoff = risk_score.quantile(1 - tier4_rate)

    is_tier5 = risk_score >= tier5_cutoff
    is_tier4 = (risk_score >= tier4_cutoff) & ~is_tier5
    is_remaining = ~(is_tier5 | is_tier4)

    levels = pd.Series(0, index=risk_score.index, dtype=int)
    levels[is_tier5] = 5
    levels[is_tier4] = 4

    # 정규분포 가정: 나머지 70%를 재표준화한 뒤, 정규분포 3분위 경계(±norm.ppf(2/3))로 1~3점 배분
    remaining_z = _zscore(risk_score[is_remaining])
    band = norm.ppf(2 / 3)

    levels[is_remaining & (remaining_z > band)] = 1
    levels[is_remaining & (remaining_z <= band) & (remaining_z > -band)] = 2
    levels[is_remaining & (remaining_z <= -band)] = 3

    return levels, risk_score


def generate_dataset(n_samples: int = N_TEAM_MEMBERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    positions = _generate_positions(n_samples, rng)
    groups = _generate_groups(n_samples, rng)
    educations = _generate_education(n_samples, rng)
    tenure_years = _generate_tenure(positions, rng)

    org_culture_columns = [f"{ORG_CULTURE_PREFIX}{item}" for item in ORG_CULTURE_ITEMS]
    org_culture_raw = _generate_likert_block(
        org_culture_columns,
        n_samples,
        rng,
        extra_corr={(f"{ORG_CULTURE_PREFIX}상사", f"{ORG_CULTURE_PREFIX}업무"): 0.75},
    )

    engagement_columns_by_quarter = {}
    engagement_raw_blocks = {}
    engagement_raw_100 = {}
    for quarter in ENGAGEMENT_QUARTERS:
        prefix = ENGAGEMENT_PREFIX_TEMPLATE.format(quarter=quarter)
        columns = [f"{prefix}{item}" for item in ENGAGEMENT_ITEMS]
        block_raw = _generate_likert_block(columns, n_samples, rng)
        engagement_columns_by_quarter[quarter] = columns
        engagement_raw_blocks[quarter] = block_raw
        engagement_raw_100[quarter] = _to_100_scale(block_raw, columns)

    # 재직상태(Target)는 미응답 처리 전 raw 점수를 기준으로 산출한다.
    raw_100_scores = pd.DataFrame({
        "조직문화": _to_100_scale(org_culture_raw, org_culture_columns),
        "1Q": engagement_raw_100["1Q"],
        "2Q": engagement_raw_100["2Q"],
    })
    employment_status, risk_score = _generate_employment_status(raw_100_scores, rng)

    # 위험군(상위 TIER4_RATE 비율, 4~5점)은 설문별 응답률을 낮춰(결정적이지 않게) 결측 패턴과 이직의사를 약하게 연동시킨다.
    at_risk_cutoff = risk_score.quantile(1 - TIER4_RATE)
    at_risk_mask = risk_score >= at_risk_cutoff
    at_risk_response_rate = 1 - AT_RISK_NON_RESPONSE_RATE
    org_response_rate = np.where(at_risk_mask, at_risk_response_rate, ORG_CULTURE_RESPONSE_RATE)
    engagement_response_rate = np.where(at_risk_mask, at_risk_response_rate, ENGAGEMENT_RESPONSE_RATE)

    org_culture_df = _apply_missing_responses(
        org_culture_raw.copy(), org_culture_columns, org_response_rate, rng
    )

    engagement_blocks = []
    for quarter in ENGAGEMENT_QUARTERS:
        columns = engagement_columns_by_quarter[quarter]
        block = _apply_missing_responses(
            engagement_raw_blocks[quarter].copy(), columns, engagement_response_rate, rng
        )
        engagement_blocks.append(block)

    df = pd.DataFrame({
        "idx": np.arange(n_samples),
        "직급": positions,
        "소속": groups,
        "학력": educations,
        "근속년수": tenure_years,
        "이직의사": employment_status,
    })
    df = pd.concat([df, org_culture_df, *engagement_blocks], axis=1)

    # 조직문화평가 / 분기별 업무몰입도평가를 100점 만점으로 환산
    hundred_scale_columns = []

    org_culture_100_col = f"{ORG_CULTURE_PREFIX}100점"
    df[org_culture_100_col] = _to_100_scale(df, org_culture_columns)
    hundred_scale_columns.append(org_culture_100_col)

    for quarter, columns in engagement_columns_by_quarter.items():
        prefix = ENGAGEMENT_PREFIX_TEMPLATE.format(quarter=quarter)
        col_100 = f"{prefix}100점"
        df[col_100] = _to_100_scale(df, columns)
        hundred_scale_columns.append(col_100)

    # 3개 100점 환산값의 평균(통합점수)과 전년대비 증감값
    df["통합점수_평균"] = df[hundred_scale_columns].astype("Float64").mean(axis=1).round(1)
    df["통합점수_전년대비증감"] = _generate_yoy_change(n_samples, rng)

    return df


def main():
    df = generate_dataset()

    output_path = Path(__file__).resolve().parent.parent / "data" / "survey_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Generated {len(df)} rows -> {output_path}")
    print(df.head(10))


if __name__ == "__main__":
    main()
