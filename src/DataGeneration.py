"""
조직문화평가 및 업무몰입도평가 Raw 설문 데이터 생성기.

README.md의 프로젝트 제안서에 정의된 Feature 스펙을 기반으로
비식별 처리된 가상의 설문 원본(raw) 데이터를 생성한다.
결측치 처리, 파생 타겟 산출 등 전처리 작업은 별도로 진행하므로
이 스크립트는 raw 응답 데이터 생성에만 집중한다.
"""

import numpy as np
import pandas as pd
from pathlib import Path

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


def generate_dataset(n_samples: int = N_TEAM_MEMBERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    positions = _generate_positions(n_samples, rng)
    groups = _generate_groups(n_samples, rng)
    educations = _generate_education(n_samples, rng)
    tenure_years = _generate_tenure(positions, rng)

    org_culture_columns = [f"{ORG_CULTURE_PREFIX}{item}" for item in ORG_CULTURE_ITEMS]
    org_culture_df = _generate_likert_block(
        org_culture_columns,
        n_samples,
        rng,
        extra_corr={(f"{ORG_CULTURE_PREFIX}상사", f"{ORG_CULTURE_PREFIX}업무"): 0.75},
    )
    org_culture_df = _apply_missing_responses(
        org_culture_df, org_culture_columns, ORG_CULTURE_RESPONSE_RATE, rng
    )

    engagement_blocks = []
    engagement_columns_by_quarter = {}
    for quarter in ENGAGEMENT_QUARTERS:
        prefix = ENGAGEMENT_PREFIX_TEMPLATE.format(quarter=quarter)
        columns = [f"{prefix}{item}" for item in ENGAGEMENT_ITEMS]
        block = _generate_likert_block(columns, n_samples, rng)
        block = _apply_missing_responses(block, columns, ENGAGEMENT_RESPONSE_RATE, rng)
        engagement_blocks.append(block)
        engagement_columns_by_quarter[quarter] = columns

    df = pd.DataFrame({
        "idx": np.arange(n_samples),
        "직급": positions,
        "소속": groups,
        "학력": educations,
        "근속년수": tenure_years,
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
