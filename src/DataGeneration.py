"""
조직문화 만족도 설문조사 가상 데이터 생성기.

README.md의 프로젝트 제안서에 정의된 Feature/Target 스펙을 기반으로
비식별 처리된 가상의 설문 데이터를 생성한다.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42

POSITIONS = ["CL1", "CL2", "CL3", "CL4"]
POSITION_PROB = [0.35, 0.35, 0.20, 0.10]

GROUPS = [f"{c}그룹" for c in "ABCDEFGH"]

EDUCATIONS = ["고졸", "학사", "석사", "박사"]
EDUCATION_PROB = [0.15, 0.45, 0.35, 0.05]

TENURE_BINS = ["3년<", "[3,10)", "[10,20)", "20>"]

# 8개 만족도 항목 (1~5점 리커트 척도)
SATISFACTION_ITEMS = ["복지", "회의", "교육", "업무", "상사", "부서", "C레벨", "동료"]

# 최종 타겟(전반적 회사 만족도) 산출 시 각 항목의 가중치
TARGET_WEIGHTS = {
    "복지": 0.10,
    "회의": 0.08,
    "교육": 0.10,
    "업무": 0.15,
    "상사": 0.20,
    "부서": 0.15,
    "C레벨": 0.07,
    "동료": 0.15,
}


def _generate_positions(n, rng):
    return rng.choice(POSITIONS, size=n, p=POSITION_PROB)


def _generate_groups(n, rng):
    return rng.choice(GROUPS, size=n)


def _generate_education(n, rng):
    return rng.choice(EDUCATIONS, size=n, p=EDUCATION_PROB)


def _generate_tenure(positions, rng):
    """직급이 높을수록 근속연수가 길어지도록 상관관계를 부여하고 4구간 지시함수로 변환."""
    position_base_years = {"CL1": 2, "CL2": 6, "CL3": 12, "CL4": 20}
    base = np.array([position_base_years[p] for p in positions])
    years = np.clip(rng.normal(loc=base, scale=4), 0, 35)

    bins = pd.cut(
        years,
        bins=[-np.inf, 3, 10, 20, np.inf],
        labels=TENURE_BINS,
        right=False,
    )
    dummies = pd.get_dummies(bins).reindex(columns=TENURE_BINS, fill_value=0).astype(int)
    return dummies


def _generate_satisfaction_scores(n, rng):
    """
    8개 만족도 항목을 공통 잠재요인(전반적 조직문화 인식) + 항목별 노이즈로 생성.
    상사 만족도와 업무(평가) 만족도는 추가 상관관계를 부여하여
    README에 언급된 다중공선성 이슈를 재현한다.
    """
    n_items = len(SATISFACTION_ITEMS)

    corr = np.full((n_items, n_items), 0.35)
    np.fill_diagonal(corr, 1.0)
    idx_boss = SATISFACTION_ITEMS.index("상사")
    idx_work = SATISFACTION_ITEMS.index("업무")
    corr[idx_boss, idx_work] = corr[idx_work, idx_boss] = 0.75

    latent = rng.multivariate_normal(mean=np.zeros(n_items), cov=corr, size=n)
    scores = np.clip(np.round(latent * 1.1 + 3.3), 1, 5).astype(int)
    return pd.DataFrame(scores, columns=SATISFACTION_ITEMS)


def _generate_target(satisfaction_df, rng):
    """8개 항목의 가중합 + 노이즈로 전반적 만족도 점수를 산출하고 3단계로 구간화."""
    weights = np.array([TARGET_WEIGHTS[c] for c in satisfaction_df.columns])
    weighted_score = satisfaction_df.to_numpy() @ weights
    noise = rng.normal(0, 0.3, size=len(satisfaction_df))
    overall_score = np.clip(np.round(weighted_score + noise), 1, 5).astype(int)

    label = pd.cut(
        overall_score,
        bins=[0, 2, 3, 5],
        labels=["불만족", "보통", "만족"],
    )
    return overall_score, label


def generate_dataset(n_samples: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    positions = _generate_positions(n_samples, rng)
    groups = _generate_groups(n_samples, rng)
    educations = _generate_education(n_samples, rng)
    tenure_dummies = _generate_tenure(positions, rng)
    satisfaction_df = _generate_satisfaction_scores(n_samples, rng)
    overall_score, satisfaction_label = _generate_target(satisfaction_df, rng)

    df = pd.DataFrame({
        "idx": np.arange(n_samples),
        "직급": positions,
        "소속": groups,
        "학력": educations,
    })
    df = pd.concat([df, tenure_dummies, satisfaction_df], axis=1)
    df["전반적만족도점수"] = overall_score
    df["만족도"] = satisfaction_label

    return df


def main():
    df = generate_dataset(n_samples=1000, seed=RANDOM_SEED)

    output_path = Path(__file__).resolve().parent.parent / "data" / "survey_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Generated {len(df)} rows -> {output_path}")
    print(df.head(10))


if __name__ == "__main__":
    main()
