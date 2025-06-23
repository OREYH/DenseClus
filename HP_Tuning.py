#!/usr/bin/env python3
"""tune_denseclus.py
-------------------------------------------------------
UMAP + HDBSCAN 하이퍼파라미터 탐색 스크립트 (재현성 강화 버전)
 - 10만 개 전수 탐색은 시간이 오래 걸리므로 2만 개 샘플로 튜닝 → YAML 저장
 - tqdm 프로그레스바로 실시간 진행률 출력
 - pathlib 대신 os 모듈만 사용
 - `set_global_seed()` 로 NumPy·random·PYTHONHASHSEED 모두 고정 → 반복 실행 시
   동일한 coverage / DBCV 보장
-------------------------------------------------------
사용 예)
    python tune_denseclus.py \
        --data_path ./data/train.csv \
        --save_yaml ./configs/best_denseclus.yaml \
        --sample 20000 \
        --seed 42
"""

import os
import argparse
import warnings
import random
from itertools import product

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from denseclus import DenseClus

# ────────────────────────────────────────────────────────────────────────────────
# 경고 억제 (sklearn force_all_finite)
# ────────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*force_all_finite.*"
)

# ────────────────────────────────────────────────────────────────────────────────
# 시드 고정 유틸리티 (재현성 확보)
# ────────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int):
    """random / numpy / hash seed 고정"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

# ────────────────────────────────────────────────────────────────────────────────
# 인자 파서
# ────────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="DenseClus 하이퍼파라미터 튜닝 (reproducible)")
    p.add_argument("--data_path", required=True, help="CSV 데이터 경로")
    p.add_argument("--save_yaml", required=True, help="최적 파라미터 YAML 저장 경로")
    p.add_argument("--sample", type=int, default=None, help="튜닝용 샘플 수")
    p.add_argument("--dropna", action="store_true", help="결측 컬럼 제거 여부")
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
# 데이터 로딩 & 샘플링
# ────────────────────────────────────────────────────────────────────────────────

def read_data(path: str, sample: int, dropna: bool, seed: int):
    set_global_seed(seed)  # 샘플링까지 시드 고정
    df = pd.read_csv(path)
    if dropna:
        df = df.dropna(axis=1)
    if sample is not None and sample < len(df):
        df = df.sample(sample, random_state=seed)
    return df

# ────────────────────────────────────────────────────────────────────────────────
# 평가 함수
# ────────────────────────────────────────────────────────────────────────────────

def evaluate(umap_params: dict, hdbscan_params: dict, data: pd.DataFrame, seed: int):
    set_global_seed(seed)
    clf = DenseClus(
        random_state=seed,
        umap_combine_method="intersection_union_mapper",
        umap_params=umap_params,
        hdbscan_params=hdbscan_params,
    )
    clf.fit(data)
    labels = clf.labels_
    coverage = (labels >= 0).mean()  # 군집에 속한 비율
    dbcv = clf.hdbscan_.relative_validity_
    return coverage, dbcv

# ────────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()
    set_global_seed(args.seed)

    # 데이터 준비
    df = read_data(args.data_path, args.sample, args.dropna, args.seed)

    # 하이퍼파라미터 그리드 정의
    umap_grid = [
        {
            "categorical": {"n_neighbors": n_cat, "min_dist": m_cat},
            "numerical":   {"n_neighbors": n_num, "min_dist": m_num},
            "combined":    {"n_neighbors": n_com, "min_dist": m_com},
        }
        for n_cat, m_cat, n_num, m_num, n_com, m_com in product(
            [15, 30],
            [0.0, 0.2],
            [20, 40],
            [0.0, 0.2],
            [5, 10],
            [0.0, 0.2],
        )
    ]

    hdbscan_grid = [
        {"min_samples": ms, "min_cluster_size": mcs, "gen_min_span_tree": True}
        for ms, mcs in product([10, 30, 50], [100, 200, 300])
    ]

    total_iter = len(umap_grid) * len(hdbscan_grid)
    pbar = tqdm(total=total_iter, desc="Grid Search", ncols=110, colour="cyan")

    best_score = float("-inf")
    best_params = None

    for u_params in umap_grid:
        for h_params in hdbscan_grid:
            coverage, dbcv = evaluate(u_params, h_params, df, args.seed)
            score = coverage * dbcv

            if score > best_score:
                best_score = score
                best_params = {"umap_params": u_params, "hdbscan_params": h_params}

                # 즉시 YAML 저장
                os.makedirs(os.path.dirname(args.save_yaml), exist_ok=True)
                with open(args.save_yaml, "w", encoding="utf-8") as f:
                    yaml.dump(best_params, f, sort_keys=False, allow_unicode=True, indent=2)

                pbar.write(
                    f"\n📈 New best → score={best_score:.4f} | cov={coverage:.3f}, dbcv={dbcv:.3f}"
                )

            pbar.update(1)

    pbar.close()

    print("\n✅ 튜닝 완료!")
    print("Best score  :", best_score)
    print("Best params :\n", yaml.safe_dump(best_params, sort_keys=False))
