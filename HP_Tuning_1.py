#!/usr/bin/env python3
"""tune_denseclus.py
-------------------------------------------------------
UMAP + HDBSCAN 하이퍼파라미터 탐색 스크립트 (재현성 강화 버전)
 - YAML 저장
 - 결과 로그는 results 폴더에, YAML 파일은 yaml 폴더에 저장
 - `set_global_seed()` 로 NumPy·random·PYTHONHASHSEED 모두 고정 → 반복 실행 시
   동일한 coverage / DBCV 보장
-------------------------------------------------------
사용 예)
    python tune_denseclus.py \
        --data_path ./data/train.csv \
        --save_name best_denseclus \
        --sample 20000 \
        --seed 42
    # 결과 파일은 yaml/best_denseclus_<날짜시간>.yaml, 
    # 로그 파일은 results/best_denseclus_<날짜시간>.log 로 저장
"""

import psutil, threading
import os, time
import argparse
import warnings
import random
from itertools import product
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from denseclus import DenseClus

def monitor_cpu():
    while True:
        print(f"CPU usage: {psutil.cpu_percent()}%")
        time.sleep(10)

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
    p = argparse.ArgumentParser(description="DenseClus 하이퍼파라미터 튜닝")
    p.add_argument("--data_path", default='./data/flat-training.csv', help="CSV 데이터 경로")
    p.add_argument("--save_path", default='./results', help="실험 결과 저장 경로")
    p.add_argument("--method", type=str, 
                   choices=["intersection", "union", "contrast", "intersection_union_mapper", "ensemble"],
                   default='intersection_union_mapper')
    p.add_argument("--sample", type=int, default=None, help="튜닝용 샘플 수")
    p.add_argument("--dropna", action="store_true", help="결측 컬럼 제거 여부")
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    p.add_argument("--max_clusters", type=int, default=10, help='최대 클러스터 개수')
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
def cluster_uniformity(counts: pd.DataFrame, eps: float = 1e-12) -> float:
    """
    0(최악) ~ 1(최고) 사이 균형 점수 반환.
    """
    counts_arr = counts.loc[counts['cluster'] != -1, "count"].values # -1 제외
    
    k = len(counts_arr) # 클러스터 개수
    p = counts_arr / counts_arr.sum() # 비율
    entropy = -np.sum(p * np.log(p + eps))

    return float(entropy / np.log(k)) # 정규화


def evaluate(method: str, umap_params: dict, hdbscan_params: dict, data: pd.DataFrame, seed: int, logger: logging.Logger):
    set_global_seed(seed)
    clf = DenseClus(
        random_state=seed,
        umap_combine_method=method,
        umap_params=umap_params,
        hdbscan_params=hdbscan_params,
    )
    clf.fit(data)
    labels = clf.labels_

    try:
        cnts = pd.DataFrame(clf.evaluate())[0].value_counts()
        cnts = cnts.reset_index()
        cnts.columns = ['cluster', 'count']
        cnts = cnts.sort_values(['cluster'], ignore_index=True)

        uniformity = cluster_uniformity(cnts)

        coverage = (labels >= 0).mean()  # 군집에 속한 비율
        dbcv = clf.hdbscan_.relative_validity_ # type: ignore
        n_cluster = len(np.unique(labels[labels >= 0]))

        return coverage, dbcv, n_cluster, uniformity
    
    except Exception as e:
        # 로그
        err_message = f"[WARN] evalute 실패: {e}"
        logger.info(err_message)
        return None

# ────────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # t = threading.Thread(target=monitor_cpu, daemon=True)
    # t.start()

    args = get_args()
    set_global_seed(args.seed)
    
    # 저장할 기본 세팅값
    setting = {
        "seed": args.seed,
        "n_samples": args.sample,
        "dropna": args.dropna,
        "method": args.method,
        "max_clusters": args.max_clusters
    }


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join(args.save_path, timestamp)
    os.makedirs(save_path)

    # 파일 저장 경로
    log_path = os.path.join(save_path, "train.log")
    yaml_path = os.path.join(save_path, "config.yaml")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

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
        for ms, mcs in product([50], [500, 1000, 2000])
    ]

    total_iter = len(umap_grid) * len(hdbscan_grid) // 2
    pbar = tqdm(total=total_iter, desc="Grid Search", ncols=110, colour="cyan")

    best_score = float("-inf")
    best_params = None

    for u_params in umap_grid:
        for h_params in hdbscan_grid:
            result = evaluate(args.method, u_params, h_params, df, args.seed, logger)

            if result is None:
                continue
            coverage, dbcv, n_clusters, uniformity = result

            msg = f"n_clusters: {n_clusters}"
            logger.info(msg)
            pbar.write("\n " + msg)

            # 클러스터 수가 args.max_clusters을 넘어가면 다음 루프로 진행
            if n_clusters > args.max_clusters:
                logger.info("클러스터 수가 10을 초과하므로 다음 루프로 넘깁니다.")
                logger.info(f"score={best_score:.4f} | cov={coverage:.4f}, dbcv={dbcv:.4f} | uniformity={uniformity:.4f}")
                continue

            # uniformity가 0.5 미만이면 다음 루프로 진행
            if uniformity < 0.5:
                logger.info("uniformity가 0.5보다 작으므로 다음 루프로 넘깁니다.")
                logger.info(f"score={best_score:.4f} | cov={coverage:.4f}, dbcv={dbcv:.4f} | uniformity={uniformity:.4f}")
                continue

            score = coverage * dbcv * uniformity

            if score > best_score:
                best_score = score

                best_params = {
                    "setting": setting,
                    "n_clusters": n_clusters,
                    "umap_params": u_params,
                    "hdbscan_params": h_params }

                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(best_params, f, sort_keys=False, allow_unicode=True, indent=4)

                best_msg = (
                    f"📈 New best → score={best_score:.4f} | cov={coverage:.4f}, dbcv={dbcv:.4f} | uniformity={uniformity:.4f}" )
                logger.info(best_msg)
                pbar.write("\n")

            pbar.update(1)

    pbar.close()

    logger.info("✅ 튜닝 완료!")
    logger.info(f"YAML saved to: {yaml_path}")
    logger.info(f"Log saved to : {log_path}")
    logger.info(f"Best score  : {best_score}")
    logger.info("Best params :\n" + yaml.safe_dump(best_params, sort_keys=False))
