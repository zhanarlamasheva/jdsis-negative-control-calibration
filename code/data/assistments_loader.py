"""ASSISTments 2012-2013 loader for MARS pipeline (Western KT external-validity).

Converts the US K-12 math tutoring log into the same DataFrame schema EdNet/XES
use (user_id, timestamp, question_id, tags, correct, elapsed_time, changed_answer,
part_id) so run_ablation_v2 / baselines run unchanged.

Unlike XES, ms_first_response is a REAL response time -> Confidence rt_cv is live.

The raw csv is ~82M rows, so we two-pass: scan user_id only, sample n_students,
then chunked-read keeping only sampled users' rows.

Usage:
    from data.assistments_loader import load_assistments
    train_df, val_df, test_df = load_assistments(
        csv_path="../assistments_2012/2012-2013-data-with-predictions-4-final.csv",
        n_students=6000, min_interactions=20, seed=42)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("mars.data.assistments")

_USECOLS = ["user_id", "skill_id", "problem_id", "correct", "start_time",
            "ms_first_response", "attempt_count"]


def load_assistments(
    csv_path: str = "../assistments_2012/2012-2013-data-with-predictions-4-final.csv",
    n_students: int = 6000,
    min_interactions: int = 20,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    path = Path(csv_path)
    logger.info("Loading ASSISTments from %s", path)

    # pass 1: unique users (user_id only) -> sample
    uids = pd.read_csv(path, usecols=["user_id"])["user_id"].dropna().unique()
    logger.info("Raw users: %d", len(uids))
    if len(uids) > n_students:
        uids = rng.choice(uids, size=n_students, replace=False)
    keep = set(uids)

    # pass 2: chunked read, keep sampled users + skill-tagged rows only
    parts = []
    for chunk in pd.read_csv(path, usecols=_USECOLS, chunksize=2_000_000):
        c = chunk[chunk["user_id"].isin(keep)]
        c = c.dropna(subset=["skill_id", "correct", "start_time"])
        if len(c):
            parts.append(c)
    df = pd.concat(parts, ignore_index=True)
    logger.info("Sampled rows: %d from %d users", len(df), df["user_id"].nunique())

    df["correct"] = df["correct"].astype(int)
    df = df[df["correct"].isin([0, 1])]
    # remap arbitrary skill_id -> contiguous 0..K-1 tags
    df["skill_id"] = df["skill_id"].astype(int)
    tag_map = {s: i for i, s in enumerate(sorted(df["skill_id"].unique()))}
    df["tags"] = df["skill_id"].map(tag_map).map(lambda t: [int(t)])
    df["timestamp"] = (pd.to_datetime(df["start_time"], errors="coerce")
                       .astype("int64") // 10**9)
    df = df.dropna(subset=["timestamp"])
    df["question_id"] = "q" + df["problem_id"].astype(int).astype(str)
    et = pd.to_numeric(df["ms_first_response"], errors="coerce").fillna(15000.0)
    df["elapsed_time"] = et.clip(lower=0, upper=300000)  # cap 5 min outliers
    df["changed_answer"] = (pd.to_numeric(df["attempt_count"], errors="coerce")
                            .fillna(1) > 1).astype(int)
    df["part_id"] = 1
    df["user_id"] = df["user_id"].astype(str)
    df = df[["user_id", "timestamp", "question_id", "tags", "correct",
             "elapsed_time", "changed_answer", "part_id"]]

    # drop users below min_interactions
    counts = df.groupby("user_id")["correct"].transform("size")
    df = df[counts >= min_interactions].reset_index(drop=True)
    logger.info("After min_interactions=%d: %d rows, %d users",
                min_interactions, len(df), df["user_id"].nunique())

    # user-level split (whole users -> train/val/test), same as XES loader
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    all_users = list(df["user_id"].unique())
    rng.shuffle(all_users)
    n_train = int(len(all_users) * train_ratio)
    n_val = int(len(all_users) * val_ratio)
    train_users = set(all_users[:n_train])
    val_users = set(all_users[n_train:n_train + n_val])

    train = df[df["user_id"].isin(train_users)].reset_index(drop=True)
    val = df[df["user_id"].isin(val_users)].reset_index(drop=True)
    test = df[~df["user_id"].isin(train_users | val_users)].reset_index(drop=True)

    n_tags = len(tag_map)
    logger.info("ASSISTments stats: %d skills, %d questions, %.1f avg int/student; "
                "splits train=%d val=%d test=%d",
                n_tags, df["question_id"].nunique(),
                len(df) / df["user_id"].nunique(), len(train), len(val), len(test))
    return train, val, test
