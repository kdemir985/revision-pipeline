from __future__ import annotations
from typing import Tuple, List
import numpy as np
import pandas as pd


def project_level_split(
    df: pd.DataFrame, seed: int, train_frac: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    projects = sorted(df["project"].unique())
    rng = np.random.RandomState(seed)
    shuffled = list(projects)
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_frac))
    train_projs = sorted(shuffled[:n_train])
    test_projs = sorted(shuffled[n_train:])
    train_df = df[df["project"].isin(train_projs)].reset_index(drop=True)
    test_df = df[df["project"].isin(test_projs)].reset_index(drop=True)
    return train_df, test_df, train_projs, test_projs
