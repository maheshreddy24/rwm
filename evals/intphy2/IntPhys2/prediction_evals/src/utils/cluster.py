# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
import subprocess
from functools import lru_cache

from src.utils.logging import get_logger

logger = get_logger("Cluster utils")

SUPPORTED_CLUSTERS = {
}

@lru_cache()
def get_cluster() -> str:
    # If the node is assigned by slurm, this is easy
    where = os.environ.get("SLURM_CLUSTER_NAME")
    if where is not None:
        if where in SUPPORTED_CLUSTERS:
            return SUPPORTED_CLUSTERS[where]
        else:
            # return the cluster name so the user knows to add support for it
            return where
    return hostname


# Gets slurm job vars, to launch another job with the same vars
def slurm_account_partition_and_qos(low_pri: bool) -> str:
    account = os.environ.get("SLURM_JOB_ACCOUNT")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    qos = os.environ.get("SLURM_JOB_QOS")
    assert None not in (account, partition, qos), "This function should only be called by a job scheduled by slurm"
    return account, partition, qos


DATASET_PATHS_BY_CLUSTER = {
}


def get_dataset_path(dataset: str, cluster=None) -> str:
    if cluster is None:
        cluster = get_cluster()

    return DATASET_PATHS_BY_CLUSTER[cluster][dataset]


def get_dataset_paths(datasets: list[str], is_train: bool = True) -> list[str]:
    cluster = get_cluster()
    assert cluster in DATASET_PATHS_BY_CLUSTER, f"No data paths for environment {cluster}!"
    paths = []
    for dataset in datasets:
        if not is_train:
            dataset = f"{dataset}_val"
        try:
            path = get_dataset_path(dataset, cluster)
        except Exception:
            raise Exception(f"Could not find dataset {dataset} for cluster {cluster}")
        paths.append(path)
    logger.info(f"Datapaths {paths}")
    return paths


def dataset_paths() -> dict[str, str]:
    cluster = get_cluster()
    assert cluster in DATASET_PATHS_BY_CLUSTER, f"No data paths for environment {cluster}!"
    return DATASET_PATHS_BY_CLUSTER[cluster]
