# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, TYPE_CHECKING

import pandas as pd
from hta.common.trace_filter import CPUOperatorFilter, GPUKernelFilter
from hta.configs.config import logger

if TYPE_CHECKING:
    from hta.common.trace import Trace


class OperatorCountAnalysis:
    """Analyzer for counting CPU operators and GPU kernels in a trace.

    This class provides methods to count the total number of CPU operators and GPU kernels,
    as well as the number of unique operator/kernel types. Results can be broken down by
    iteration (profiler step) and by rank.
    """

    def __init__(self):
        pass

    @classmethod
    def get_operator_counts(
        cls,
        t: "Trace",
        ranks: Optional[List[int]] = None,
        by_iteration: bool = False,
    ) -> pd.DataFrame:
        """Count CPU operators and GPU kernels across ranks.

        Args:
            t (Trace): A Trace object containing the parsed trace data.
            ranks (List[int], optional): List of ranks to analyze. If None, all ranks are used.
            by_iteration (bool): If True, counts are broken down by iteration (profiler step).
                Default = False.

        Returns:
            pd.DataFrame: A DataFrame with columns:
                - rank: the rank number
                - iteration (if by_iteration=True): the iteration/profiler step number
                - cpu_operator_count: total number of CPU operator events
                - cpu_unique_operator_count: number of unique CPU operator types
                - gpu_kernel_count: total number of GPU kernel events
                - gpu_unique_kernel_count: number of unique GPU kernel types
        """
        if ranks is None:
            ranks = sorted(t.traces.keys())

        sym_table = t.symbol_table.get_sym_table()
        _sym_lookup = sym_table.get if isinstance(sym_table, dict) else (
            lambda idx, default=None: sym_table[idx] if 0 <= idx < len(sym_table) else default
        )
        results = []

        for rank in ranks:
            if rank not in t.traces:
                logger.warning(f"Rank {rank} not found in trace data.")
                continue

            trace_df = t.traces[rank].copy()

            # Decode integer name IDs to string names for unique counting
            if "name" in trace_df.columns:
                trace_df["name_str"] = trace_df["name"].apply(
                    lambda idx: _sym_lookup(idx, str(idx))
                )
            else:
                trace_df["name_str"] = trace_df.get("s_name", "unknown")

            # Filter CPU and GPU events
            cpu_events = CPUOperatorFilter()(trace_df, t.symbol_table)
            gpu_events = GPUKernelFilter()(trace_df, t.symbol_table)

            iterations: List[int]
            if by_iteration and "iteration" in trace_df.columns:
                iterations = sorted(set(
                    cpu_events["iteration"].unique().tolist()
                    + gpu_events["iteration"].unique().tolist()
                ))
                # Remove -1 iteration (unassigned)
                iterations = [it for it in iterations if it >= 0]
                if not iterations:
                    iterations = [-1]
            else:
                iterations = [-1]

            for iteration in iterations:
                if iteration >= 0:
                    cpu_iter = cpu_events[cpu_events["iteration"] == iteration]
                    gpu_iter = gpu_events[gpu_events["iteration"] == iteration]
                else:
                    cpu_iter = cpu_events
                    gpu_iter = gpu_events

                cpu_total = len(cpu_iter)
                cpu_unique = cpu_iter["name_str"].nunique() if not cpu_iter.empty else 0
                gpu_total = len(gpu_iter)
                gpu_unique = gpu_iter["name_str"].nunique() if not gpu_iter.empty else 0

                row = {
                    "rank": rank,
                    "cpu_operator_count": cpu_total,
                    "cpu_unique_operator_count": cpu_unique,
                    "gpu_kernel_count": gpu_total,
                    "gpu_unique_kernel_count": gpu_unique,
                }
                if by_iteration:
                    row["iteration"] = iteration
                results.append(row)

        result_df = pd.DataFrame(results)
        if result_df.empty:
            logger.warning("No operator count data generated.")
            return result_df

        if by_iteration:
            col_order = [
                "rank",
                "iteration",
                "cpu_operator_count",
                "cpu_unique_operator_count",
                "gpu_kernel_count",
                "gpu_unique_kernel_count",
            ]
        else:
            col_order = [
                "rank",
                "cpu_operator_count",
                "cpu_unique_operator_count",
                "gpu_kernel_count",
                "gpu_unique_kernel_count",
            ]
        return result_df[col_order].reset_index(drop=True)

    @classmethod
    def get_operator_count_summary(
        cls,
        t: "Trace",
        ranks: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Get a summary of operator counts across all ranks.

        Args:
            t (Trace): A Trace object containing the parsed trace data.
            ranks (List[int], optional): List of ranks to analyze. If None, all ranks are used.

        Returns:
            pd.DataFrame: A summary DataFrame with min, max, mean, std, and total
                for each count metric across ranks.
        """
        counts_df = cls.get_operator_counts(t, ranks=ranks, by_iteration=False)
        if counts_df.empty:
            return counts_df

        metrics = [
            "cpu_operator_count",
            "cpu_unique_operator_count",
            "gpu_kernel_count",
            "gpu_unique_kernel_count",
        ]

        summary = {}
        for metric in metrics:
            if metric in counts_df.columns:
                summary[f"{metric}_min"] = [counts_df[metric].min()]
                summary[f"{metric}_max"] = [counts_df[metric].max()]
                summary[f"{metric}_mean"] = [counts_df[metric].mean()]
                summary[f"{metric}_std"] = [counts_df[metric].std()]
                summary[f"{metric}_total"] = [counts_df[metric].sum()]

        return pd.DataFrame(summary)

    @classmethod
    def get_unique_operator_names(
        cls,
        t: "Trace",
        rank: int = 0,
        device: str = "both",
    ) -> pd.DataFrame:
        """Get the set of unique operator/kernel names for a given rank.

        Args:
            t (Trace): A Trace object containing the parsed trace data.
            rank (int): The rank to analyze. Default = 0.
            device (str): Which device types to include.
                "cpu" - only CPU operators
                "gpu" - only GPU kernels
                "both" - both CPU and GPU (default)

        Returns:
            pd.DataFrame: A DataFrame with columns:

                - name: the operator/kernel name
                - count: number of occurrences of this operator/kernel
                - device: "CPU" or "GPU"
                - total_duration_us: total duration in microseconds
                - mean_duration_us: mean duration in microseconds
        """
        if rank not in t.traces:
            logger.error(f"Rank {rank} not found in trace data.")
            return pd.DataFrame()

        trace_df = t.traces[rank].copy()
        sym_table = t.symbol_table.get_sym_table()
        # get_sym_table() may return a dict or a list depending on trace format
        if isinstance(sym_table, dict):
            _sym_lookup = sym_table.get
        else:
            _sym_lookup = lambda idx, default=None: sym_table[idx] if 0 <= idx < len(sym_table) else default

        if "name" in trace_df.columns:
            trace_df["name_str"] = trace_df["name"].apply(
                lambda idx: _sym_lookup(idx, str(idx))
            )
        else:
            trace_df["name_str"] = trace_df.get("s_name", "unknown")

        results = []

        if device in ("cpu", "both"):
            cpu_events = CPUOperatorFilter()(trace_df, t.symbol_table)
            if not cpu_events.empty:
                cpu_stats = (
                    cpu_events.groupby("name_str")
                    .agg(count=("name_str", "count"), total_dur=("dur", "sum"))
                    .reset_index()
                )
                cpu_stats["mean_duration_us"] = (
                    cpu_stats["total_dur"] / cpu_stats["count"]
                )
                cpu_stats["device"] = "CPU"
                cpu_stats = cpu_stats.rename(
                    columns={
                        "name_str": "name",
                        "total_dur": "total_duration_us",
                    }
                )
                results.append(cpu_stats)

        if device in ("gpu", "both"):
            gpu_events = GPUKernelFilter()(trace_df, t.symbol_table)
            if not gpu_events.empty:
                gpu_stats = (
                    gpu_events.groupby("name_str")
                    .agg(count=("name_str", "count"), total_dur=("dur", "sum"))
                    .reset_index()
                )
                gpu_stats["mean_duration_us"] = (
                    gpu_stats["total_dur"] / gpu_stats["count"]
                )
                gpu_stats["device"] = "GPU"
                gpu_stats = gpu_stats.rename(
                    columns={
                        "name_str": "name",
                        "total_dur": "total_duration_us",
                    }
                )
                results.append(gpu_stats)

        if not results:
            return pd.DataFrame()

        result_df = pd.concat(results, ignore_index=True)
        return result_df[
            ["name", "device", "count", "total_duration_us", "mean_duration_us"]
        ].sort_values("count", ascending=False)

