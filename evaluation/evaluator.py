#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yulin
@created: 2026/6/19 15:58
@updated: 2026/6/19 15:58
@version: 1.0
@description: 
"""
class XHSRAGEvaluator:
    def build_golden_set(self):
        """手动准备10条query，每条标注应该召回的note_id"""
        return [
            {"query": "iPhone16Pro值得买吗", "expected_note_ids": ["xxx", "yyy"]},
            {"query": "平价护肤品推荐", "expected_note_ids": ["aaa", "bbb"]},
            # ...共10条
        ]

    def hit_rate(self, results, expected, k=5) -> float:
        """Top-K 命中率"""

    def mrr(self, results, expected) -> float:
        """平均倒数排名"""

    def run_ablation_eval(self) -> dict:
        """
        对比四种配置：
        bm25_only / dense_only / hybrid_no_rerank / hybrid_rerank
        输出各配置的 Hit@5 和 MRR
        """