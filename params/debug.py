#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass

@dataclass
class DebugParam:
    """デバッグ設定パラメータ"""
    
    # 分岐・統合イベントのログ
    log_branch_events: bool = False
    log_integration_events: bool = False
    
    # 学習進捗のログ
    log_learning_progress: bool = False
    
    # 一般的なデバッグログ
    enable_debug_log: bool = False
    
    # VFH-Fuzzyアルゴリズムのデバッグ
    log_vfh_fuzzy: bool = False
    
    # エージェント行動のデバッグ
    log_agent_actions: bool = False 