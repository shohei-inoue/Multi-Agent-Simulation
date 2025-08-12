@echo off
echo ========================================
echo  Config A/B/C/D Comparison Analysis
echo ========================================
echo.
echo Config A: VFH-Fuzzy only (No branching/integration, No learning)
echo Config B: Pre-trained model (No branching/integration)
echo Config C: Branching/Integration enabled (No learning)
echo Config D: Branching/Integration + Learning
echo.

echo Pythonスクリプトを実行中...
python config_comparison_analysis.py

echo.
echo ========================================
echo 分析完了！結果を確認してください。
echo ========================================
pause 