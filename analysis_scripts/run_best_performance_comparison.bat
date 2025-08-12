@echo off
echo ========================================
echo  Best Performance Comparison Analysis
echo ========================================
echo.
echo 各Configで最終探査率が最も良かった環境を抽出し、
echo ステップごとの探査率上昇を比較します
echo.
echo - Config A: VFH-Fuzzy only
echo - Config B: Pre-trained model  
echo - Config C: Branching/Integration
echo - Config D: Branching/Integration + Learning
echo.

echo Pythonスクリプトを実行中...
python best_performance_comparison.py

echo.
echo ========================================
echo 最高性能比較分析完了！結果を確認してください。
echo ========================================
pause 