@echo off
echo ========================================
echo  Exploration Rate Progress Single Graphs
echo ========================================
echo.
echo 1エピソード目のExploration Rate Progressグラフを
echo 3つの環境（障害物密度）別に単体で生成します
echo.
echo - 障害物密度 0.0   (障害物なし)
echo - 障害物密度 0.003 (低密度)  
echo - 障害物密度 0.005 (中密度)
echo.

echo Pythonスクリプトを実行中...
python exploration_rate_progress_single.py

echo.
echo ========================================
echo 単体グラフ生成完了！結果を確認してください。
echo ========================================
pause 