@echo off
echo ========================================
echo   探査速度分析スクリプト
echo ========================================
echo.

echo 📊 Config A、B、C、Dの探査率向上スピードを分析します
echo.

REM 必要なライブラリをインストール
echo 📦 必要なライブラリを確認・インストール中...
pip install pandas matplotlib seaborn scipy numpy --quiet

echo.
echo 🚀 探査速度分析を開始します...
echo.

REM 探査速度分析スクリプトを実行
python exploration_speed_analysis.py

echo.
echo 分析が完了しました！
echo 結果は exploration_speed_analysis/ ディレクトリに保存されています。

REM 結果ディレクトリを開く
if exist "exploration_speed_analysis" (
    echo.
    echo 結果ディレクトリを開きますか？ (Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        explorer exploration_speed_analysis
    )
)

pause 