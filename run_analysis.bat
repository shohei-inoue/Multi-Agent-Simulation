@echo off
echo ========================================
echo   シミュレーション結果解析スクリプト実行
echo ========================================
echo.

REM 必要なライブラリをインストール
echo 必要なライブラリをチェック中...
python -c "import pandas, matplotlib, seaborn, numpy" 2>nul
if %errorlevel% neq 0 (
    echo 必要なライブラリをインストール中...
    pip install pandas matplotlib seaborn numpy
)

echo.
echo 解析を開始します...
python analyze_verification_results.py

echo.
echo 解析が完了しました！
echo 結果は analysis_results/ ディレクトリに保存されています。
echo.

REM 結果ディレクトリを開く
if exist "analysis_results" (
    echo 結果ディレクトリを開きますか？ (Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        explorer analysis_results
    )
)

pause 