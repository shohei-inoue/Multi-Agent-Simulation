@echo off
echo ========================================
echo   Markdown to PDF Converter
echo ========================================
echo.

echo 📚 docsディレクトリのMarkdownファイルをPDFに変換します
echo.

REM PDF変換スクリプトを実行
python convert_to_pdf.py

echo.
echo 変換が完了しました！
echo 結果は pdf_output/ ディレクトリに保存されています。

REM 結果ディレクトリを開く
if exist "pdf_output" (
    echo.
    echo 結果ディレクトリを開きますか？ (Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        explorer pdf_output
    )
)

pause 