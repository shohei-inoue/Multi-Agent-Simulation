#!/usr/bin/env python3
"""
Markdown to PDF Converter
MarkdownファイルをPDFに変換するスクリプト
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def install_requirements():
    """必要なライブラリをインストール"""
    requirements = [
        'markdown',
        'weasyprint',
        'pymdown-extensions',
        'markdown-math-extension'
    ]
    
    print("必要なライブラリをインストール中...")
    for req in requirements:
        try:
            __import__(req.replace('-', '_'))
            print(f"✓ {req} は既にインストールされています")
        except ImportError:
            print(f"📦 {req} をインストール中...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
            print(f"✓ {req} インストール完了")

def markdown_to_html(md_file: Path, output_dir: Path) -> Path:
    """MarkdownをHTMLに変換"""
    import markdown
    from markdown.extensions import codehilite, toc, tables
    from pymdown import superfences
    
    # Markdownファイルを読み込み
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Markdown拡張を設定
    extensions = [
        'markdown.extensions.codehilite',
        'markdown.extensions.toc',
        'markdown.extensions.tables',
        'markdown.extensions.fenced_code',
        'pymdown.superfences',
        'pymdown.arithmatex'
    ]
    
    extension_configs = {
        'codehilite': {
            'css_class': 'highlight',
            'use_pygments': True
        },
        'toc': {
            'permalink': True
        },
        'pymdown.arithmatex': {
            'generic': True
        }
    }
    
    # MarkdownをHTMLに変換
    md = markdown.Markdown(
        extensions=extensions,
        extension_configs=extension_configs
    )
    
    html_content = md.convert(md_content)
    
    # HTMLテンプレートを作成
    html_template = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{md_file.stem}</title>
    <style>
        body {{
            font-family: 'Yu Gothic', 'Hiragino Sans', 'Meiryo', sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 2em;
            margin-bottom: 1em;
        }}
        
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        
        h2 {{
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
        }}
        
        h3 {{
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        
        code {{
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
        }}
        
        pre {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            margin: 1em 0;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 1em 0;
            padding-left: 20px;
            color: #555;
            background-color: #f9f9f9;
            padding: 10px 20px;
        }}
        
        /* 数式スタイル */
        .math {{
            font-family: 'Times New Roman', serif;
            font-size: 1.1em;
        }}
        
        /* 印刷用スタイル */
        @media print {{
            body {{
                font-size: 12pt;
                line-height: 1.4;
            }}
            
            h1 {{
                page-break-before: always;
            }}
            
            h1:first-child {{
                page-break-before: auto;
            }}
            
            h2, h3 {{
                page-break-after: avoid;
            }}
            
            pre, table {{
                page-break-inside: avoid;
            }}
        }}
        
        /* コードハイライト */
        .highlight {{
            background-color: #f8f9fa;
        }}
        
        .highlight .k {{ color: #d73a49; }} /* キーワード */
        .highlight .s {{ color: #032f62; }} /* 文字列 */
        .highlight .c {{ color: #6a737d; }} /* コメント */
        .highlight .n {{ color: #24292e; }} /* 名前 */
    </style>
    
    <!-- MathJax for mathematical expressions -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                processEnvironments: true
            }},
            options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
            }}
        }};
    </script>
</head>
<body>
    {html_content}
</body>
</html>
"""
    
    # HTMLファイルを保存
    html_file = output_dir / f"{md_file.stem}.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    return html_file

def html_to_pdf(html_file: Path, output_dir: Path) -> Path:
    """HTMLをPDFに変換"""
    try:
        from weasyprint import HTML, CSS
        
        # CSSスタイルを追加
        css = CSS(string="""
            @page {
                size: A4;
                margin: 2cm;
            }
            
            body {
                font-size: 11pt;
            }
            
            h1 {
                font-size: 18pt;
            }
            
            h2 {
                font-size: 16pt;
            }
            
            h3 {
                font-size: 14pt;
            }
            
            code {
                font-size: 9pt;
            }
        """)
        
        # PDFに変換
        pdf_file = output_dir / f"{html_file.stem}.pdf"
        HTML(filename=str(html_file)).write_pdf(str(pdf_file), stylesheets=[css])
        
        return pdf_file
        
    except ImportError:
        print("❌ WeasyPrint が利用できません。代替方法を試します...")
        return html_to_pdf_alternative(html_file, output_dir)

def html_to_pdf_alternative(html_file: Path, output_dir: Path) -> Path:
    """代替方法でHTMLをPDFに変換"""
    try:
        import pdfkit
        
        options = {
            'page-size': 'A4',
            'margin-top': '2cm',
            'margin-right': '2cm',
            'margin-bottom': '2cm',
            'margin-left': '2cm',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        pdf_file = output_dir / f"{html_file.stem}.pdf"
        pdfkit.from_file(str(html_file), str(pdf_file), options=options)
        
        return pdf_file
        
    except (ImportError, OSError):
        print("❌ PDF変換ライブラリが利用できません。")
        print("💡 以下の方法でPDFを作成してください：")
        print(f"   1. ブラウザで {html_file} を開く")
        print("   2. Ctrl+P で印刷ダイアログを開く")
        print("   3. 「PDFに保存」を選択して保存")
        return html_file

def convert_markdown_to_pdf(md_file: str, output_dir: str = "pdf_output"):
    """メイン変換関数"""
    md_path = Path(md_file)
    output_path = Path(output_dir)
    
    if not md_path.exists():
        print(f"❌ ファイルが見つかりません: {md_file}")
        return False
    
    # 出力ディレクトリを作成
    output_path.mkdir(exist_ok=True)
    
    try:
        print(f"📄 変換開始: {md_path.name}")
        
        # MarkdownをHTMLに変換
        print("  🔄 Markdown → HTML")
        html_file = markdown_to_html(md_path, output_path)
        print(f"  ✓ HTML生成完了: {html_file}")
        
        # HTMLをPDFに変換
        print("  🔄 HTML → PDF")
        pdf_file = html_to_pdf(html_file, output_path)
        
        if pdf_file.suffix == '.pdf':
            print(f"  ✅ PDF生成完了: {pdf_file}")
            return True
        else:
            print(f"  ⚠️ HTML生成完了: {pdf_file}")
            print("     ブラウザでHTMLを開いてPDFとして印刷してください")
            return True
            
    except Exception as e:
        print(f"❌ 変換エラー: {e}")
        return False

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Markdown to PDF Converter')
    parser.add_argument('files', nargs='*', help='変換するMarkdownファイル')
    parser.add_argument('-o', '--output', default='pdf_output', help='出力ディレクトリ')
    parser.add_argument('--install', action='store_true', help='必要なライブラリをインストール')
    
    args = parser.parse_args()
    
    if args.install:
        install_requirements()
        return
    
    # ファイルが指定されていない場合、docsディレクトリのMarkdownファイルを検索
    if not args.files:
        docs_dir = Path('docs')
        if docs_dir.exists():
            md_files = list(docs_dir.glob('*.md'))
            if md_files:
                print(f"📁 docsディレクトリから {len(md_files)} 個のMarkdownファイルを発見")
                args.files = [str(f) for f in md_files]
            else:
                print("❌ 変換するMarkdownファイルが見つかりません")
                return
        else:
            print("❌ docsディレクトリが見つかりません")
            return
    
    # 必要なライブラリをインストール
    install_requirements()
    
    # 各ファイルを変換
    success_count = 0
    for md_file in args.files:
        if convert_markdown_to_pdf(md_file, args.output):
            success_count += 1
    
    print(f"\n🎉 変換完了: {success_count}/{len(args.files)} ファイル")
    print(f"📁 出力ディレクトリ: {args.output}")

if __name__ == "__main__":
    main() 