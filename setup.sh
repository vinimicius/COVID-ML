#!/bin/bash
echo "🚀 Criando ambiente virtual..."
python -m venv .venv
source .venv/bin/activate
echo "📦 Instalar dependências..."
pip install -r requirements.txt
echo "✅ Tudo pronto! Use 'source .venv/bin/activate' para começar."