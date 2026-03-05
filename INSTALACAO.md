# 📦 Instalação - MarketMind

Este guia explica como instalar as novas dependências para os recursos de PDF e Jupyter Notebook.

## 🔧 Dependências Adicionadas

As seguintes bibliotecas foram adicionadas ao projeto:

- **reportlab>=4.0.0** - Geração de relatórios PDF profissionais
- **matplotlib>=3.7.0** - Gráficos para PDF e análises no Jupyter

## 🧰 Criar e ativar um ambiente virtual

Se você quiser manter as dependências isoladas, crie um ambiente virtual antes de instalar qualquer pacote. No Windows:

```powershell
python -m venv venv      # cria a pasta venv
venv\Scripts\activate   # ativa o ambiente
```

No Linux ou macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Depois que o ambiente estiver ativo, instale as dependências conforme as seções seguintes.

## 📥 Como Instalar

### Opção 1: Instalar todas as dependências (Recomendado)

```bash
pip install -r requirements.txt
```

### Opção 2: Instalar apenas as novas dependências

```bash
pip install reportlab>=4.0.0 matplotlib>=3.7.0
```

## ✅ Verificar Instalação

Execute o seguinte comando para verificar se as bibliotecas foram instaladas:

```python
python -c "import reportlab; import matplotlib; print('✅ Todas as bibliotecas instaladas com sucesso!')"
```

## 🚀 Como Usar

### 1. Gerar Relatório PDF

1. Execute o aplicativo: `streamlit run app.py`
2. Faça login
3. Busque uma ação
4. Clique em "Gerar Previsão"
5. Clique em "📄 Gerar Relatório PDF"
6. O PDF será gerado e estará disponível para download

### 2. Usar Jupyter Notebook para Análises

```bash
# Instalar Jupyter (se ainda não tiver)
pip install jupyter

# Iniciar Jupyter Notebook
jupyter notebook

# Abrir o arquivo: analise_dados.ipynb
```

## 📄 Recursos do PDF

O relatório PDF inclui:

- ✅ Cabeçalho profissional com identidade visual MarketMind
- ✅ Informações básicas da ação (ticker, nome, data, método)
- ✅ Tabela de previsões com preço atual, 1 dia e 5 dias
- ✅ Tendência e confiança do modelo
- ✅ Detalhes por modelo individual (RandomForest, ExtraTrees, Ridge, ElasticNet)
- ✅ Gráfico histórico + previsão embutido
- ✅ Interpretação automática dos resultados
- ✅ Aviso legal e disclaimer completo
- ✅ Formatação profissional com cores e estilos

## 📊 Recursos do Jupyter Notebook

O notebook `analise_dados.ipynb` inclui:

- ✅ Funções para buscar dados da API BraPI
- ✅ Cálculo de 15+ indicadores técnicos (RSI, MACD, Bollinger, etc)
- ✅ Visualizações interativas (preço, volume, indicadores)
- ✅ Análise estatística completa
- ✅ Matriz de correlação entre indicadores
- ✅ Preparação de dados para ML (mesmo formato do app)
- ✅ Comparação de múltiplas ações
- ✅ Área de experimentação livre

## ⚠️ Solução de Problemas

### Erro ao importar reportlab

```bash
# Tente reinstalar
pip uninstall reportlab
pip install reportlab>=4.0.0
```

### Erro ao importar matplotlib

```bash
# Tente reinstalar
pip uninstall matplotlib
pip install matplotlib>=3.7.0
```

### Erro "No module named 'PIL'"

```bash
# Instalar Pillow (dependência do reportlab)
pip install Pillow
```

### Fonte não encontrada no PDF

Se o PDF apresentar problemas de fonte, verifique se as fontes padrão do ReportLab estão instaladas corretamente. O código usa fontes padrão (Helvetica) que devem funcionar em qualquer sistema.

## 🔄 Atualizando o Ambiente Virtual

Se você estiver usando um ambiente virtual (recomendado):

```bash
# Ativar ambiente virtual
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## 📝 Notas Importantes

1. **O erro do relatório TXT foi corrigido**: O código antigo tentava acessar `previsoes_ensemble[13]` (dia 14), mas o array só tinha 5 elementos. Isso foi corrigido na nova implementação em PDF.

2. **Formato do relatório mudou de TXT para PDF**: O relatório agora é gerado em formato PDF profissional com gráficos embutidos.

3. **Jupyter Notebook é opcional**: O notebook é para análises exploratórias e estudos. O aplicativo principal funciona independentemente.

## 📧 Suporte

Se encontrar problemas durante a instalação, verifique:

1. Versão do Python (recomendado: 3.8+)
2. Pip atualizado: `pip install --upgrade pip`
3. Conflitos de dependências: `pip check`

---

**✅ Pronto!** Todas as dependências instaladas e o projeto está configurado para gerar PDFs e realizar análises no Jupyter Notebook.
