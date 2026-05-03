# 🔥 Pipeline de Classificação de Risco de Queimadas — Pará (2019–2025)

Classificação de risco de queimadas nos 144 municípios do estado do Pará por meio de aprendizado de máquina (Random Forest, Extra Trees, XGBoost) integrado ao método multicritério ELECTRE Tri-B.

---

## Requisitos

- Python **3.10**
- Chave de API do **NASA FIRMS** (gratuita) → [Solicitar aqui](https://firms.modaps.eosdis.nasa.gov/api/area/)

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Crie e ative um ambiente virtual

```bash
python3.10 -m venv .venv
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## Estrutura do Projeto

```
.
├── main.py                  # Ponto de entrada do pipeline
├── requirements.txt
├── core/
│   ├── __init__.py
│   ├── data_IBGE.py         # Download dos municípios via API do IBGE
│   ├── data_NASAPOWER.py    # Coleta de variáveis climáticas mensais
│   ├── data_NASAFIRMS.py    # Coleta e limpeza de focos de calor
│   ├── build_features.py    # Engenharia de features e definição do target
│   ├── treinamento.py       # Treino, validação e seleção do modelo
│   └── visualizacao.py      # Mapas e gráficos de importância
└── output/                  # Gerado automaticamente na primeira execução
```

---

## Execução

```bash
python main.py --real --key SUA_CHAVE_FIRMS
```

Substitua `SUA_CHAVE_FIRMS` pela chave obtida no portal da NASA FIRMS.

---

## Saídas

Todos os arquivos são salvos na pasta `output/` criada automaticamente:

| Arquivo | Descrição |
|---|---|
| `nasa_power_para.csv` | Variáveis climáticas mensais por município |
| `firms_bruto.csv` | Contagem mensal de focos por município (antes da limpeza) |
| `firms_limpo.csv` | Contagem mensal de focos por município (após limpeza) |
| `firms_mensal.csv` | Contagem mensal de focos por município (após limpeza geral) |
| `mapa_risco_electre_trib.png` | Mapa de risco classificado pelo ELECTRE Tri-B |

---

Este projeto foi desenvolvido como trabalho acadêmico no Programa de Pós-Graduação em Computação (PPGCOMP) — Universidade Federal do Pará (UFPA).