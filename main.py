"""
PIPELINE — Classificação de Risco de Queimadas no Pará (2019–2025)
==================================================================
Fontes:
  - NASA POWER  - variáveis climáticas mensais (sem autenticação)
  - NASA FIRMS  - focos de incêndio (target)

Abordagem ML:
  Por município x mês - classificação binária:
  "Haverá foco relevante no próximo mês?"
  Modelos comparados: Random Forest, Extra Trees, XGBoost.
  Saída final: probabilidade por município -> classificação ELECTRE Tri-B
  em categorias de risco (Baixo / Moderado / Alto / Crítico).

Divisão temporal:
  Treino: 2019–2022 | Validação: 2023 | Teste: 2024–2025

Execução:
    python main.py
    python main.py --real --key SUA_KEY_FIRMS
"""

import warnings
import numpy as np
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import fbeta_score, classification_report, roc_auc_score, precision_recall_curve

from core.data_IBGE import get_municipios_para
from core.data_NASAPOWER import collect_nasa_power
from core.data_NASAFIRMS import get_firms_municipios
from core.build_features import build_features
from core.treinamento import avaliar_com_tscv, split_temporal, train_models
from core.visualizacao import plot_feature_importance, plot_mapa_geral

warnings.filterwarnings("ignore")

OUT = Path("output"); OUT.mkdir(exist_ok=True)

def main(firms_api_key: str = "SUA_KEY_AQUI",
         demo_municipios: int = 30):
    """
    demo_municipios: número de municípios para coleta rápida na demo.
                     Use None para todos os 144 municípios do Pará.
    """
    print("=" * 60)
    print("  Risco de Queimadas — Pará (2019–2025) · NASA POWER + ML")
    print("=" * 60)

    # 1. Municípios
    gdf = get_municipios_para()

    # 2. NASA POWER (dados climáticos mensais)
    print("\n Passo 2 — NASA POWER")
    df_power = collect_nasa_power(gdf, out=OUT, start="2019")

    # 3. FIRMS (focos)
    print("\n Passo 3 — NASA FIRMS")
    df_firms = get_firms_municipios(gdf, out=OUT, api_key=firms_api_key)

    # 4. Features
    print("\n Passo 4 — Build Features")
    # df = build_features(df_power, df_firms)
    df = build_features(df_power, df_firms, train_cutoff="2023-01-01")

    # 5. Treino
    print("\n Passo 5 — Validação")
    aucs = avaliar_com_tscv(df, n_splits=3)
    if np.mean(aucs) < 0.65:
        print("AUC médio abaixo de 0.65 — revisar features antes de continuar")
        return

    # O split_temporal() e train_models() continuam para o modelo final
    print("\n Passo 5b — Treino Final")
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), feats = split_temporal(df)
    
    # Desempacota todos os 4 valores retornados
    rf, xgb_model, et, best = train_models(X_tr, y_tr, X_val, y_val)

    beta = 2  # recall vale 2x mais que precision

    probs_val = best.predict_proba(X_val)[:, 1]
    _, _, thresholds = precision_recall_curve(y_val, probs_val)

    scores = [fbeta_score(y_val, (probs_val >= t).astype(int), beta=beta)
            for t in thresholds]
    best_threshold = thresholds[np.argmax(scores)]
    print(f"  Threshold ótimo (F{beta}): {best_threshold:.3f}")

    # 6. Avaliação no teste
    print("\n Resultado no conjunto de teste:")
    probs_te = best.predict_proba(X_te)[:, 1]
    y_pred = (probs_te >= best_threshold).astype(int)

    print(classification_report(y_te, y_pred, target_names=["Sem foco", "Com foco"]))
    print(f"  AUC-ROC (teste): {roc_auc_score(y_te, probs_te):.3f}")

    # 7. Visualizações
    print("\n Gerando visualizações...")
    if best is rf:
        name_model = "Random Forest"
    elif best is et:
        name_model = "Extra Trees"
    else:
        name_model = "XGBoost"

    plot_feature_importance(best, feats, name_model, out=OUT)
    plot_mapa_geral(df, best, feats, gdf, out=OUT, calibrar_perfis=True)

    # Curva Precision-Recall
    fig, ax = plt.subplots(figsize=(7, 5))
    for nome, modelo in [("Random Forest", rf),
                         ("Extra Trees", et),
                         ("XGBoost", xgb_model)]:
        PrecisionRecallDisplay.from_estimator(
            modelo, X_val, y_val, name=nome, ax=ax
        )
    ax.axvline(x=best_threshold, color="gray", linestyle="--",
               alpha=0.5, label=f"τ={best_threshold:.3f}")
    ax.set_title("Curva Precision-Recall — Validação (2023)")
    plt.tight_layout()
    plt.savefig(OUT / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP
    try:
        modelo_base = xgb_model.calibrated_classifiers_[0].estimator
    except AttributeError:
        modelo_base = xgb_model

    import tempfile
    import os

    # Salva e recarrega o modelo para normalizar o base_score
    booster = modelo_base.get_booster()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    booster.save_model(tmp_path)
    booster.load_model(tmp_path)
    os.unlink(tmp_path)

    explainer   = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_te)

    shap.summary_plot(shap_values, X_te,
                      feature_names=feats, show=False)
    plt.savefig(OUT / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_te,
                      feature_names=feats, plot_type="dot", show=False)
    plt.savefig(OUT / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP salvo")
    return best, df, feats


if __name__ == "__main__":
    import sys
    if "--real" in sys.argv:
        # Coleta dados reais: python main.py --real --key SUA_KEY
        key_idx = sys.argv.index("--key") + 1 if "--key" in sys.argv else None
        key = sys.argv[key_idx] if key_idx else "SUA_KEY_AQUI"
        main(firms_api_key=key, demo_municipios=None)