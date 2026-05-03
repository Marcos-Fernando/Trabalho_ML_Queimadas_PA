import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb

# ================================
# PASSO 5 — Treino e Avaliação
# ================================

# Definindo exatamente quais colunas entram nos modelos
FEATURES = [
    # Focos históricos
    "focos_lag1", "focos_lag2", "focos_lag3", "focos_media3m",
    # Clima (mês atual e anterior)
    "T2M", "T2M_MAX", "PRECTOTCORR", "RH2M", "WS2M",
    "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DWN_lag1",
    "T2M_lag1", "T2M_MAX_lag1", "PRECTOTCORR_lag1",
    "RH2M_lag1", "WS2M_lag1",
    "precip_acum3m",
    # Sazonalidade
    "mes", "trimestre",
]

# === Divisão temporal ===
# Divide os dados em três fatias cronológicas para evitar dataleakage, o modelo "não vê o futuro" durante o aprendizado
def split_temporal(df: pd.DataFrame):
    """
    Divisão CRONOLÓGICA — nunca aleatória em séries temporais.
    Alinhada ao período do artigo (2019–2025):
      Treino    : 2019–2022
      Validação : 2023
      Teste     : 2024–2025
    """
    train = df[df["data"] <  "2023-01-01"]
    val   = df[(df["data"] >= "2023-01-01") & (df["data"] < "2024-01-01")]
    test  = df[df["data"] >= "2024-01-01"]

    feats_disponiveis = [f for f in FEATURES if f in df.columns]

    def xy(split):
        return split[feats_disponiveis], split["target"]

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = xy(train), xy(val), xy(test)

    print(f"  Treino : {len(train):,} amostras  ({train['data'].min().date()} – {train['data'].max().date()})")
    print(f"  Val    : {len(val):,} amostras  ({val['data'].min().date()} – {val['data'].max().date()})")
    print(f"  Teste  : {len(test):,} amostras  ({test['data'].min().date()} – {test['data'].max().date()})")

    print(f"  Distribuição target — treino: {y_tr.value_counts().to_dict()}")
    print(f"  Distribuição target — val:    {y_val.value_counts().to_dict()}")

    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te), feats_disponiveis


def train_models(X_train, y_train, X_val, y_val):
    """Treina Random Forest e XGBoost."""
    # Verifica se o threshold ficou alto demais e nenhum município foi classificado como positivo, senão os modelos gerariam erros
    if y_train.nunique() < 2:
        raise ValueError(
            f"Treino tem só uma classe: {y_train.unique()}. "
            f"Ajuste o threshold em build_features (atual p75)."
        )
    if y_val.nunique() < 2:
        raise ValueError(
            f"Validação tem só uma classe: {y_val.unique()}. "
            f"Tente aumentar o período de dados ou reduzir o threshold."
        )

    # Calcula a proporção entre negativos e positivos no treino.
    scale = (y_train == 0).sum() / (y_train == 1).sum()

    # Random Forest 
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    auc_rf = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    print(f"  Random Forest  AUC-ROC (val): {auc_rf:.3f}")

    # Extra Trees 
    et = ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
    et.fit(X_train, y_train)
    auc_et = roc_auc_score(y_val, et.predict_proba(X_val)[:, 1])
    print(f"  Extra Trees    AUC-ROC (val): {auc_et:.3f}")

    # XGBoost 
    # learning_rate 0.05 - aprendizado lento e mais estavel / subsample e colsample 0.8 = usar 80% das linhas e features por aŕvores / early_stopping 30 - para o treino se auc não melhorar em 30 rodadas
    xgb_model = xgb.XGBClassifier( n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale, eval_metric="aucpr", early_stopping_rounds=30, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Calibração de probabilidade — corrige a compressão causada pelo desbalanceamento
    xgb_model = CalibratedClassifierCV(xgb_model, method="isotonic", cv="prefit")
    xgb_model.fit(X_val, y_val)

    auc_xgb = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])
    print(f"  XGBoost AUC-ROC (val): {auc_xgb:.3f}")

    # Precision-Recall = Mostra dois pontos da curva para cada modelo. 
    print("\n  Análise Precision-Recall (validação):")
    for nome, modelo in [("Random Forest", rf), ("Extra Trees", et),("XGBoost", xgb_model)]:
        prec, rec, thresholds = precision_recall_curve(y_val, modelo.predict_proba(X_val)[:, 1])
        idx = np.argmin(np.abs(prec[:-1] - rec[:-1]))
        print(f"  {nome}:")
        print(f" Threshold padrão (0.5) - precision: {prec[np.searchsorted(thresholds, 0.5)]:.3f} "
              f"| recall: {rec[np.searchsorted(thresholds, 0.5)]:.3f}")
        print(f" Threshold ótimo ({thresholds[idx]:.3f}) - precision: {prec[idx]:.3f} "
              f"| recall: {rec[idx]:.3f}")

    # Seleção do melhor modelo 
    aucs = {"Random Forest": auc_rf, "Extra Trees": auc_et, "XGBoost": auc_xgb}
    modelos = {"Random Forest": rf, "Extra Trees": et, "XGBoost": xgb_model}

    melhor_nome = max(aucs, key=aucs.get)
    best = modelos[melhor_nome]
    print(f"\nMelhor modelo: {melhor_nome} (AUC: {aucs[melhor_nome]:.3f})")

    return rf, xgb_model, best

def avaliar_com_tscv(df: pd.DataFrame, n_splits: int = 3):
    """
    Avaliação com validação cruzada temporal.
    """
    # Ordena cronologicamente e indexa por data
    df = df.sort_values("data").reset_index(drop=True)

    feats_disponiveis = [f for f in FEATURES if f in df.columns]
    X = df[feats_disponiveis]
    y = df["target"]

    # O TimeSeriesSplit divide os dados em folds mantendo a ordem cronológica — cada fold tem mais dados de treino que o anterior, e a validação sempre está no futuro em relação ao treino.
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []

    for fold, (idx_tr, idx_val) in enumerate(tscv.split(X), 1):
        X_tr, y_tr   = X.iloc[idx_tr],  y.iloc[idx_tr]
        X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]

        data_ini = df["data"].iloc[idx_tr[0]].date()
        data_fim = df["data"].iloc[idx_tr[-1]].date()
        data_val = df["data"].iloc[idx_val[-1]].date()
        print(f"  Fold {fold}: treino {data_ini}-{data_fim} | val até {data_val}")

        # ===== Verifica separação por meses completos ====
        ultimo_treino  = df["data"].iloc[idx_tr[-1]].to_period("M")
        primeiro_val   = df["data"].iloc[idx_val[0]].to_period("M")
        assert ultimo_treino != primeiro_val, (
            f"Fold {fold}: vazamento detectado — treino e validação "
            f"compartilham o mês {ultimo_treino}"
        )
        # 

        scale = (y_tr == 0).sum() / (y_tr == 1).sum()
        modelo = xgb.XGBClassifier( n_estimators=300, max_depth=5, scale_pos_weight=scale, eval_metric="auc", random_state=42, verbosity=0)
        modelo.fit(X_tr, y_tr)

        auc = roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])
        aucs.append(auc)
        print(f"    AUC-ROC: {auc:.3f}")

    print(f"\n  AUC médio: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    return aucs