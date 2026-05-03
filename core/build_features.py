import pandas as pd

# =========================
# PASSO 4 — Build Features
# =========================
NASA_POWER_VARS = [
    "T2M", "T2M_MAX", "PRECTOTCORR",
    "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN"
]

def build_features(df_power: pd.DataFrame, df_firms: pd.DataFrame, foco_threshold_pct: float = 0.75, train_cutoff="2023-01-01") -> pd.DataFrame:
    # Padronização inicial
    # Garante que o código do município tem exatamente 7 dígitos nos dois DataFrames antes de qualquer join.
    df_power = df_power.copy()
    df_firms = df_firms.copy()
    df_power["codigo_ibge"] = df_power["codigo_ibge"].astype(str).str.zfill(7)
    df_firms["codigo_ibge"] = df_firms["codigo_ibge"].astype(str).str.zfill(7)

    # União das fontes do NASA POWER e FIRMS por meio das chaves município e mês
    if "focos" not in df_power.columns:
        df = df_power.merge(df_firms[["codigo_ibge", "data", "focos"]], on=["codigo_ibge", "data"], how="left").fillna({"focos": 0})
    else:
        df = df_power.copy()

    df = df.sort_values(["codigo_ibge", "data"])

    # Lags de focos - cria três colunas com o histórico de focos dos meses anteriores.
    # O .shift(lag) desloca a série para baixo — para cada linha o modelo vê o que aconteceu 1, 2 e 3 meses atrás naquele município.
    for lag in [1, 2, 3]:
        df[f"focos_lag{lag}"] = df.groupby("codigo_ibge")["focos"].shift(lag)

    # Médias móveis dos focos - calculamos a média dos últimos 3 meses de focos.
    df["focos_media3m"] = ( df.groupby("codigo_ibge")["focos"].transform(lambda x: x.shift(1).rolling(3).mean()))

    #  Lags das variáveis climáticas - mesma lógica dos lags de focos, mas para cada variável climática. 
    for var in NASA_POWER_VARS:
        if var in df.columns:
            df[f"{var}_lag1"] = df.groupby("codigo_ibge")[var].shift(1)

    #  Acumulado de precipitação - soma a precipitação dos 3 meses anteriores. 
    if "PRECTOTCORR" in df.columns:
        df["precip_acum3m"] = (df.groupby("codigo_ibge")["PRECTOTCORR"].transform(lambda x: x.shift(1).rolling(3).sum()))

    # Adicionando duas colunas numéricas que capturam o padrão sazonal. 
    # O modelo aprende que agosto–outubro (estação seca na Amazônia) tem comportamento diferente de janeiro–março
    df["mes"]       = df["data"].dt.month
    df["trimestre"] = df["data"].dt.quarter

    #  TARGET: focos no próximo mês acima do limiar 
    print(f"  Focos — min: {df['focos'].min()}, max: {df['focos'].max()}, "
          f"p50: {df['focos'].quantile(0.5):.1f}, p75: {df['focos'].quantile(0.75):.1f}")

    # === Definição do target ===
    # Definição do que o modelo vai aprender e prever
    # threshold = df["focos"].quantile(foco_threshold_pct)
    mask_treino = df["data"] < train_cutoff
    threshold = df.loc[mask_treino, "focos"].quantile(foco_threshold_pct)

    # O shift(-1) desloca os focos para cima, fazendo com que cada linha veja o que vai acontecer no mês seguinte. 
    # Combinado com as features que olham para o passado, o modelo aprende: "dado o histórico e o clima de hoje, haverá foco relevante no próximo mês?"
    # O threshold é calculado por município usando apenas dados de treino — isso resolve dois problemas de uma vez. 
    # Primeiro, evita data leakage (o threshold não usa dados de 2023 em diante). Segundo, respeita a heterogeneidade do Pará: 9 focos pode ser muito para um município pequeno e pouco para um município grande como Altamira.
    threshold_por_mun = ( df[df["data"] < train_cutoff].groupby("codigo_ibge")["focos"].quantile(0.75))
    df["threshold_mun"] = df["codigo_ibge"].map(threshold_por_mun).fillna(1)
    df["target"] = (df.groupby("codigo_ibge")["focos"].shift(-1) > df["threshold_mun"]).astype(int)

    print(f"  Threshold final (p{foco_threshold_pct*100:.0f}): {threshold:.1f} focos")
    print(f"  Classe positiva: {df['target'].mean()*100:.1f}% das amostras")
    print(f"  Distribuição target: {df['target'].value_counts().to_dict()}")

    # === Remoção de NaNs ===
    #Os lags e médias móveis geram NaN nas primeiras linhas de cada município — os 3 primeiros meses não têm histórico suficiente para preencher focos_lag3 ou focos_media3m. 
    # O dropna() remove essas linhas.
    n_antes = len(df)
    df = df.dropna()
    print(f"  dropna: removidas {n_antes - len(df)} linhas ({(n_antes-len(df))/n_antes*100:.1f}%)")

    return df