import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from io import StringIO

# ======================================================
# PASSO 3 — NASA FIRMS (focos de incêndio — target)
# ======================================================

# ==== Verificação ===
# Primeiro verifica os dados salvos — se o arquivo final já existe, não faz nenhuma requisição. 
def get_firms_municipios(gdf: gpd.GeoDataFrame, out, api_key: str = None) -> pd.DataFrame:
    """
    Retorna contagem mensal de focos por município.

    Com api_key:
        Baixa do NASA FIRMS via API (VIIRS S-NPP).
    """
    out_file = out / "firms_mensal.csv"
    if out_file.exists():
        print("FIRMS já processado — carregando cache")
        return pd.read_csv(out_file, parse_dates=["data"])

    if api_key and api_key != "SUA_KEY_AQUI":
        return _firms_real(gdf, api_key, out_file)

# === Configuração da requisição ===
def _firms_real(gdf, api_key, out_file):
    # O BBOX define o retângulo geográfico do Pará no formato lon_min, lat_min, lon_max, lat_max. 
    # O VIIRS_SNPP_SP é o sensor escolhido — o Suomi NPP com dados consolidados históricos (SP = Standard Product)
    BBOX = "-58.5,-9.9,-46.0,2.7"

    # FIRMS Archive Download — sem limite de registros
    BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    SOURCE = "VIIRS_SNPP_SP"

    dfs = []
    # Período: 2019–2025
    for ano in range(2019, 2026):
        # Download dividido por trimestre para contornar limite de registros
        # A API do FIRMS tem um limite de registros por requisição. Pedir o ano inteiro de uma vez pode truncar os dados silenciosamente
        trimestres = [
            (f"{ano}-01-01", f"{ano}-03-31"),
            (f"{ano}-04-01", f"{ano}-06-30"),
            (f"{ano}-07-01", f"{ano}-09-30"),
            (f"{ano}-10-01", f"{ano}-12-31"),
        ]
        for data_ini, data_fim in trimestres:
            url = f"{BASE}/{api_key}/{SOURCE}/{BBOX}/1/{data_ini}/{data_fim}"
            try:
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                
                df_t = pd.read_csv(StringIO(r.text))
                print(f" {ano} {data_ini[5:7]}-{data_fim[5:7]}: {len(df_t):,} focos")
                if len(df_t) > 0:
                    dfs.append(df_t)
                time.sleep(1)
            except Exception as e:
                print(f"    Erro {ano} {data_ini}: {e}")

    if not dfs:
        raise RuntimeError("Nenhum dado FIRMS baixado.")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["acq_date"] = pd.to_datetime(df_all["acq_date"])
    print(f"  Total focos brutos: {len(df_all):,}")

    # Salvando dados brutos nte da limpeza
    bruto_file = out_file.parent / "firms_bruto.csv"
    df_all.to_csv(bruto_file, index=False)
    print(f"  Bruto salvo: {bruto_file}")

    # ==== Limpeza dos focos ====
    # Dois filtros aplicados: 
    # Confiança: o VIIRS classifica cada detecção em "l" (low), "n" (nominal) ou "h" (high).
    # Detecções low podem ter alta taxa de falso positivo, como refelxo solares
    # FRP (Fire Radiative Power): mede a energia irradiada pelo fogo em megawatts. 
    # Queimadas vegetais têm FRP intermitente e raramente ultrapassam 500 MW. Acima disso pode ser calor industrial.
    df_all = df_all[df_all["confidence"] != "l"]
    # df_all = df_all[df_all["frp"] <= 500]
    print(f"  Focos após limpeza (confiança): {len(df_all):,}")

    #Salva os dados depois da limpeza
    limpo_file = out_file.parent / "firms_limpo.csv"
    df_all.to_csv(limpo_file, index=False)
    print(f"  Limpo salvo: {limpo_file}")

    # Spatial join e agregação mensal
    # O FIRMS retorna coordenadas brutas — cada foco é um ponto com latitude e longitude. 
    # Esse bloco transforma esses pontos em geometria (gpd.points_from_xy) e faz o spatial join: verifica qual polígono municipal cada ponto está dentro (predicate="within"). 
    gdf_mun = gdf[["codigo_ibge", "geometry"]].copy().to_crs("EPSG:4326")
    gdf_focos = gpd.GeoDataFrame( df_all, geometry=gpd.points_from_xy(df_all["longitude"], df_all["latitude"]), crs="EPSG:4326")
    joined = gpd.sjoin(gdf_focos, gdf_mun, how="inner", predicate="within")
    print(f"  Focos após spatial join: {len(joined):,}")

    # Cada foco tem uma data diária (acq_date). O .to_period("M") trunca para o mês (2022-10-15 = 2022-10) e .to_timestamp() converte de volta para datetime (2022-10-01), padronizando todas as datas do mês para o primeiro dia. 
    joined["data"] = joined["acq_date"].dt.to_period("M").dt.to_timestamp()
    mensal = joined.groupby(["codigo_ibge", "data"]).size().reset_index(name="focos")
    print(f"  {len(mensal):,} registros mensais | {mensal['focos'].sum():,} focos totais")

    mensal.to_csv(out_file, index=False)
    return mensal