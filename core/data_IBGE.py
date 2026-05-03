import time
import requests
import pandas as pd
import geopandas as gpd

# =====================================
# PASSO 1 — Municípios do Pará (IBGE)
# =====================================

# Escopo restrito ao Pará — alinhado ao artigo (144 municípios)
PARA_UF = {"15": "Pará"}

def get_municipios_para() -> gpd.GeoDataFrame:
    """
    Baixa os 144 municípios do Pará via API do IBGE.
    """
    gdfs = []
    total = len(PARA_UF)

    # ==== Requisição da API do IBGE ====
    # Obtenção dos dados e tratamentos de erros
    for i, (cod_uf, nome_uf) in enumerate(PARA_UF.items(), 1):
        print(f"  Baixando {nome_uf} ({i}/{total})...")
        url = (
            f"https://servicodados.ibge.gov.br/api/v3/malhas/estados/{cod_uf}"
            f"?formato=application/json&qualidade=minima&intrarregiao=municipio"
        )
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 422:
                try:
                    msg = r.json().get("errors", r.text[:200])
                except Exception:
                    msg = r.text[:200]
                print(f" 422 — parametros invalidos: {msg}")
                return pd.DataFrame() 
            r.raise_for_status()
            gdf_uf = gpd.read_file(r.text)
            gdf_uf["estado"] = nome_uf
            gdf_uf["uf"] = cod_uf
            gdfs.append(gdf_uf)
            time.sleep(0.3)
        except Exception as e:
            print(f"Erro {nome_uf}: {e}")

    # ==== Processamento do GeoDataFrame ====
    # Usamos o EPSG:4326 como sistema de coordenada
    # Calcula o centroide para cada polígono muniipal e extrai as coordenadas como colunas separadas 
    gdf = pd.concat(gdfs, ignore_index=True)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")
    gdf["centroid"] = gdf.geometry.centroid
    gdf["lon"] = gdf["centroid"].x
    gdf["lat"] = gdf["centroid"].y

    # ==== Padronização ====
    # A API do IBGE pode retornar o código do município em colunas com nomes diferentes dependendo da versão. 
    # O loop testa os nomes mais comuns e usa o primeiro que encontrar.
    # .zfill(7) garante que o código sempre tenha 7 dígitos
    for col in ["codarea", "CD_MUN", "id", "code"]:
        if col in gdf.columns:
            gdf["codigo_ibge"] = gdf[col].astype(str).str.zfill(7)
            break
    if "codigo_ibge" not in gdf.columns:
        gdf = gdf.reset_index()
        gdf["codigo_ibge"] = gdf["uf"] + gdf.index.astype(str).str.zfill(5)

    for col in ["NM_MUN", "name", "nome"]:
        if col in gdf.columns:
            gdf = gdf.rename(columns={col: "municipio"})
            break
    if "municipio" not in gdf.columns:
        gdf["municipio"] = gdf["codigo_ibge"]

    print(f"{len(gdf)} municípios do Pará carregados")
    
    return gdf[["codigo_ibge", "municipio", "estado", "uf", "lat", "lon", "geometry"]]
