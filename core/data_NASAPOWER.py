import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import date


# ===============================================
# PASSO 2 — NASA POWER (variáveis climáticas)
# ===============================================

# Variáveis disponíveis no NASA POWER:
# T2M      = temperatura a 2 m (°C)
# PRECTOTCORR = precipitação corrigida (mm/dia)
# RH2M     = umidade relativa a 2 m (%)
# WS2M     = velocidade do vento a 2 m (m/s)
# ALLSKY_SFC_SW_DWN = radiação solar (kWh/m²/dia)
# T2M_MAX  = temperatura máxima diária
# T2M_MIN  = temperatura mínima diária

# Variáveis pedidas para API - cada uma se torna uma coluna
NASA_POWER_VARS = [
    "T2M", "T2M_MAX", "PRECTOTCORR",
    "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN"
]

# ==== Calculando Data ====
# O NASA POWER tem um atraso de processamento — dados dos últimos meses ainda não estão disponíveis. 
# Essa função subtrai 6 meses da data atual para garantir que só peça dados que já existem. 
# O while month <= 0 trata a virada de ano: se hoje é março (mês 3) e subtrai 6, o resultado seria mês -3, que não existe — o loop corrige para outubro do ano anterior.
def safe_end_date(lag_months: int = 12) -> str:
    """
    Retorna o ano seguro para o endpoint monthly do NASA POWER.
    Formato: "YYYY"
    """
    today = date.today()
    month = today.month - lag_months
    year  = today.year
    while month <= 0:
        month += 12
        year  -= 1
    return str(year)

# ==== Requisição por município ====
# Monta a URL para um único ponto geográfico , recebendo o lat e lon calculados no data_IBGE
# O community=re indica a comunidade "Renewable Energy", que libera todas as variáveis usadas. 
# O temporal/monthly/point retorna médias mensais usados na aplicação.
def get_nasa_power(lat: float, lon: float, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    params = ",".join(NASA_POWER_VARS)
    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters={params}"
        f"&community=re"
        f"&longitude={lon:.4f}&latitude={lat:.4f}"
        f"&start={start}&end={end}"
        f"&format=JSON"
    )

    # ==== Tratamento de erros e retentativas ====
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=45)

            if r.status_code == 429:
                wait = 10 * attempt
                print(f"    rate-limit (429) — aguardando {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code == 422:
                try:
                    body = r.json()
                    msgs = body.get("messages", body.get("errors", r.text[:300]))
                except Exception:
                    msgs = r.text[:300]
                print(f"    422 — {msgs}")
                return pd.DataFrame()   # erro de parâmetro, sem retry

            r.raise_for_status()
            data = r.json()

            # Passing da resposta JSON
            # Resposta: {"properties": {"parameter": {"T2M": {"YYYYMM": val}}}}
            records = {}
            for var, monthly in data["properties"]["parameter"].items():
                for key, val in monthly.items():
                    # O loop inverte essa estrutura — em vez de organizar por variável -> mês, 
                    # organiza por mês -> variáveis, que é o formato que o pandas espera para virar linhas de uma tabela. 
                    # Ignora chave "YYYYMM" onde MM > 12 (ex: "202013" = média anual)
                    if len(key) == 6 and int(key[4:6]) > 12:
                        continue
                    if key not in records:
                        records[key] = {}
                    records[key][var] = val if val != -999.0 else np.nan

            # Converte o dicionário para DataFrame onde cada linha é um mês e cada coluna é uma variável climática.
            df = pd.DataFrame.from_dict(records, orient="index")
            df.index = pd.to_datetime(df.index, format="%Y%m")
            df.index.name = "data"
            return df

        except requests.exceptions.Timeout:
            print(f"Timeout (tentativa {attempt}/{retries})")
        except requests.exceptions.ConnectionError as e:
            print(f"Erro de conexão: {e}")
            break
        except Exception as e:
            print(f"Erro inesperado (tentativa {attempt}/{retries}): {e}")
        time.sleep(2 * attempt)

    return pd.DataFrame()


# ==== Coleta em lote com checkpoint ====
def collect_nasa_power(gdf: gpd.GeoDataFrame, out, start: str = "2020", end: str = None) -> pd.DataFrame:
    if end is None:
        end = safe_end_date(lag_months=12)
        print(f"  Data fim calculada: {end}")
    out_file  = out / "nasa_power_para.csv"
    ckpt_file = out / "nasa_power_para_checkpoint.csv"

    # Antes de qualquer coisa verifica se já existe o arquivo final — se sim, pula toda a coleta.
    if out_file.exists():
        print("  NASA POWER ja coletado - carregando cache")
        return pd.read_csv(out_file, parse_dates=["data"])

    ja_coletados = set()
    all_dfs = []
    if ckpt_file.exists():
        df_ckpt = pd.read_csv(ckpt_file, parse_dates=["data"])
        all_dfs.append(df_ckpt)
        ja_coletados = set(df_ckpt["codigo_ibge"].unique())
        print(f"  Retomando checkpoint: {len(ja_coletados)} municipios ja coletados")

    pendentes = gdf[~gdf["codigo_ibge"].isin(ja_coletados)]
    print(f"Coletando NASA POWER: {len(pendentes)} municipios restantes ({start}-{end})")

    erros = []
    for i, (_, row) in enumerate(pendentes.iterrows(), 1):
        df_clima = get_nasa_power(row["lat"], row["lon"], start, end)
        if not df_clima.empty:
            df_clima["codigo_ibge"] = row["codigo_ibge"]
            df_clima["municipio"]   = row["municipio"]
            df_clima = df_clima.reset_index()
            all_dfs.append(df_clima)
        else:
            erros.append(row["codigo_ibge"])

        # A cada 20 municípios salva um checkpoint. 
        # Se a execução cair no meio (queda de internet, por exemplo), na próxima vez o código lê o checkpoint e retoma de onde parou
        # Evita ter que baixar tudo de novo.
        if i % 20 == 0 or i == len(pendentes):
            pct = (len(ja_coletados) + i) / len(gdf) * 100
            print(f"  {len(ja_coletados)+i}/{len(gdf)} ({pct:.0f}%) - erros: {len(erros)}")
            if all_dfs:
                pd.concat(all_dfs, ignore_index=True).to_csv(ckpt_file, index=False)

        time.sleep(0.4)

    if not all_dfs:
        raise RuntimeError(
            "\nNenhum dado coletado do NASA POWER.\n"
            "Possiveis causas:\n"
            "  1. Sem conexao com a internet\n"
            "  2. API do NASA POWER fora do ar (https://power.larc.nasa.gov)\n"
            "  3. Parametros de lat/lon invalidos\n"
            "Teste manualmente:\n"
            "  curl 'https://power.larc.nasa.gov/api/temporal/monthly/point"
            "?parameters=T2M&community=AG&longitude=-49.0&latitude=-5.0"
            f"&start={start}&end={end}&format=JSON'"
        )

    df_power = pd.concat(all_dfs, ignore_index=True)
    if erros:
        print(f"  {len(erros)} municipios sem dados (nao afetam o modelo)")
    df_power.to_csv(out_file, index=False)
    if ckpt_file.exists():
        ckpt_file.unlink()
    print(f"  Salvo: {out_file}  ({len(df_power):,} linhas)")
    return df_power