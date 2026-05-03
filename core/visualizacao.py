import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =========================
# PASSO 6 — Visualizações
# =========================

# O hasattr verifica se o modelo tem o atributo feature_importances_ antes de tentar acessá-lo. Isso é necessário porque o XGBoost calibrado com CalibratedClassifierCV encapsula o modelo original
def plot_feature_importance(model, feature_names: list, title: str, out):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
    else:
        return
    imp = imp.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="#e74c3c", edgecolor="white")
    ax.set_title(f"Importância das Features — {title}", fontsize=13)
    ax.set_xlabel("Importância")
    plt.tight_layout()
    plt.savefig(out / f"feature_importance_{title.lower().replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"feature_importance salvo")

# Recebe uma probabilidade contínua (saída do ML) e classifica em uma das quatro categorias. 
# Os três perfis b1, b2, b3 são os limiares entre as categorias — um município precisa "superar" cada perfil para subir de categoria.
def electre_trib_prob(prob: float, b1: float, b2: float, b3: float, q: float, p: float, v: float, lam: float = 0.65) -> str:
    """
    ELECTRE Tri-B — Regra Pessimista (Roy 1996).
    Atribui a categoria mais alta cujo perfil é superado pela alternativa.
    """
    profiles = [b1, b2, b3]
    labels   = ["Low", "Moderate", "High", "Critical"]

    def concordance(a, b):
        diff = b - a
        if diff <= q: return 1.0
        if diff >= p: return 0.0
        return (p - diff) / (p - q)

    def discordance(a, b):
        diff = b - a
        if diff <= p: return 0.0
        if diff >= v: return 1.0
        return (diff - p) / (v - p)

    def credibility(a, b):
        c = concordance(a, b)
        d = discordance(a, b)
        if d > c:
            return c * (1 - d) / (1 - c) if c < 1 else 0.0
        return c

    # Regra pessimista: categoria = maior bk superado por prob
    categoria = 0   # default: Low
    for k, bk in enumerate(profiles):
        if credibility(prob, bk) >= lam:
            categoria = k + 1   # supera bk - sobe uma categoria

    return labels[categoria]

def plot_mapa_geral(df, model, feature_names, gdf, out, calibrar_perfis=False, b1=0.25, b2=0.50, b3=0.75):
    feats = [f for f in feature_names if f in df.columns]
    df = df.copy()
    df["prob_foco"] = model.predict_proba(df[feats])[:, 1]
    prob_media = df.groupby("codigo_ibge")["prob_foco"].mean().reset_index()

    pmin = prob_media["prob_foco"].min()
    p25  = prob_media["prob_foco"].quantile(0.25)
    p50  = prob_media["prob_foco"].quantile(0.50)
    p75  = prob_media["prob_foco"].quantile(0.75)
    pmax = prob_media["prob_foco"].max()

    print(f"  Probabilidades — min: {pmin:.3f} | p25: {p25:.3f} "
          f"| p50: {p50:.3f} | p75: {p75:.3f} | max: {pmax:.3f}")

    if calibrar_perfis:
        rang = pmax - pmin
        b1 = pmin + rang * 0.30
        b2 = pmin + rang * 0.55
        b3 = pmin + rang * 0.88

    # Escala q, p, v proporcionalmente ao range real
    rang = pmax - pmin
    q = rang * 0.10   # indiferença: 10% do range
    p = rang * 0.20   # preferência: 20% do range
    v = rang * 0.60   # veto:        60% do range

    print(f"  Perfis  — b1: {b1:.3f} | b2: {b2:.3f} | b3: {b3:.3f}")
    print(f"  Limiares — q: {q:.3f} | p: {p:.3f} | v: {v:.3f}")

    prob_media["classe"] = prob_media["prob_foco"].apply(
        lambda prob: electre_trib_prob(prob, b1=b1, b2=b2, b3=b3, q=q, p=p, v=v)
    )

    dist = prob_media["classe"].value_counts()
    print(f"  Distribuição ELECTRE Tri-B:")
    for cat in ["Low", "Moderate", "High", "Critical"]:
        n = dist.get(cat, 0)
        print(f"    {cat:10s}: {n:4d} ({n/len(prob_media)*100:.1f}%)")

    gdf_plot = gdf.merge(prob_media, on="codigo_ibge", how="left")

    cores = {
        "Low"     : "#2ecc71",
        "Moderate": "#f39c12",
        "High"    : "#e74c3c",
        "Critical": "#8e44ad",
    }

    fig, ax = plt.subplots(figsize=(12, 10))
    for classe, cor in cores.items():
        subset = gdf_plot[gdf_plot["classe"] == classe]
        if not subset.empty:
            subset.plot(ax=ax, color=cor, linewidth=0.3, edgecolor="white")

    gdf.dissolve().boundary.plot(ax=ax, color="black", linewidth=0.8)

    
    patches = [mpatches.Patch(color=c, label=l) for l, c in cores.items()]
    ax.legend(handles=patches, title="Risk Class (ELECTRE Tri-B)",
              loc="lower right", fontsize=11)

    ax.set_title(
        "Risco de Queimadas — Pará (2019–2025)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out / "mapa_risco_electre_trib.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(" Mapa ELECTRE Tri-B salvo")