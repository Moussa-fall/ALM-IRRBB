"""
===============================================================================
 Gestion actif-passif (ALM) d'une banque de détail : risque de taux (IRRBB)
===============================================================================
 Auteur  : Moussa FALL, M2 Banque Finance, Université Lumière Lyon 2
 Version : 1.0 (octobre 2026)

 Le programme modélise le bilan simplifié d'une banque de détail qui prête à taux fixe
 (crédits immobiliers) et se finance à court terme (dépôts à vue, interbancaire 3 mois),
 puis mesure et couvre son risque de taux en cinq étapes :

   1. Échéancier du portefeuille de crédits et effet des remboursements anticipés (CPR)
   2. Gap de taux statique et sensibilité de la marge nette d'intérêt (MNI)
   3. Sensibilité de la valeur économique des fonds propres (ΔEVE) aux 6 chocs
      réglementaires (Comité de Bâle / EBA) et test des valeurs extrêmes (SOT)
   4. Couverture par swap payeur de taux fixe : nominal admissible et coût en MNI
   5. Remboursements anticipés dépendants des taux : convexité négative et limite
      d'une couverture par swap seul

 Lancement : python alm_irrbb.py
 Sorties   : rapport dans la console + graphiques PNG dans le dossier resultats/
 Dépendances : numpy, pandas, matplotlib

 AVERTISSEMENT : portefeuille fictif, courbe des taux ILLUSTRATIVE et paramètres
 comportementaux non calibrés. Le programme illustre une méthode, il ne décrit
 aucune banque réelle.
===============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0. PARAMÈTRES
# =============================================================================
PARAMS = {
    # Portefeuille de crédits
    "n_credits": 500,
    "graine": 42,
    # Passif (en % de l'encours de crédits)
    "part_fonds_propres": 0.08,
    "part_dav": 0.40,
    # Convention d'écoulement des dépôts à vue (DAV)
    "dav_part_stable": 0.70,
    "dav_duree_stable_ans": 5,
    # Remboursements anticipés
    "cpr_fixe": 0.05,
    "cpr_comportemental": {"cpr_min": 0.04, "cpr_max": 0.30, "centre": 0.010,
                           "pente": 400.0, "marge": 0.004},
    # Test des valeurs extrêmes : seuil réglementaire et cible interne
    "seuil_sot": -0.15,
    "cible_interne": -0.10,
    # Calibrage des chocs EUR retenu pour le rapport
    "calibrage": "Bâle 2016",
}

# Chocs EUR (taux décimaux). Bâle 2016 = calibrage repris par l'EBA.
# Recalibrage Bâle publié le 16/07/2024, applicable au 01/01/2026 : vérifier sa
# transposition dans l'UE avant de l'utiliser comme référence.
CALIBRAGES = {
    "Bâle 2016": {"par": 0.0200, "court": 0.0250, "long": 0.0100},
    "Bâle 2024": {"par": 0.0225, "court": 0.0350, "long": 0.0200},
}

MOIS = pd.RangeIndex(1, 361)                     # horizon de 30 ans, pas mensuel
BUCKETS = [3, 6, 12, 24, 36, 60, 84, 120, 180, 240, 360]
LABELS = ["0-3M", "3-6M", "6-12M", "1-2A", "2-3A", "3-5A",
          "5-7A", "7-10A", "10-15A", "15-20A", "20A+"]

DOSSIER = Path(__file__).resolve().parent / "resultats"

# Palette (validée daltonisme, fond clair) : bleu, orange, aqua ; gris pour les repères
BLEU, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ENCRE, ENCRE_2, GRILLE = "#0b0b0b", "#52514e", "#e4e3df"


# =============================================================================
# 1. PORTEFEUILLE, ÉCHÉANCIER ET BILAN
# =============================================================================
def generer_portefeuille(n=PARAMS["n_credits"], graine=PARAMS["graine"]):
    """Portefeuille fictif de crédits immobiliers à taux fixe."""
    rng = np.random.default_rng(graine)
    duree_initiale = rng.choice([15, 20, 25], size=n) * 12
    anciennete = (rng.uniform(0, 0.6, size=n) * duree_initiale).astype(int)
    return pd.DataFrame({
        "crd": rng.uniform(80_000, 400_000, size=n).round(-2),     # capital restant dû
        "taux": rng.uniform(0.010, 0.040, size=n).round(4),        # taux fixe client
        "duree_restante": duree_initiale - anciennete,             # en mois
    })


def echeancier_credit(crd, taux_annuel, duree_restante, cpr=0.0):
    """
    Échéancier mensuel à mensualités constantes avec remboursements anticipés.
    CPR annuel converti en taux mensuel : SMM = 1 - (1 - CPR)^(1/12).
    Après un remboursement partiel, le client garde sa durée (sa mensualité baisse).
    """
    r, smm = taux_annuel / 12, 1 - (1 - cpr) ** (1 / 12)
    lignes = []
    for t in range(1, duree_restante + 1):
        n = duree_restante - t + 1
        mensualite = crd * r / (1 - (1 + r) ** -n)
        interets = crd * r
        amort = mensualite - interets
        anticipe = (crd - amort) * smm
        crd = crd - amort - anticipe
        lignes.append((t, interets, amort, anticipe, crd))
    return pd.DataFrame(lignes, columns=["mois", "interets", "amort", "anticipe", "crd_fin"])


def echeancier_portefeuille(pf, cprs):
    """Échéancier agrégé ; cprs = un CPR par crédit."""
    ech = pd.concat([echeancier_credit(c.crd, c.taux, int(c.duree_restante), k)
                     for c, k in zip(pf.itertuples(), cprs)]).groupby("mois").sum()
    ech["capital"] = ech["amort"] + ech["anticipe"]
    ech["flux"] = ech["interets"] + ech["capital"]
    return ech.reindex(MOIS, fill_value=0.0)


def bilan_initial(pf):
    total = pf["crd"].sum()
    fp, dav = total * PARAMS["part_fonds_propres"], total * PARAMS["part_dav"]
    return pd.Series({"Fonds propres": fp, "Dépôts à vue": dav,
                      "Emprunt interbancaire 3M": total - fp - dav})


def duree_de_vie_moyenne(ech):
    return (ech.index * ech["capital"]).sum() / ech["capital"].sum() / 12


# =============================================================================
# 2. GAP DE TAUX
# =============================================================================
def par_bucket(serie):
    b = pd.cut(serie.index, bins=[0] + BUCKETS, labels=LABELS)
    return serie.groupby(b, observed=False).sum()


def ecoulement_dav(encours, part_stable, duree_ans):
    """Part volatile révisée à 1 mois, part stable écoulée linéairement."""
    n = int(duree_ans * 12)
    flux = pd.Series(0.0, index=MOIS)
    flux[1] += encours * (1 - part_stable)
    if part_stable > 0:
        flux.loc[1:n] += encours * part_stable / n
    return flux


def gap_de_taux(ech, passif, part_stable):
    emprunt = pd.Series(0.0, index=MOIS)
    emprunt[3] = passif["Emprunt interbancaire 3M"]
    dav = ecoulement_dav(passif["Dépôts à vue"], part_stable, PARAMS["dav_duree_stable_ans"])
    gap = pd.DataFrame({"Actifs": par_bucket(ech["capital"]),
                        "Passifs": par_bucket(emprunt) + par_bucket(dav)})
    gap["Gap"] = gap["Actifs"] - gap["Passifs"]
    gap["Gap cumulé"] = gap["Gap"].cumsum()
    return gap


def delta_mni_gap(gap, choc=0.02):
    """Approximation à 1 an : un montant révisé en milieu de bucket subit le choc le reste de l'année."""
    milieux = {"0-3M": 1.5, "3-6M": 4.5, "6-12M": 9}
    return sum(gap.loc[b, "Gap"] * choc * (12 - m) / 12 for b, m in milieux.items())


# =============================================================================
# 3. COURBE, CHOCS RÉGLEMENTAIRES ET EVE
# =============================================================================
def courbe_base(t):
    """Courbe zéro-coupon ILLUSTRATIVE (taux continus) : ~2,0 % court, ~2,9 % long."""
    return 0.020 + 0.009 * (1 - np.exp(-np.asarray(t, dtype=float) / 5))


def plancher_eba(t):
    """Plancher post-choc EBA : -150 pb à maturité 0, +3 pb par an, 0 % dès 50 ans."""
    return np.minimum(-0.015 + 0.0003 * t, 0.0)


def chocs(t, cal):
    """Les 6 scénarios standardisés du Comité de Bâle (IRRBB, 2016)."""
    t = np.asarray(t, dtype=float)
    court = cal["court"] * np.exp(-t / 4)
    long_ = cal["long"] * (1 - np.exp(-t / 4))
    return {
        "Parallèle +":   np.full_like(t, cal["par"]),
        "Parallèle -":   np.full_like(t, -cal["par"]),
        "Pentification": -0.65 * court + 0.90 * long_,
        "Aplatissement": +0.80 * court - 0.60 * long_,
        "Court terme +": court,
        "Court terme -": -court,
    }


def fonction_courbe(scenario, cal):
    """Renvoie la fonction t -> taux de la courbe choquée (ou de base si scenario=None)."""
    if scenario is None:
        return courbe_base

    def f(t):
        base = courbe_base(t)
        choquee = base + chocs(t, cal)[scenario]
        return np.maximum(choquee, np.minimum(base, plancher_eba(np.asarray(t, dtype=float))))
    return f


def valeur_actuelle(flux, f):
    t = flux.index.values / 12
    return float((flux.values * np.exp(-f(t) * t)).sum())


def flux_passif(passif, part_stable):
    emprunt = pd.Series(0.0, index=MOIS)
    emprunt[3] = passif["Emprunt interbancaire 3M"] * np.exp(courbe_base(0.25) * 0.25)
    dav = ecoulement_dav(passif["Dépôts à vue"], part_stable, PARAMS["dav_duree_stable_ans"])
    return emprunt + dav                     # DAV non rémunérés : flux de capital seuls


# =============================================================================
# 4. SWAP PAYEUR DE TAUX FIXE
# =============================================================================
def taux_swap(maturite):
    t = np.arange(1, maturite + 1)
    df = np.exp(-courbe_base(t) * t)
    return (1 - df[-1]) / df.sum()


def valeur_swap_payeur(nominal, maturite, K, f):
    """Jambe variable reçue N(1 - DF(T)) moins jambe fixe payée N.K.somme(DF)."""
    t = np.arange(1, maturite + 1, dtype=float)
    df = np.exp(-f(t) * t)
    return nominal * ((1 - df[-1]) - K * df.sum())


def intervalle_nominal(bilan, swap_unitaire, cible):
    """
    ΔEVE couverte = bilan + N x swap_unitaire (linéaire en N).
    Chaque scénario impose une borne ; renvoie (N_min, N_max) ou None si incompatible.
    """
    n_min, n_max = 0.0, np.inf
    for s in bilan.index:
        b, u = bilan[s], swap_unitaire[s]
        if u > 0:
            n_min = max(n_min, (cible - b) / u)
        elif u < 0:
            n_max = min(n_max, (cible - b) / u)
    return (n_min, n_max) if n_min <= n_max else None


# =============================================================================
# 5. REMBOURSEMENTS ANTICIPÉS COMPORTEMENTAUX
# =============================================================================
def cpr_comportemental(taux_credits, taux_10ans, p=PARAMS["cpr_comportemental"]):
    """Courbe en S de l'incitation = taux client - (taux 10 ans + marge)."""
    incitation = taux_credits - (taux_10ans + p["marge"])
    return p["cpr_min"] + (p["cpr_max"] - p["cpr_min"]) / (
        1 + np.exp(-p["pente"] * (incitation - p["centre"])))


# =============================================================================
# CALCULS
# =============================================================================
def executer():
    cal = CALIBRAGES[PARAMS["calibrage"]]
    scenarios = list(chocs([1.0], cal).keys())
    pf = generer_portefeuille()
    passif = bilan_initial(pf)
    tier1 = passif["Fonds propres"]
    n = len(pf)
    R = {"pf": pf, "passif": passif, "tier1": tier1, "scenarios": scenarios}

    # --- Étape 1 -------------------------------------------------------------
    R["ech"] = {c: echeancier_portefeuille(pf, np.full(n, c)) for c in [0.0, 0.05, 0.10]}
    ech5 = R["ech"][PARAMS["cpr_fixe"]]

    # --- Étape 2 -------------------------------------------------------------
    R["gap"] = {"A": gap_de_taux(ech5, passif, 0.0),
                "B": gap_de_taux(ech5, passif, PARAMS["dav_part_stable"])}
    R["dmni"] = {k: delta_mni_gap(g) for k, g in R["gap"].items()}

    # --- Étape 3 -------------------------------------------------------------
    pas_flux = flux_passif(passif, PARAMS["dav_part_stable"])

    def eve(flux_actif, f):
        return valeur_actuelle(flux_actif, f) - valeur_actuelle(pas_flux, f)

    eve0 = eve(ech5["flux"], courbe_base)
    R["eve0"] = eve0
    R["deve_fixe"] = pd.Series({s: (eve(ech5["flux"], fonction_courbe(s, cal)) - eve0) / tier1
                                for s in scenarios})
    R["deve_cal"] = {}
    for nom_cal, c in CALIBRAGES.items():
        R["deve_cal"][nom_cal] = pd.Series(
            {s: (eve(ech5["flux"], fonction_courbe(s, c)) - eve0) / tier1 for s in scenarios})

    # --- Étape 4 -------------------------------------------------------------
    def swap_unitaire(m):
        K = taux_swap(m)
        v0 = valeur_swap_payeur(1.0, m, K, courbe_base)
        return pd.Series({s: (valeur_swap_payeur(1.0, m, K, fonction_courbe(s, cal)) - v0) / tier1
                          for s in scenarios})

    R["swaps"] = {}
    for m in [5, 7, 10, 15]:
        u = swap_unitaire(m)
        iv = intervalle_nominal(R["deve_fixe"], u, PARAMS["cible_interne"])
        R["swaps"][m] = {"K": taux_swap(m), "unit": u, "intervalle": iv}
    m_ref = 10
    n_ref = R["swaps"][m_ref]["intervalle"][0]
    R["n_ref"], R["m_ref"] = n_ref, m_ref
    R["deve_couvert_fixe"] = R["deve_fixe"] + n_ref * R["swaps"][m_ref]["unit"]
    R["cout_portage"] = n_ref * (courbe_base(0.25) - taux_swap(m_ref))
    R["dmni_couvert"] = R["dmni"]["B"] + n_ref * 0.02 * 10.5 / 12

    # --- Étape 5 -------------------------------------------------------------
    cpr0 = cpr_comportemental(pf.taux.values, courbe_base(10.0))
    ech0 = echeancier_portefeuille(pf, cpr0)
    eve0_c = eve(ech0["flux"], courbe_base)
    R["cpr_base"] = np.average(cpr0, weights=pf.crd)
    cpr_moy, deve_c = {}, {}
    for s in scenarios:
        f = fonction_courbe(s, cal)
        cprs = cpr_comportemental(pf.taux.values, f(np.array([10.0]))[0])
        cpr_moy[s] = np.average(cprs, weights=pf.crd)
        deve_c[s] = (eve(echeancier_portefeuille(pf, cprs)["flux"], f) - eve0_c) / tier1
    R["cpr_scen"] = pd.Series(cpr_moy)
    R["deve_comp"] = pd.Series(deve_c)
    R["deve_couvert_comp"] = R["deve_comp"] + n_ref * R["swaps"][m_ref]["unit"]
    R["intervalle_comp_brut"] = {}
    for m in [7, 10]:
        u, b = R["swaps"][m]["unit"], R["deve_comp"]
        R["intervalle_comp_brut"][m] = (
            max((PARAMS["seuil_sot"] - b[s]) / u[s] for s in scenarios if u[s] > 0),
            min((PARAMS["seuil_sot"] - b[s]) / u[s] for s in scenarios if u[s] < 0))
    return R


# =============================================================================
# RAPPORT CONSOLE
# =============================================================================
def m(x, d=1):
    return f"{x / 1e6:,.{d}f} M€".replace(",", " ")


def pct(x):
    return f"{x:+.1%}"


def titre(txt):
    print("\n" + "=" * 78 + f"\n {txt}\n" + "=" * 78)


def rapport(R):
    pf, passif, tier1 = R["pf"], R["passif"], R["tier1"]
    seuil = PARAMS["seuil_sot"]

    titre("1. PORTEFEUILLE, BILAN ET ÉCOULEMENT")
    print(f"{len(pf)} crédits immobiliers à taux fixe | encours {m(pf.crd.sum())} | "
          f"taux moyen {np.average(pf.taux, weights=pf.crd):.2%} | "
          f"durée restante moyenne {np.average(pf.duree_restante, weights=pf.crd) / 12:.1f} ans")
    print("\nPassif :")
    for k, v in passif.items():
        print(f"  {k:<28}{m(v):>12}")
    print("\nDurée de vie moyenne de l'actif selon le CPR :")
    for c, ech in R["ech"].items():
        print(f"  CPR {c:>4.0%} : {duree_de_vie_moyenne(ech):.1f} ans")

    titre("2. GAP DE TAUX STATIQUE (CPR 5 %)")
    print("Convention A : DAV 100 % révisés à 1 mois | "
          f"Convention B : DAV {PARAMS['dav_part_stable']:.0%} stables sur {PARAMS['dav_duree_stable_ans']} ans\n")
    g = pd.concat({"A": R["gap"]["A"][["Gap", "Gap cumulé"]],
                   "B": R["gap"]["B"][["Gap", "Gap cumulé"]]}, axis=1) / 1e6
    print(g.round(1).to_string())
    print(f"\nΔMNI à 1 an, choc +200 pb : A = {m(R['dmni']['A'], 2)} | B = {m(R['dmni']['B'], 2)}")
    print(f"Contrôle : gap cumulé final = {m(R['gap']['B']['Gap cumulé'].iloc[-1])} "
          f"= fonds propres ({m(tier1)}), exclus du gap.")

    titre(f"3. ΔEVE / TIER 1 SOUS LES 6 CHOCS RÉGLEMENTAIRES (seuil SOT {seuil:.0%})")
    print(f"EVE de base : {m(R['eve0'])} | Tier 1 : {m(tier1)}\n")
    t3 = pd.DataFrame(R["deve_cal"]).map(pct)
    print(t3.to_string())
    pire = R["deve_fixe"]
    print(f"\nPire scénario ({PARAMS['calibrage']}) : {pire.idxmin()} {pct(pire.min())} "
          f"-> {'ALERTE SOT' if pire.min() < seuil else 'sous le seuil'}")

    titre(f"4. COUVERTURE PAR SWAP PAYEUR (cible interne {PARAMS['cible_interne']:.0%} du Tier 1)")
    for mat, s in R["swaps"].items():
        iv = s["intervalle"]
        txt = (f"nominal admissible {m(iv[0])} à {m(iv[1])}" if iv
               else "AUCUN nominal admissible")
        print(f"  Swap {mat:>2} ans | taux fixe {s['K']:.2%} | {txt}")
    print(f"\nSwap retenu : {R['m_ref']} ans, nominal {m(R['n_ref'])}")
    print(f"  Coût de portage (MNI an 1)       : {m(R['cout_portage'], 2)}")
    print(f"  ΔMNI choc +200 pb, sans / avec   : {m(R['dmni']['B'], 2)} / {m(R['dmni_couvert'], 2)}")
    print("\n" + pd.DataFrame({"Sans couverture": R["deve_fixe"],
                               "Avec swap": R["deve_couvert_fixe"]}).map(pct).to_string())

    titre("5. REMBOURSEMENTS ANTICIPÉS DÉPENDANTS DES TAUX (CONVEXITÉ NÉGATIVE)")
    print(f"CPR moyen du scénario de base : {R['cpr_base']:.1%}\n")
    t5 = pd.DataFrame({
        "CPR moyen": R["cpr_scen"].map("{:.1%}".format),
        "Couvert, CPR fixe": R["deve_couvert_fixe"].map(pct),
        "Couvert, CPR comport.": R["deve_couvert_comp"].map(pct),
    })
    print(t5.to_string())
    print(f"\nNominal de swap tenant les 6 scénarios au seuil de {seuil:.0%} :")
    for mat, (lo, hi) in R["intervalle_comp_brut"].items():
        verdict = "IMPOSSIBLE avec un swap seul" if lo > hi else f"de {m(lo)} à {m(hi)}"
        print(f"  Swap {mat:>2} ans : N >= {m(lo)} (hausse) et N <= {m(hi)} (baisse) -> {verdict}")

    titre("CONCLUSION")
    print("La banque transforme (prête long à taux fixe, se finance court) : sans couverture,\n"
          "une hausse parallèle de 200 pb détruit l'essentiel de ses fonds propres en valeur\n"
          "économique. Un swap payeur bien dimensionné ramène l'exposition sous le seuil,\n"
          "au prix d'une MNI plus faible. Mais dès que les remboursements anticipés dépendent\n"
          "des taux, aucun nominal de swap ne tient à la fois la hausse et la baisse :\n"
          "l'option de remboursement vendue aux clients impose une couverture optionnelle\n"
          "(swaptions receveuses, par exemple).")


# =============================================================================
# GRAPHIQUES
# =============================================================================
def style(ax, titre_graphique, ylabel):
    ax.set_title(titre_graphique, loc="left", fontsize=12, color=ENCRE, pad=12)
    ax.set_ylabel(ylabel, color=ENCRE_2)
    ax.tick_params(colors=ENCRE_2, length=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRILLE)
    ax.grid(axis="y", color=GRILLE, linewidth=0.8)
    ax.set_axisbelow(True)


def graphiques(R):
    DOSSIER.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    tier1, pf = R["tier1"], R["pf"]

    # 1. Écoulement de l'encours
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (c, ech), col in zip(R["ech"].items(), [BLEU, ORANGE, AQUA]):
        y = ech["crd_fin"] / pf.crd.sum() * 100
        ax.plot(ech.index / 12, y, color=col, linewidth=2, label=f"CPR {c:.0%}")
    style(ax, "Encours restant du portefeuille selon le taux de remboursement anticipé",
          "% de l'encours initial")
    ax.set_xlabel("Années", color=ENCRE_2); ax.set_xlim(0, 25); ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(DOSSIER / "1_ecoulement_encours.png"); plt.close(fig)

    # 2. Gap de taux
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(LABELS)); w = 0.38
    ax.bar(x - w / 2 - 0.01, R["gap"]["A"]["Gap"] / 1e6, w, color=BLEU, label="A : DAV 100 % court terme")
    ax.bar(x + w / 2 + 0.01, R["gap"]["B"]["Gap"] / 1e6, w, color=ORANGE, label="B : DAV 70 % stables sur 5 ans")
    ax.axhline(0, color=ENCRE_2, linewidth=0.8)
    style(ax, "Gap de taux par période de révision (actifs - passifs)", "M€")
    ax.set_xticks(x, LABELS); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(DOSSIER / "2_gap_de_taux.png"); plt.close(fig)

    # 3. ΔEVE avant / après couverture / avec CPR comportemental
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = R["scenarios"]; y = np.arange(len(sc)); h = 0.26
    series = [("Non couvert", R["deve_fixe"], BLEU),
              ("Couvert par swap, CPR fixe", R["deve_couvert_fixe"], ORANGE),
              ("Couvert par swap, CPR comportemental", R["deve_couvert_comp"], AQUA)]
    for i, (nom, s, col) in enumerate(series):
        ax.barh(y + (i - 1) * (h + 0.02), s[sc] * 100, h, color=col, label=nom)
    ax.axvline(PARAMS["seuil_sot"] * 100, color=ENCRE, linestyle="--", linewidth=1)
    ax.text(PARAMS["seuil_sot"] * 100 - 2, len(sc) - 0.45, "Seuil SOT -15 %", ha="right",
            color=ENCRE, fontsize=9)
    ax.axvline(0, color=ENCRE_2, linewidth=0.8)
    ax.set_yticks(y, sc); ax.invert_yaxis()
    ax.set_title("ΔEVE en % du Tier 1 sous les 6 chocs réglementaires", loc="left",
                 fontsize=12, color=ENCRE, pad=12)
    ax.set_xlabel("% du Tier 1", color=ENCRE_2)
    for s_ in ["top", "right", "left"]:
        ax.spines[s_].set_visible(False)
    ax.grid(axis="x", color=GRILLE, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=ENCRE_2, length=0)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(DOSSIER / "3_delta_eve_scenarios.png"); plt.close(fig)

    # 4. Courbe en S des remboursements anticipés
    p = PARAMS["cpr_comportemental"]
    inc = np.linspace(-0.03, 0.04, 300)
    cpr = p["cpr_min"] + (p["cpr_max"] - p["cpr_min"]) / (1 + np.exp(-p["pente"] * (inc - p["centre"])))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(inc * 100, cpr * 100, color=BLEU, linewidth=2)
    style(ax, "Modèle de remboursement anticipé : CPR selon l'incitation à renégocier", "CPR annuel (%)")
    ax.set_xlabel("Incitation = taux du crédit - taux d'un nouveau crédit (points de %)", color=ENCRE_2)
    ax.set_ylim(0, 32)
    fig.tight_layout(); fig.savefig(DOSSIER / "4_courbe_cpr.png"); plt.close(fig)


if __name__ == "__main__":
    resultats = executer()
    rapport(resultats)
    graphiques(resultats)
    print(f"\nGraphiques enregistrés dans : {DOSSIER}")
