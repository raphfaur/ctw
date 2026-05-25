import matplotlib.pyplot as plt
import numpy as np
import colorsys

import numpy as np

def compute_regime(data, thresholds):
    """
    Calcule un régime à partir de data et thresholds.

    Parameters:
    - data:
        soit une liste (1 série)
        soit une liste de listes (2 séries)
    - thresholds:
        soit une liste (1 série)
        soit une liste de listes (2 séries)

    Returns:
    - regime: liste d'entiers
    """

    # --- Mise en forme ---
    if not isinstance(data[0], (list, np.ndarray)):
        data = [data]
        thresholds = [thresholds]

    n_series = len(data)
    length = len(data[0])

    # Vérifications
    for d in data:
        assert len(d) == length, "Toutes les séries doivent avoir la même taille"

    for t in thresholds:
        t.sort()  # important pour bien définir les intervalles

    # --- Fonction pour trouver la zone ---
    def get_zone(value, thresh):
        """
        Retourne l'indice de l'intervalle où tombe value.
        """
        for i, t in enumerate(thresh):
            if value < t:
                return i
        return len(thresh)

    # --- Calcul des régimes ---
    regime = []

    for i in range(length):
        zones = []

        for s in range(n_series):
            z = get_zone(data[s][i], thresholds[s])
            zones.append(z)

        # combinaison des zones en un seul entier
        if n_series == 1:
            regime.append(zones[0])
        else:
            # encodage type base-n
            combined = zones[0]
            multiplier = len(thresholds[0]) + 1

            for s in range(1, n_series):
                combined = combined * (len(thresholds[s]) + 1) + zones[s]

            regime.append(combined)

    return regime

def generate_distinct_colors(n):
    colors = []
    golden_ratio = 0.618033988749895
    h = 0

    for _ in range(n):
        h = (h + golden_ratio) % 1
        s = 0.65
        v = 0.9
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb)

    return colors


import matplotlib.pyplot as plt
import numpy as np

def plot_time_series(data, thresholds=None, list_regimes = None,
                     title="", xlabel="", ylabel="",
                     title2="", xlabel2="", ylabel2="",
                     path=None, data_to_plot = None):
    """
    Plot 1D ou 2D avec :
    - subplot du haut : série principale + régimes
    - subplot du bas : 2e série (si existe)

    Parameters:
    - data: liste ou liste de listes
    - thresholds: liste ou liste de listes
    """

    # --- Gestion data ---
    if isinstance(data[0], (list, np.ndarray)):
        data = [np.array(d) for d in data]
        multi_series = True
    else:
        data = [np.array(data)]
        multi_series = False

    plot_data1 = data[0]

    # --- Limite du nombre de points ---
    if data_to_plot is not None:
        plot_data1 = plot_data1[:data_to_plot]
        data = [d[:data_to_plot] for d in data]

    x = np.arange(len(plot_data1))

    # --- Régimes (toujours calculés sur toutes les séries) ---
    if list_regimes is None and thresholds is not None:
        regime = compute_regime(data if multi_series else plot_data1, thresholds)
    else:
        regime = list_regimes

    # --- Figure ---
    if multi_series:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = None

    # =========================
    # 📈 PLOT 1 (principal)
    # =========================
    ax1.plot(x, plot_data1, color='black', linewidth=1.5)

    # thresholds série 1
    if thresholds is not None:
        if isinstance(thresholds[0], (list, np.ndarray)):
            th1 = thresholds[0]
        else:
            th1 = thresholds

        for t in th1:
            ax1.axhline(y=t, linestyle='--', linewidth=1.2,
                        color='red', alpha=0.7)

    # régimes
    if regime is not None:
        regime = np.array(regime)
        unique_regimes = np.unique(regime)
        distinct_colors = generate_distinct_colors(len(unique_regimes))

        regime_to_color = {
            r: distinct_colors[i]
            for i, r in enumerate(unique_regimes)
        }

        start = 0
        current_regime = regime[0]

        for i in range(1, len(regime)):
            if regime[i] != current_regime:
                ax1.axvspan(start, i,
                            color=regime_to_color[current_regime],
                            alpha=0.3)
                start = i
                current_regime = regime[i]

        ax1.axvspan(start, len(regime),
                    color=regime_to_color[current_regime],
                    alpha=0.3)

    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.grid(False)

    # =========================
    # 📉 PLOT 2 (si 2e série)
    # =========================
    if multi_series and len(data) > 1:
        plot_data2 = data[1]

        ax2.plot(x, plot_data2, color='black', linewidth=1.5)

        # thresholds série 2
        if thresholds is not None and isinstance(thresholds[0], (list, np.ndarray)):
            th2 = thresholds[1]
            for t in th2:
                ax2.axhline(y=t, linestyle='--', linewidth=1.2,
                            color='blue', alpha=0.7)

        ax2.set_title(title2)
        ax2.set_xlabel(xlabel2 if xlabel2 else xlabel)
        ax2.set_ylabel(ylabel2)
        ax2.grid(False)

    else:
        ax1.set_xlabel(xlabel)

    # --- Layout ---
    plt.tight_layout()

    if path is not None:
        plt.savefig(path)