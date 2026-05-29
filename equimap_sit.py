# -*- coding: utf-8 -*-

import math
import os
import json
import csv
import html
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from matplotlib.colors import LinearSegmentedColormap

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterString,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingOutputFile,
    QgsProcessingOutputFolder,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsFeature,
    QgsSymbol,
    QgsRendererCategory,
    QgsCategorizedSymbolRenderer,
    QgsProcessingUtils
)


# ---------------------------------------------------------------------------
# Modern light-theme style applied globally for all poster outputs
# ---------------------------------------------------------------------------
_POSTER_STYLE = {
    'figure.facecolor':      '#f7f5f2',
    'axes.facecolor':        'none',
    'axes.edgecolor':        'none',
    'text.color':            '#1a1a1a',
    'font.family':           'serif',
    'savefig.facecolor':     '#f7f5f2',
    'savefig.edgecolor':     'none',
}

# ---------------------------------------------------------------------------
# Optional analytical dependencies for Inequality, Moran, and LISA
# ---------------------------------------------------------------------------
try:
    from libpysal.weights import Queen, Rook, KNN
    from esda.moran import Moran, Moran_Local
    _HAS_PYSAL = True
except Exception:
    _HAS_PYSAL = False

try:
    from shapely import wkt as shapely_wkt
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


# ---------------------------------------------------------------------------
# Inequality, concentration, Moran, and LISA helpers
# ---------------------------------------------------------------------------
def _ia_safe_float(v):
    try:
        if v is None:
            return np.nan
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except Exception:
        return np.nan


def _ia_clean_nonnegative_array(vals):
    arr = np.array([_ia_safe_float(v) for v in vals], dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    return arr


def _ia_clean_positive_array(vals):
    arr = np.array([_ia_safe_float(v) for v in vals], dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    return arr


def _ia_gini(x):
    x = _ia_clean_nonnegative_array(x)
    n = x.size
    if n == 0:
        return float('nan')
    s = x.sum()
    if s == 0:
        return 0.0
    xs = np.sort(x)
    i = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(i * xs)) / (n * s) - (n + 1.0) / n)


def _ia_theil_t(x):
    x = _ia_clean_positive_array(x)
    n = x.size
    if n == 0:
        return 0.0
    mu = x.mean()
    if mu <= 0:
        return 0.0
    r = x / mu
    return float(np.mean(r * np.log(r)))


def _ia_hoover(x):
    x = _ia_clean_nonnegative_array(x)
    n = x.size
    if n == 0:
        return float('nan')
    s = x.sum()
    if s == 0:
        return 0.0
    share = x / s
    eq = 1.0 / n
    return float(0.5 * np.sum(np.abs(share - eq)))


def _ia_describe_nonnegative(values):
    arr = _ia_clean_nonnegative_array(values)
    if arr.size == 0:
        return {
            'n_valid': 0, 'total': 0.0, 'mean': np.nan, 'median': np.nan,
            'std': np.nan, 'min': np.nan, 'max': np.nan, 'cv': np.nan
        }
    meanv = float(arr.mean())
    stdv = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        'n_valid': int(arr.size),
        'total': float(arr.sum()),
        'mean': meanv,
        'median': float(np.median(arr)),
        'std': stdv,
        'min': float(arr.min()),
        'max': float(arr.max()),
        'cv': float(stdv / meanv) if meanv != 0 else np.nan
    }


def _ia_pearson_corr(x, y):
    xv, yv = [], []
    for a, b in zip(x, y):
        fa = _ia_safe_float(a)
        fb = _ia_safe_float(b)
        if np.isfinite(fa) and np.isfinite(fb):
            xv.append(fa)
            yv.append(fb)
    if len(xv) < 2:
        return np.nan
    xa = np.array(xv, dtype=float)
    ya = np.array(yv, dtype=float)
    if np.std(xa) == 0 or np.std(ya) == 0:
        return np.nan
    return float(np.corrcoef(xa, ya)[0, 1])


def _ia_rankdata_average(arr):
    arr = np.asarray(arr)
    order = np.argsort(arr, kind='mergesort')
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _ia_spearman_corr(x, y):
    xv, yv = [], []
    for a, b in zip(x, y):
        fa = _ia_safe_float(a)
        fb = _ia_safe_float(b)
        if np.isfinite(fa) and np.isfinite(fb):
            xv.append(fa)
            yv.append(fb)
    if len(xv) < 2:
        return np.nan
    xa = np.array(xv, dtype=float)
    ya = np.array(yv, dtype=float)
    rx = _ia_rankdata_average(xa)
    ry = _ia_rankdata_average(ya)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _ia_rank_indices(values, ascending=True, min_allowed=0.0):
    idx = []
    for i, v in enumerate(values):
        f = _ia_safe_float(v)
        if np.isfinite(f) and f >= min_allowed:
            idx.append(i)
    idx.sort(key=lambda i: float(values[i]), reverse=(not ascending))
    return idx


def _ia_lorenz_points_sorted(vals_sorted):
    n = len(vals_sorted)
    cp = np.linspace(0.0, 1.0, n + 1)
    total = float(np.sum(vals_sorted)) if n > 0 else 0.0
    if total <= 0:
        cv = cp.copy()
    else:
        cv = np.concatenate([[0.0], np.cumsum(vals_sorted) / total])
    return cp, cv


def _ia_concentration_index(ranker_sorted_vals, ranked_y_vals):
    ranker_sorted_vals = np.array(ranker_sorted_vals, dtype=float)
    ranked_y_vals = np.array(ranked_y_vals, dtype=float)
    cp, _ = _ia_lorenz_points_sorted(ranker_sorted_vals.tolist())
    total_y = float(np.sum(ranked_y_vals))
    if total_y <= 0:
        cy = cp.copy()
    else:
        cy = np.concatenate([[0.0], np.cumsum(ranked_y_vals) / total_y])
    area = float(np.trapezoid(cy, cp)) if hasattr(np, 'trapezoid') else float(np.trapz(cy, cp))
    ci = 1.0 - 2.0 * float(area)
    return cp, cy, ci


def _ia_zscore(values):
    arr = np.array([_ia_safe_float(v) for v in values], dtype=float)
    out = [None] * len(arr)
    valid = np.isfinite(arr)
    if valid.sum() < 2:
        return out
    vals = arr[valid]
    mu = float(vals.mean())
    sd = float(vals.std(ddof=1))
    if sd == 0:
        for i in range(len(arr)):
            if np.isfinite(arr[i]):
                out[i] = 0.0
        return out
    for i, v in enumerate(arr):
        if np.isfinite(v):
            out[i] = float((v - mu) / sd)
    return out


def _ia_build_feature_metrics(values):
    n_all = len(values)
    out = {
        'VAL': [None] * n_all,
        'RANK': [None] * n_all,
        'SHARE': [None] * n_all,
        'CUMUNIT': [None] * n_all,
        'CUMVAL': [None] * n_all,
        'DEVEQ': [None] * n_all,
        'THEILTERM': [None] * n_all,
    }
    idx_sorted = _ia_rank_indices(values, ascending=True, min_allowed=0.0)
    if len(idx_sorted) == 0:
        return out, idx_sorted, [], None, None

    vals_sorted = [float(values[i]) for i in idx_sorted]
    cp, cv = _ia_lorenz_points_sorted(vals_sorted)
    total = float(np.sum(vals_sorted))
    n = len(vals_sorted)
    eq = 1.0 / n if n > 0 else None
    mu = total / n if n > 0 else None

    for j, i in enumerate(idx_sorted):
        v = float(values[i])
        share = (v / total) if total > 0 else 0.0
        dev = abs(share - eq) if eq is not None else None
        if mu and mu > 0 and v > 0:
            tt = float((v / mu) * np.log(v / mu))
        else:
            tt = 0.0 if v == 0 else None

        out['VAL'][i] = v
        out['RANK'][i] = j + 1
        out['SHARE'][i] = share
        out['CUMUNIT'][i] = float(cp[j + 1])
        out['CUMVAL'][i] = float(cv[j + 1])
        out['DEVEQ'][i] = dev
        out['THEILTERM'][i] = tt

    return out, idx_sorted, vals_sorted, cp, cv


def _ia_default_spatial_result(n):
    return {
        'z': [None] * n,
        'wz': [None] * n,
        'global_I': np.nan,
        'global_p': np.nan,
        'local_I': [None] * n,
        'local_p': [None] * n,
        'quad': [None] * n,
        'cluster_code': [None] * n,
        'cluster_label': [None] * n,
    }


def _ia_run_global_local_moran_from_features(features, values, weight_mode='queen', knn_k=8, permutations=999, alpha=0.05):
    n = len(values)
    out = _ia_default_spatial_result(n)

    if not _HAS_PYSAL or not _HAS_SHAPELY:
        return out

    valid_vals = []
    geoms = []
    map_back = []

    for i, feat in enumerate(features):
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        fv = _ia_safe_float(values[i])
        if not np.isfinite(fv):
            continue
        try:
            sg = shapely_wkt.loads(geom.asWkt())
            if sg is None or sg.is_empty:
                continue
        except Exception:
            continue
        valid_vals.append(fv)
        geoms.append(sg)
        map_back.append(i)

    if len(valid_vals) < 3:
        return out

    x = np.array(valid_vals, dtype=float)

    try:
        if weight_mode == 'queen':
            wv = Queen.from_iterable(geoms)
        elif weight_mode == 'rook':
            wv = Rook.from_iterable(geoms)
        else:
            wv = KNN.from_iterable(geoms, k=max(1, int(knn_k)))
        wv.transform = 'r'

        mi = Moran(x, wv, permutations=permutations)
        lisa = Moran_Local(x, wv, permutations=permutations)

        sd = x.std(ddof=1)
        z = (x - x.mean()) / (sd if sd != 0 else 1.0)
        lag = wv.sparse.dot(z)

        out['global_I'] = float(mi.I)
        out['global_p'] = float(mi.p_sim)

        for j, original_idx in enumerate(map_back):
            out['z'][original_idx] = float(z[j])
            out['wz'][original_idx] = float(lag[j])
            out['local_I'][original_idx] = float(lisa.Is[j])
            out['local_p'][original_idx] = float(lisa.p_sim[j])

            if lisa.p_sim[j] <= alpha:
                q = int(lisa.q[j])
                out['quad'][original_idx] = q
                if q == 1:
                    out['cluster_code'][original_idx] = 1
                    out['cluster_label'][original_idx] = 'HH'
                elif q == 2:
                    out['cluster_code'][original_idx] = 3
                    out['cluster_label'][original_idx] = 'LH'
                elif q == 3:
                    out['cluster_code'][original_idx] = 2
                    out['cluster_label'][original_idx] = 'LL'
                elif q == 4:
                    out['cluster_code'][original_idx] = 4
                    out['cluster_label'][original_idx] = 'HL'
            else:
                out['quad'][original_idx] = 0
                out['cluster_code'][original_idx] = 0
                out['cluster_label'][original_idx] = 'NS'
    except Exception:
        return out


    return out


def _ia_run_bivariate_moran_from_features(features, focal_values, neighbor_values,
                                          weight_mode='queen', knn_k=8,
                                          permutations=999, alpha=0.05,
                                          random_seed=12345):
    """Run Bivariate Moran and Bivariate Local Moran/LISA.

    Directional interpretation:
    - focal_values are standardized at location i.
    - neighbor_values are standardized and spatially lagged around i.
    - global_I is mean(z_focal_i * Wz_neighbor_i).
    - local_I_i is z_focal_i * Wz_neighbor_i.

    This is intentionally implemented without relying on Moran_BV so it is
    more robust across different esda versions in QGIS Python environments.
    """
    n = len(focal_values)
    out = _ia_default_spatial_result(n)

    if not _HAS_PYSAL or not _HAS_SHAPELY:
        return out

    focal_valid = []
    neigh_valid = []
    geoms = []
    map_back = []

    for i, feat in enumerate(features):
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        fv = _ia_safe_float(focal_values[i])
        nv = _ia_safe_float(neighbor_values[i])
        if not np.isfinite(fv) or not np.isfinite(nv):
            continue
        try:
            sg = shapely_wkt.loads(geom.asWkt())
            if sg is None or sg.is_empty:
                continue
        except Exception:
            continue
        focal_valid.append(fv)
        neigh_valid.append(nv)
        geoms.append(sg)
        map_back.append(i)

    if len(focal_valid) < 3:
        return out

    try:
        if weight_mode == 'queen':
            wv = Queen.from_iterable(geoms)
        elif weight_mode == 'rook':
            wv = Rook.from_iterable(geoms)
        else:
            kk = min(max(1, int(knn_k)), max(1, len(geoms) - 1))
            wv = KNN.from_iterable(geoms, k=kk)
        wv.transform = 'r'

        xf = np.array(focal_valid, dtype=float)
        yn = np.array(neigh_valid, dtype=float)
        xf_sd = xf.std(ddof=1)
        yn_sd = yn.std(ddof=1)
        zx = (xf - xf.mean()) / (xf_sd if xf_sd != 0 else 1.0)
        zy = (yn - yn.mean()) / (yn_sd if yn_sd != 0 else 1.0)
        lag_zy = wv.sparse.dot(zy)

        local_i = zx * lag_zy
        global_i = float(np.mean(local_i))

        rng = np.random.default_rng(random_seed)
        permutations = int(permutations) if permutations is not None else 0
        permutations = max(0, permutations)
        global_extreme = 0
        local_extreme = np.zeros(len(zx), dtype=int)

        if permutations > 0:
            abs_global = abs(global_i)
            abs_local = np.abs(local_i)
            for _ in range(permutations):
                zy_perm = rng.permutation(zy)
                lag_perm = wv.sparse.dot(zy_perm)
                local_perm = zx * lag_perm
                global_perm = float(np.mean(local_perm))
                if abs(global_perm) >= abs_global:
                    global_extreme += 1
                local_extreme += (np.abs(local_perm) >= abs_local)
            global_p = float((global_extreme + 1) / (permutations + 1))
            local_p = (local_extreme + 1) / float(permutations + 1)
        else:
            global_p = np.nan
            local_p = np.full(len(zx), np.nan, dtype=float)

        out['global_I'] = global_i
        out['global_p'] = global_p

        for j, original_idx in enumerate(map_back):
            out['z'][original_idx] = float(zx[j])
            out['wz'][original_idx] = float(lag_zy[j])
            out['local_I'][original_idx] = float(local_i[j])
            out['local_p'][original_idx] = float(local_p[j]) if np.isfinite(local_p[j]) else np.nan

            significant = np.isfinite(local_p[j]) and local_p[j] <= alpha
            if significant:
                if zx[j] >= 0 and lag_zy[j] >= 0:
                    out['quad'][original_idx] = 1
                    out['cluster_code'][original_idx] = 1
                    out['cluster_label'][original_idx] = 'HH'
                elif zx[j] < 0 and lag_zy[j] >= 0:
                    out['quad'][original_idx] = 2
                    out['cluster_code'][original_idx] = 3
                    out['cluster_label'][original_idx] = 'LH'
                elif zx[j] < 0 and lag_zy[j] < 0:
                    out['quad'][original_idx] = 3
                    out['cluster_code'][original_idx] = 2
                    out['cluster_label'][original_idx] = 'LL'
                else:
                    out['quad'][original_idx] = 4
                    out['cluster_code'][original_idx] = 4
                    out['cluster_label'][original_idx] = 'HL'
            else:
                out['quad'][original_idx] = 0
                out['cluster_code'][original_idx] = 0
                out['cluster_label'][original_idx] = 'NS'
    except Exception:
        return out

    return out



def _ia_setup_card_axes(ax, with_grid=True):
    ax.set_facecolor('#ffffff')
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#d4ccc4')
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors='#4a4540', labelsize=8.5, length=3, width=0.7)
    if with_grid:
        ax.grid(True, linestyle=':', linewidth=0.55, alpha=0.55, color='#c8c1b8')
    ax.set_axisbelow(True)


def _ia_add_analytics_header(fig, kicker, title, subtitle=None, accent='#1f7a8c'):
    """Draw a clean header with tighter vertical spacing between kicker, title, subtitle, and chart area."""
    fig.text(0.055, 0.972, kicker.upper(), ha='left', va='top', fontsize=7.5,
             color=accent, fontweight='bold', family='sans-serif')
    fig.text(0.055, 0.948, title, ha='left', va='top', fontsize=15.5,
             color='#171513', fontweight='bold', family='sans-serif')
    if subtitle:
        fig.text(0.055, 0.904, subtitle, ha='left', va='top', fontsize=8.8,
                 color='#706860', family='serif', style='italic')

def _ia_save_curve_plot(title, xlabel, ylabel, cp, cv, png_path, subtitle=None):
    fig = plt.figure(figsize=(8.5, 6.4), facecolor='#f7f5f2')
    # Header ~25% top, plot 60%, bottom margin 8%, note strip at very bottom
    ax = fig.add_axes([0.11, 0.16, 0.84, 0.64])
    _ia_setup_card_axes(ax, with_grid=True)
    _ia_add_analytics_header(fig, 'EquiMap', title, subtitle, accent='#1e847f')

    cp = np.array(cp, dtype=float)
    cv = np.array(cv, dtype=float)

    ax.fill_between(cp, cp, cv, where=(cv >= cp), color='#7ecac3', alpha=0.18, zorder=1)
    ax.fill_between(cp, cv, cp, where=(cp > cv), color='#e07070', alpha=0.10, zorder=1)
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1.3, color='#afa79e', label='Equality line', zorder=2)
    ax.plot(cp, cv, linewidth=2.4, color='#1e847f', label='Observed curve', zorder=3)

    if len(cp) > 1:
        ax.scatter(cp[1:], cv[1:], s=18, color='#1a5c60', edgecolors='white', linewidths=0.6, zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel, fontsize=9.5, color='#2d2a26', labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9.5, color='#2d2a26', labelpad=6)
    ax.tick_params(labelsize=8.5)

    leg = ax.legend(loc='upper left', frameon=True, fontsize=8.2, handlelength=1.8)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#ddd5cb')
    leg.get_frame().set_alpha(0.96)
    leg.get_frame().set_linewidth(0.8)

    # Annotation note — placed BELOW the axes as a figure-level text
    fig.text(0.50, 0.055, 'The farther the curve is from the equality line, the more unequal the distribution is.',
             ha='center', va='bottom', fontsize=8.0, color='#6d655d', style='italic')

    fig.savefig(png_path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

def _ia_save_moran_scatter_png(zvals, wzvals, var_name, moran_i, pval, png_path):
    pts = [(a, b) for a, b in zip(zvals, wzvals) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pts) == 0:
        return False

    zv = np.array([p[0] for p in pts], dtype=float)
    wv = np.array([p[1] for p in pts], dtype=float)

    fig = plt.figure(figsize=(7.8, 6.2), facecolor='#f7f5f2')
    ax = fig.add_axes([0.12, 0.12, 0.83, 0.66])
    _ia_setup_card_axes(ax, with_grid=True)
    _ia_add_analytics_header(fig, 'EquiMap', 'Moran Scatterplot: {}'.format(var_name),
                             "Global Moran's I = {:.4f} | p-value = {:.4f}".format(moran_i, pval), accent='#6a4c93')

    # Quadrant shading — tighter alpha
    x_lo, x_hi = np.min(zv), np.max(zv)
    w_lo, w_hi = np.min(wv), np.max(wv)
    ax.fill_betweenx([0, max(w_hi * 1.1, 0.1)], x_lo * 1.1, 0, color='#dce8f0', alpha=0.55, zorder=0)
    ax.fill_betweenx([min(w_lo * 1.1, -0.1), 0], 0, max(x_hi * 1.1, 0.1), color='#fde8de', alpha=0.55, zorder=0)
    ax.axvline(0, linestyle='--', linewidth=1.0, color='#9e968d', zorder=1)
    ax.axhline(0, linestyle='--', linewidth=1.0, color='#9e968d', zorder=1)

    ax.scatter(zv, wv, s=32, alpha=0.88, facecolor='#4d8ca3', edgecolors='white', linewidths=0.7, zorder=3)

    if len(zv) > 1:
        coef = np.polyfit(zv, wv, 1)
        xs = np.linspace(x_lo, x_hi, 200)
        ax.plot(xs, coef[0] * xs + coef[1], linewidth=2.0, color='#6a4c93', zorder=4)

    ax.set_xlabel('{} standardized'.format(var_name), fontsize=9.5, color='#2d2a26', labelpad=6)
    ax.set_ylabel('Spatial lag of {}'.format(var_name), fontsize=9.5, color='#2d2a26', labelpad=6)

    quad_box = dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#ddd5cb', alpha=0.94)
    ax.text(0.97, 0.97, 'HH', transform=ax.transAxes, ha='right', va='top', fontsize=8.0, color='#5a3224', bbox=quad_box)
    ax.text(0.03, 0.97, 'LH', transform=ax.transAxes, ha='left',  va='top', fontsize=8.0, color='#2b5a6f', bbox=quad_box)
    ax.text(0.97, 0.03, 'HL', transform=ax.transAxes, ha='right', va='bottom', fontsize=8.0, color='#7b4d2a', bbox=quad_box)
    ax.text(0.03, 0.03, 'LL', transform=ax.transAxes, ha='left',  va='bottom', fontsize=8.0, color='#355c7d', bbox=quad_box)

    fig.savefig(png_path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _ia_save_bivariate_moran_scatter_png(zvals, wzvals, focal_name, lag_name, moran_i, pval, png_path):
    pts = [(a, b) for a, b in zip(zvals, wzvals) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pts) == 0:
        return False

    zv = np.array([p[0] for p in pts], dtype=float)
    wv = np.array([p[1] for p in pts], dtype=float)

    fig = plt.figure(figsize=(7.8, 6.2), facecolor='#f7f5f2')
    ax = fig.add_axes([0.12, 0.12, 0.83, 0.66])
    _ia_setup_card_axes(ax, with_grid=True)
    _ia_add_analytics_header(
        fig, 'EquiMap',
        'Bivariate Moran Scatterplot: {} vs lag({})'.format(focal_name, lag_name),
        "Bivariate Moran's I = {:.4f} | p-value = {:.4f}".format(moran_i, pval),
        accent='#2d6a9f'
    )

    x_lo, x_hi = np.min(zv), np.max(zv)
    w_lo, w_hi = np.min(wv), np.max(wv)
    ax.fill_betweenx([0, max(w_hi * 1.1, 0.1)], x_lo * 1.1, 0, color='#dce8f0', alpha=0.55, zorder=0)
    ax.fill_betweenx([min(w_lo * 1.1, -0.1), 0], 0, max(x_hi * 1.1, 0.1), color='#fde8de', alpha=0.55, zorder=0)
    ax.axvline(0, linestyle='--', linewidth=1.0, color='#9e968d', zorder=1)
    ax.axhline(0, linestyle='--', linewidth=1.0, color='#9e968d', zorder=1)

    ax.scatter(zv, wv, s=32, alpha=0.88, facecolor='#2d6a9f', edgecolors='white', linewidths=0.7, zorder=3)

    if len(zv) > 1:
        coef = np.polyfit(zv, wv, 1)
        xs = np.linspace(x_lo, x_hi, 200)
        ax.plot(xs, coef[0] * xs + coef[1], linewidth=2.0, color='#2d6a9f', zorder=4)

    ax.set_xlabel('{} standardized'.format(focal_name), fontsize=9.5, color='#2d2a26', labelpad=6)
    ax.set_ylabel('Spatial lag of {}'.format(lag_name), fontsize=9.5, color='#2d2a26', labelpad=6)

    quad_box = dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#ddd5cb', alpha=0.94)
    ax.text(0.97, 0.97, 'HH', transform=ax.transAxes, ha='right', va='top', fontsize=8.0, color='#5a3224', bbox=quad_box)
    ax.text(0.03, 0.97, 'LH', transform=ax.transAxes, ha='left',  va='top', fontsize=8.0, color='#2b5a6f', bbox=quad_box)
    ax.text(0.97, 0.03, 'HL', transform=ax.transAxes, ha='right', va='bottom', fontsize=8.0, color='#7b4d2a', bbox=quad_box)
    ax.text(0.03, 0.03, 'LL', transform=ax.transAxes, ha='left',  va='bottom', fontsize=8.0, color='#355c7d', bbox=quad_box)

    fig.savefig(png_path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return True

def _ia_extent_from_features(features):
    xmin = ymin = xmax = ymax = None
    for feat in features:
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        bbox = geom.boundingBox()
        if xmin is None:
            xmin = bbox.xMinimum()
            ymin = bbox.yMinimum()
            xmax = bbox.xMaximum()
            ymax = bbox.yMaximum()
        else:
            xmin = min(xmin, bbox.xMinimum())
            ymin = min(ymin, bbox.yMinimum())
            xmax = max(xmax, bbox.xMaximum())
            ymax = max(ymax, bbox.yMaximum())
    if xmin is None:
        return (0, 1, 0, 1)
    padx = (xmax - xmin) * 0.03 if xmax > xmin else 1
    pady = (ymax - ymin) * 0.03 if ymax > ymin else 1
    return (xmin - padx, xmax + padx, ymin - pady, ymax + pady)


def _ia_draw_qgis_polygon(ax, geom, facecolor, edgecolor='#4d4d4d', lw=0.25):
    if geom is None or geom.isEmpty():
        return
    try:
        if geom.isMultipart():
            polys = geom.asMultiPolygon()
        else:
            polys = [geom.asPolygon()]
        for poly in polys:
            if not poly:
                continue
            outer = poly[0]
            if outer:
                xs = [pt.x() for pt in outer]
                ys = [pt.y() for pt in outer]
                ax.fill(xs, ys, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw)
    except Exception:
        pass


def _ia_save_lisa_map_png(features, lisa_cluster_codes, title, png_path):
    lisa_colors = {
        0: '#d8d2c9',  # NS  – visible warm neutral
        1: '#c84b5a',  # HH  – crimson
        2: '#2b4fa8',  # LL  – cobalt
        3: '#6dbcd4',  # LH  – sky blue
        4: '#e8803a',  # HL  – amber
    }

    fig = plt.figure(figsize=(10.5, 7.0), facecolor='#f7f5f2')
    # Map axes: left 66% of figure width, vertically from 10% to 88%
    ax     = fig.add_axes([0.04, 0.11, 0.62, 0.75], facecolor='white')
    ax_leg = fig.add_axes([0.70, 0.14, 0.26, 0.68], facecolor='white')
    _ia_add_analytics_header(fig, 'EquiMap', title,
                             'Local Moran cluster map', accent='#b85c38')

    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    for feat, c in zip(features, lisa_cluster_codes):
        geom  = feat.geometry()
        code  = 0 if c is None else int(c)
        color = lisa_colors.get(code, '#ece8e2')
        _ia_draw_qgis_polygon(ax, geom, facecolor=color, edgecolor='#f1ece4', lw=0.42)

    xmin, xmax, ymin, ymax = _ia_extent_from_features(features)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')

    # Legend panel
    ax_leg.set_axis_off()
    card = FancyBboxPatch((0.04, 0.04), 0.92, 0.92,
                          boxstyle='round,pad=0.02,rounding_size=0.04',
                          transform=ax_leg.transAxes,
                          facecolor='white', edgecolor='#d8d0c8', linewidth=0.8)
    ax_leg.add_patch(card)
    ax_leg.text(0.10, 0.93, 'LISA LEGEND', transform=ax_leg.transAxes,
                ha='left', va='top', fontsize=8.5, color='#b85c38', fontweight='bold')
    ax_leg.text(0.10, 0.87, 'Local cluster type', transform=ax_leg.transAxes,
                ha='left', va='top', fontsize=7.8, color='#6d655d')

    entries = [
        ('High-High',      lisa_colors[1], 'High concentration\nsurrounded by high values'),
        ('Low-Low',        lisa_colors[2], 'Low concentration\nsurrounded by low values'),
        ('Low-High',       lisa_colors[3], 'Low outlier\namong high values'),
        ('High-Low',       lisa_colors[4], 'High outlier\namong low values'),
        ('Not Significant',lisa_colors[0], 'Not significant\nlocally'),
    ]
    y0 = 0.77
    row_h = 0.155
    for lbl, col, desc in entries:
        ax_leg.add_patch(Rectangle((0.10, y0 - 0.040), 0.11, 0.072,
                                   transform=ax_leg.transAxes,
                                   facecolor=col, edgecolor='#c4bcb4', linewidth=0.7))
        ax_leg.text(0.26, y0 + 0.012, lbl,
                    transform=ax_leg.transAxes, ha='left', va='top',
                    fontsize=8.5, color='#2d2a26', fontweight='bold')
        ax_leg.text(0.26, y0 - 0.012, desc,
                    transform=ax_leg.transAxes, ha='left', va='top',
                    fontsize=7.4, color='#6b645d', linespacing=1.25)
        y0 -= row_h

    fig.savefig(png_path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _ia_save_bilisa_map_png(features, cluster_codes_y, cluster_codes_x,
                             label_y, label_x, png_path):
    """Bivariate LISA (BiLISA) map — one combined map.

    Each polygon is drawn with a SPLIT rendering:
    • Left half  → LISA cluster colour for Variable Y
    • Right half → LISA cluster colour for Variable X

    A 5×5 bivariate legend matrix (Y rows × X cols) sits on the right.
    A compact count table is shown at the bottom of the legend panel.
    """
    LISA_COLORS = {
        0: '#d8d2c9',  # NS
        1: '#c84b5a',  # HH – crimson
        2: '#2b4fa8',  # LL – cobalt
        3: '#6dbcd4',  # LH – sky
        4: '#e8803a',  # HL – amber
    }
    CLUSTER_LABELS  = {0: 'NS', 1: 'HH', 2: 'LL', 3: 'LH', 4: 'HL'}
    CLUSTER_ORDER   = [1, 3, 4, 2, 0]  # display order in legend: HH LH HL LL NS
    CLUSTER_DISPLAY = ['HH', 'LH', 'HL', 'LL', 'NS']

    fig = plt.figure(figsize=(13.0, 7.5), facecolor='#f7f5f2')

    # ── Header ───────────────────────────────────────────────────────────────
    _ia_add_analytics_header(
        fig, 'EquiMap',
        'Bivariate LISA: {} ↔ {}'.format(label_y, label_x),
        'Local Moran spatial association of two variables in one map (left=Y, right=X)',
        accent='#2d6a9f'
    )

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax     = fig.add_axes([0.03, 0.11, 0.66, 0.75], facecolor='white')
    ax_leg = fig.add_axes([0.72, 0.11, 0.26, 0.73], facecolor='white')

    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # ── Draw each polygon split left/right ───────────────────────────────────
    xmn, xmx, ymn, ymx = _ia_extent_from_features(features)
    ax.set_xlim(xmn, xmx); ax.set_ylim(ymn, ymx)
    ax.set_aspect('equal', adjustable='box')

    def _draw_split_polygon(ax, geom, col_left, col_right, edgecolor='#efe9df', lw=0.38):
        """Draw a polygon split vertically: left half col_left, right half col_right."""
        if geom is None or geom.isEmpty():
            return
        try:
            if geom.isMultipart():
                polys = geom.asMultiPolygon()
            else:
                polys = [geom.asPolygon()]
        except Exception:
            return
        for poly in polys:
            if not poly:
                continue
            outer = poly[0]
            if not outer:
                continue
            xs = [pt.x() for pt in outer]
            ys = [pt.y() for pt in outer]
            if not xs:
                continue
            cx = (min(xs) + max(xs)) / 2.0
            # Build left and right point lists with the vertical mid-line as clip
            from matplotlib.patches import PathPatch
            from matplotlib.path import Path
            pts = list(zip(xs, ys))
            # left polygon: clip x <= cx
            left_xs  = [min(x, cx) for x in xs]
            right_xs = [max(x, cx) for x in xs]
            ax.fill(left_xs,  ys, facecolor=col_left,  edgecolor='none', zorder=2)
            ax.fill(right_xs, ys, facecolor=col_right, edgecolor='none', zorder=2)
            # Redraw outline on top
            ax.plot(xs + [xs[0]], ys + [ys[0]],
                    color=edgecolor, linewidth=lw, zorder=3)

    for feat, cy_code, cx_code in zip(features, cluster_codes_y, cluster_codes_x):
        geom   = feat.geometry()
        col_y  = LISA_COLORS.get(0 if cy_code is None else int(cy_code), LISA_COLORS[0])
        col_x  = LISA_COLORS.get(0 if cx_code is None else int(cx_code), LISA_COLORS[0])
        _draw_split_polygon(ax, geom, col_y, col_x)

    # ── Split indicator arrow inside map ─────────────────────────────────────
    ax.text(0.26, 0.026, '← {}'.format(label_y), transform=ax.transAxes,
            ha='center', va='bottom', fontsize=7.5, color='#2d2a26',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                      edgecolor='#d4ccc4', alpha=0.90))
    ax.text(0.74, 0.026, '{} →'.format(label_x), transform=ax.transAxes,
            ha='center', va='bottom', fontsize=7.5, color='#2d2a26',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                      edgecolor='#d4ccc4', alpha=0.90))
    # Thin mid-line hint (dashed)
    ax.axvline((xmn + xmx) / 2.0, linestyle=':', linewidth=0.6, color='#aaa49e', zorder=1)

    # ── Legend panel ─────────────────────────────────────────────────────────
    ax_leg.set_axis_off()
    card = FancyBboxPatch((0.03, 0.02), 0.94, 0.96,
                          boxstyle='round,pad=0.015,rounding_size=0.03',
                          transform=ax_leg.transAxes,
                          facecolor='white', edgecolor='#d0c9c2', linewidth=0.8)
    ax_leg.add_patch(card)

    # ─ Title
    ax_leg.text(0.50, 0.965, 'BiLISA LEGEND', transform=ax_leg.transAxes,
                ha='center', va='top', fontsize=8.5, color='#2d6a9f', fontweight='bold')
    ax_leg.text(0.50, 0.940, 'Left = {}   |   Right = {}'.format(label_y, label_x),
                transform=ax_leg.transAxes, ha='center', va='top',
                fontsize=7.2, color='#6b645d')

    # ─ 5×5 matrix grid (Y rows top→bottom, X cols left→right)
    CELL   = 0.092   # cell size in axes fraction
    ORIGIN_X = 0.13  # left edge of matrix
    ORIGIN_Y = 0.87  # top edge of matrix
    LABEL_OFFSET = 0.02

    # Column header (Variable X) — top of each column
    ax_leg.text(ORIGIN_X + 2.5 * CELL, ORIGIN_Y + LABEL_OFFSET + 0.024,
                label_x[:12], transform=ax_leg.transAxes,
                ha='center', va='bottom', fontsize=7.0, color='#3a3530',
                fontweight='bold')
    for ci, c_code in enumerate(CLUSTER_ORDER):
        ax_leg.text(ORIGIN_X + (ci + 0.5) * CELL, ORIGIN_Y + LABEL_OFFSET,
                    CLUSTER_LABELS[c_code],
                    transform=ax_leg.transAxes, ha='center', va='bottom',
                    fontsize=6.8, color=LISA_COLORS[c_code],
                    fontweight='bold')

    # Row header (Variable Y) — left of each row
    ax_leg.text(ORIGIN_X - LABEL_OFFSET - 0.020,
                ORIGIN_Y - 2.5 * CELL,
                label_y[:10], transform=ax_leg.transAxes,
                ha='right', va='center', fontsize=7.0, color='#3a3530',
                fontweight='bold', rotation=90)
    for ri, r_code in enumerate(CLUSTER_ORDER):
        ax_leg.text(ORIGIN_X - LABEL_OFFSET,
                    ORIGIN_Y - (ri + 0.5) * CELL,
                    CLUSTER_LABELS[r_code],
                    transform=ax_leg.transAxes, ha='right', va='center',
                    fontsize=6.8, color=LISA_COLORS[r_code], fontweight='bold')

    # Matrix cells
    for ri, r_code in enumerate(CLUSTER_ORDER):     # Y (rows)
        for ci, c_code in enumerate(CLUSTER_ORDER): # X (cols)
            col_y = LISA_COLORS[r_code]
            col_x = LISA_COLORS[c_code]
            cx0 = ORIGIN_X + ci * CELL
            cy0 = ORIGIN_Y - (ri + 1) * CELL
            # Left half: Y colour
            ax_leg.add_patch(Rectangle(
                (cx0, cy0), CELL / 2, CELL,
                transform=ax_leg.transAxes,
                facecolor=col_y, edgecolor='white', linewidth=0.6))
            # Right half: X colour
            ax_leg.add_patch(Rectangle(
                (cx0 + CELL / 2, cy0), CELL / 2, CELL,
                transform=ax_leg.transAxes,
                facecolor=col_x, edgecolor='white', linewidth=0.6))
            # Outer border
            ax_leg.add_patch(Rectangle(
                (cx0, cy0), CELL, CELL,
                transform=ax_leg.transAxes,
                facecolor='none', edgecolor='#bdb6ae', linewidth=0.5))

    # ─ Count table below the matrix
    TBASE = ORIGIN_Y - (len(CLUSTER_ORDER) + 0.6) * CELL
    ax_leg.text(0.50, TBASE, 'Count per cluster type', transform=ax_leg.transAxes,
                ha='center', va='top', fontsize=7.2, color='#5a5450',
                fontweight='bold')
    TBASE -= 0.040
    ax_leg.text(0.08, TBASE, 'Cluster', transform=ax_leg.transAxes,
                ha='left', va='top', fontsize=6.8, color='#444', fontweight='bold')
    ax_leg.text(0.53, TBASE, label_y[:8], transform=ax_leg.transAxes,
                ha='center', va='top', fontsize=6.8, color=BAR_COL_Y if False else '#2980b9', fontweight='bold')
    ax_leg.text(0.85, TBASE, label_x[:8], transform=ax_leg.transAxes,
                ha='center', va='top', fontsize=6.8, color='#e07b39', fontweight='bold')
    BAR_COL_Y_LEG = '#2980b9'
    BAR_COL_X_LEG = '#e07b39'
    TBASE -= 0.032
    for code in CLUSTER_ORDER:
        cnt_y = sum(1 for v in cluster_codes_y if (v or 0) == code)
        cnt_x = sum(1 for v in cluster_codes_x if (v or 0) == code)
        lbl   = CLUSTER_LABELS[code]
        ax_leg.add_patch(Rectangle((0.06, TBASE - 0.010), 0.020, 0.022,
                                   transform=ax_leg.transAxes,
                                   facecolor=LISA_COLORS[code], edgecolor='#c4bcb4',
                                   linewidth=0.5))
        ax_leg.text(0.10, TBASE, lbl, transform=ax_leg.transAxes,
                    ha='left', va='top', fontsize=7.0, color='#2d2a26', fontweight='bold')
        ax_leg.text(0.53, TBASE, str(cnt_y), transform=ax_leg.transAxes,
                    ha='center', va='top', fontsize=7.0, color='#2d2a26')
        ax_leg.text(0.85, TBASE, str(cnt_x), transform=ax_leg.transAxes,
                    ha='center', va='top', fontsize=7.0, color='#2d2a26')
        TBASE -= 0.030

    # ── Bottom note ──────────────────────────────────────────────────────────
    fig.text(0.36, 0.038,
             'Each polygon is split: left=LISA cluster {} | right=LISA cluster {}. '
             'Uniform polygons indicate spatial congruence, for example both HH or both LL.'.format(label_y, label_x),
             ha='center', va='bottom', fontsize=7.6, color='#706860', style='italic')

    fig.savefig(png_path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _ia_write_curve_csv(path, names_sorted, vals_sorted, cp, cv):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['rank', 'name', 'value', 'cum_unit_share', 'cum_value_share'])
        for i, (nm, v) in enumerate(zip(names_sorted, vals_sorted), start=1):
            w.writerow([i, nm, f'{float(v):.12f}', f'{float(cp[i]):.12f}', f'{float(cv[i]):.12f}'])


def _ia_write_concentration_csv(path, names_sorted, x_sorted, y_sorted, cp, cy, x_name, y_name):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['rank', 'name', x_name, y_name, 'cum_x_share', 'cum_y_share'])
        for i, (nm, xv, yv) in enumerate(zip(names_sorted, x_sorted, y_sorted), start=1):
            w.writerow([i, nm, f'{float(xv):.12f}', f'{float(yv):.12f}', f'{float(cp[i]):.12f}', f'{float(cy[i]):.12f}'])


def _ia_norm01_list(values):
    arr = np.array([_ia_safe_float(v) for v in values], dtype=float)
    out = [0.0] * len(arr)
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return out
    lo = float(np.nanmin(arr[valid]))
    hi = float(np.nanmax(arr[valid]))
    if abs(hi - lo) < 1e-12:
        return [0.5 if np.isfinite(v) else 0.0 for v in arr]
    for i, v in enumerate(arr):
        out[i] = float((v - lo) / (hi - lo)) if np.isfinite(v) else 0.0
    return out


def _ia_priority_level(score):
    try:
        s = float(score)
    except Exception:
        return 'No Data'
    if s >= 75:
        return 'Very High'
    if s >= 55:
        return 'High'
    if s >= 35:
        return 'Moderate'
    if s > 0:
        return 'Low'
    return 'No Data'


def _ia_policy_recommendation(typology):
    mapping = {
        'Critical Y-High X-Low Mismatch': 'Prioritize equitable X intervention in areas with high Y pressure or need but low X support.',
        'Double High Cluster': 'Maintain service capacity and manage concentration risk so spatial inequality does not intensify.',
        'Double Low Cluster': 'Promote basic equalization, access improvement, and minimum investment in clustered low-value areas.',
        'Y-Dominant Spatial Hotspot': 'Manage Y pressure through capacity improvement, connectivity enhancement, or intensity control according to the indicator context.',
        'X-Dominant Concentration': 'Evaluate possible X overconcentration and opportunities to redistribute benefits toward high-inequality areas.',
        'Spatial Outlier / Transition Zone': 'A local assessment is required because the area shows an outlier pattern relative to its neighbors.',
        'Balanced / Monitor': 'Maintain relatively balanced conditions and conduct periodic monitoring.'
    }
    return mapping.get(typology, 'Conduct local verification and interpret the results according to the selected indicator context.')


def _ia_build_equimap_priority(values_y, values_x, y_cls_list, x_cls_list, metrics_y, metrics_x, spatial_y, spatial_x):
    n = len(values_y)
    y_norm = _ia_norm01_list(values_y)
    x_norm = _ia_norm01_list(values_x)
    y_dev = _ia_norm01_list(metrics_y.get('DEVEQ', [0] * n))
    x_dev = _ia_norm01_list(metrics_x.get('DEVEQ', [0] * n))
    y_theil = _ia_norm01_list(metrics_y.get('THEILTERM', [0] * n))
    x_theil = _ia_norm01_list(metrics_x.get('THEILTERM', [0] * n))
    score = [0.0] * n
    level = ['No Data'] * n
    typology = ['No Data'] * n
    rec = ['No Data'] * n
    mismatch = [0.0] * n
    lisa_sig = [0.0] * n
    for i in range(n):
        vy = _ia_safe_float(values_y[i])
        vx = _ia_safe_float(values_x[i])
        if not np.isfinite(vy) or not np.isfinite(vx):
            continue
        mismatch[i] = max(0.0, y_norm[i] - x_norm[i])
        lisa_y = spatial_y.get('cluster_code', [None] * n)[i]
        lisa_x = spatial_x.get('cluster_code', [None] * n)[i]
        lisa_y_p = spatial_y.get('local_p', [None] * n)[i]
        lisa_x_p = spatial_x.get('local_p', [None] * n)[i]
        sig_y = 1.0 if lisa_y_p is not None and np.isfinite(lisa_y_p) and lisa_y_p <= 0.05 else 0.0
        sig_x = 1.0 if lisa_x_p is not None and np.isfinite(lisa_x_p) and lisa_x_p <= 0.05 else 0.0
        lisa_sig[i] = max(sig_y, sig_x)
        inequality_contrib = 0.35 * y_dev[i] + 0.25 * x_dev[i] + 0.20 * y_theil[i] + 0.20 * x_theil[i]
        spatial_penalty = 0.0
        if lisa_y in (1, 2, 3, 4):
            spatial_penalty += 0.50
        if lisa_x in (1, 2, 3, 4):
            spatial_penalty += 0.50
        score_raw = 0.42 * mismatch[i] + 0.25 * inequality_contrib + 0.18 * min(1.0, spatial_penalty) + 0.15 * y_norm[i]
        score[i] = float(max(0.0, min(100.0, score_raw * 100.0)))
        level[i] = _ia_priority_level(score[i])
        ycls = y_cls_list[i]
        xcls = x_cls_list[i]
        if ycls is not None and xcls is not None and ycls >= 2 and xcls == 0:
            typology[i] = 'Critical Y-High X-Low Mismatch'
        elif lisa_y == 1 and lisa_x == 1:
            typology[i] = 'Double High Cluster'
        elif lisa_y == 2 and lisa_x == 2:
            typology[i] = 'Double Low Cluster'
        elif y_norm[i] >= 0.67 and x_norm[i] < 0.50:
            typology[i] = 'Y-Dominant Spatial Hotspot'
        elif x_norm[i] >= 0.67 and y_norm[i] < 0.50:
            typology[i] = 'X-Dominant Concentration'
        elif lisa_y in (3, 4) or lisa_x in (3, 4):
            typology[i] = 'Spatial Outlier / Transition Zone'
        else:
            typology[i] = 'Balanced / Monitor'
        rec[i] = _ia_policy_recommendation(typology[i])
    return {'score': score, 'level': level, 'typology': typology, 'recommendation': rec, 'mismatch': mismatch, 'lisa_sig': lisa_sig, 'y_norm': y_norm, 'x_norm': x_norm}


def _ia_write_priority_ranking_csv(path, names, label_y, label_x, values_y, values_x, priority):
    rows = []
    for i, nm in enumerate(names):
        rows.append([i, nm, _ia_safe_float(values_y[i]), _ia_safe_float(values_x[i]), priority['y_norm'][i], priority['x_norm'][i], priority['mismatch'][i], priority['score'][i], priority['level'][i], priority['typology'][i], priority['recommendation'][i]])
    rows.sort(key=lambda r: r[7], reverse=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['rank', 'feature_index', 'feature_name', label_y, label_x, 'y_norm', 'x_norm', 'mismatch_score', 'priority_score', 'priority_level', 'typology', 'recommendation'])
        for rank, row in enumerate(rows, start=1):
            w.writerow([rank] + row)


def _ia_save_summary_dashboard_png(path, label_y, label_x, gini_y, gini_x, theil_y, theil_x, hoover_y, hoover_x, pearson, spearman, spatial_y, spatial_x, priority):
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(15.5, 8.2), facecolor='#f7f5f2')
    _ia_add_analytics_header(fig, 'EquiMap v1.3.1', 'Spatial Inequality Intelligence Dashboard',
                             '{} versus {}'.format(label_y, label_x), accent='#1f7a8c')

    # ── Layout: 2 rows × 3 cols ───────────────────────────────────────────────
    # Bottom row: LISA (col 0) is narrow; policy typologies (col 1–2) has enough
    # left margin so its y-tick labels clear the LISA axes completely.
    gs = gridspec.GridSpec(2, 3,
                           left=0.050, right=0.975,
                           top=0.800, bottom=0.090,
                           wspace=0.42, hspace=0.54,
                           width_ratios=[1, 0.92, 1.08])

    ax1 = fig.add_subplot(gs[0, 0])   # Variable Y inequality
    ax2 = fig.add_subplot(gs[0, 1])   # Variable X inequality
    ax3 = fig.add_subplot(gs[0, 2])   # Correlation
    ax4 = fig.add_subplot(gs[1, 0])   # LISA cluster count
    ax5 = fig.add_subplot(gs[1, 1:])  # Top policy typologies

    BAR_COL_Y = '#2980b9'
    BAR_COL_X = '#e07b39'
    BAR_W = 0.52

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        _ia_setup_card_axes(ax, with_grid=True)

    # ── Variable Y inequality ─────────────────────────────────────────────────
    vals_y = [gini_y, theil_y, hoover_y]
    ax1.bar([0, 1, 2], vals_y, width=BAR_W, color=BAR_COL_Y, zorder=3)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(['Gini', 'Theil', 'Hoover'], fontsize=8.5)
    ax1.set_title(label_y, fontsize=9.5, fontweight='bold', color='#1a1a1a', pad=5)
    ylim1 = max(1.0, np.nanmax([v for v in vals_y if np.isfinite(v)] + [0.1]) * 1.28)
    ax1.set_ylim(0, ylim1)
    for i, v in enumerate(vals_y):
        if np.isfinite(v):
            ax1.text(i, v + ylim1 * 0.02, '{:.3f}'.format(v),
                     ha='center', va='bottom', fontsize=7.8, color='#333')

    # ── Variable X inequality ─────────────────────────────────────────────────
    vals_x = [gini_x, theil_x, hoover_x]
    ax2.bar([0, 1, 2], vals_x, width=BAR_W, color=BAR_COL_X, zorder=3)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['Gini', 'Theil', 'Hoover'], fontsize=8.5)
    ax2.set_title(label_x, fontsize=9.5, fontweight='bold', color='#1a1a1a', pad=5)
    ylim2 = max(1.0, np.nanmax([v for v in vals_x if np.isfinite(v)] + [0.1]) * 1.28)
    ax2.set_ylim(0, ylim2)
    for i, v in enumerate(vals_x):
        if np.isfinite(v):
            ax2.text(i, v + ylim2 * 0.02, '{:.3f}'.format(v),
                     ha='center', va='bottom', fontsize=7.8, color='#333')

    # ── Correlation ───────────────────────────────────────────────────────────
    pear_v  = pearson  if np.isfinite(pearson)  else 0.0
    spear_v = spearman if np.isfinite(spearman) else 0.0
    colors_corr = ['#2980b9' if pear_v  >= 0 else '#c0392b',
                   '#2980b9' if spear_v >= 0 else '#c0392b']
    ax3.bar([0, 1], [pear_v, spear_v], width=0.48, color=colors_corr, zorder=3)
    ax3.axhline(0, linestyle='--', linewidth=0.9, color='#9e968d', zorder=2)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Pearson', 'Spearman'], fontsize=8.5)
    ax3.set_ylim(-1.10, 1.10)
    ax3.set_title('Correlation', fontsize=9.5, fontweight='bold', color='#1a1a1a', pad=5)
    for i, v in enumerate([pear_v, spear_v]):
        yoff = 0.05 if v >= 0 else -0.14
        ax3.text(i, v + yoff, '{:.3f}'.format(v),
                 ha='center', va='bottom', fontsize=7.8, color='#333')

    # ── LISA cluster count ────────────────────────────────────────────────────
    lisa_labels = ['NS', 'HH', 'LL', 'LH', 'HL']
    lisa_codes  = [0, 1, 2, 3, 4]
    y_counts = [sum(1 for v in spatial_y.get('cluster_code', []) if v == c) for c in lisa_codes]
    x_counts = [sum(1 for v in spatial_x.get('cluster_code', []) if v == c) for c in lisa_codes]
    ind = np.arange(len(lisa_labels))
    bw  = 0.35
    ax4.bar(ind - bw / 2, y_counts, width=bw, color=BAR_COL_Y, label=label_y, zorder=3)
    ax4.bar(ind + bw / 2, x_counts, width=bw, color=BAR_COL_X, label=label_x, zorder=3)
    ax4.set_xticks(ind)
    ax4.set_xticklabels(lisa_labels, fontsize=8.5)
    ax4.set_title('LISA Cluster Count', fontsize=9.5, fontweight='bold', color='#1a1a1a', pad=5)
    leg4 = ax4.legend(fontsize=7.5, frameon=True, loc='upper right',
                      handlelength=1.1, handleheight=0.9)
    leg4.get_frame().set_facecolor('white')
    leg4.get_frame().set_edgecolor('#ddd5cb')
    leg4.get_frame().set_linewidth(0.7)

    # ── Top policy typologies ─────────────────────────────────────────────────
    typ_counts = {}
    for t in priority.get('typology', []):
        if t and t != 'No Data':
            typ_counts[t] = typ_counts.get(t, 0) + 1
    top_typ = sorted(typ_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    if top_typ:
        # Truncate long label strings so they don't overflow left
        MAX_CHARS = 28
        def _trunc(s): return s if len(s) <= MAX_CHARS else s[:MAX_CHARS - 1] + '…'
        labels_typ = [_trunc(x[0]) for x in reversed(top_typ)]
        vals_typ   = [x[1]         for x in reversed(top_typ)]
        cmap = matplotlib.colormaps.get_cmap('Blues') if hasattr(matplotlib, 'colormaps') else plt.cm.get_cmap('Blues')
        bar_colors = [cmap(0.38 + 0.10 * i) for i in range(len(labels_typ))]
        bars = ax5.barh(range(len(labels_typ)), vals_typ, color=bar_colors, height=0.62, zorder=3)
        ax5.set_yticks(range(len(labels_typ)))
        ax5.set_yticklabels(labels_typ, fontsize=8.2)
        ax5.set_title('Top Policy Typologies', fontsize=9.5, fontweight='bold', color='#1a1a1a', pad=5)
        ax5.set_xlabel('Feature count', fontsize=8.0, labelpad=4)
        max_v = max(vals_typ) if vals_typ else 1
        for bar, v in zip(bars, vals_typ):
            ax5.text(v + max_v * 0.015, bar.get_y() + bar.get_height() / 2,
                     str(v), va='center', ha='left', fontsize=7.8, color='#333')
        # Ensure x-axis has a small right margin so value labels aren't clipped
        ax5.set_xlim(0, max_v * 1.14)
    else:
        ax5.set_visible(False)

    fig.savefig(path, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def _ia_write_html_report(path, title, label_y, label_x, metrics, image_paths, priority_csv_name):
    def esc(v):
        return html.escape(str(v))
    def img_tag(p, caption):
        if not p:
            return ''
        fn = os.path.basename(p)
        return '<figure><img src="{}" alt="{}"><figcaption>{}</figcaption></figure>'.format(esc(fn), esc(caption), esc(caption))
    priority_rows = ''.join('<tr><td>{}</td><td>{}</td></tr>'.format(esc(k), esc(v)) for k, v in metrics.items())
    imgs = ''.join(img_tag(p, cap) for p, cap in image_paths)
    html_text = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>{title}</title>
<style>
body {{ margin:0; font-family: Arial, sans-serif; background:#f7f5f2; color:#1d1a17; }}
.hero {{ padding:38px 52px 24px 52px; background:linear-gradient(120deg,#f7f5f2,#eef7f6); border-bottom:1px solid #ddd5cb; }}
.kicker {{ color:#1f7a8c; font-weight:700; letter-spacing:1.5px; font-size:12px; text-transform:uppercase; }}
h1 {{ margin:8px 0 6px 0; font-size:34px; }}
.subtitle {{ color:#615a53; font-style:italic; }}
section {{ padding:26px 52px; }}
.card {{ background:white; border:1px solid #ddd5cb; border-radius:16px; padding:18px 22px; box-shadow:0 10px 28px rgba(45,40,35,.07); margin-bottom:24px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
td, th {{ padding:10px 12px; border-bottom:1px solid #eee8df; text-align:left; }}
th {{ background:#f2eee7; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
figure {{ background:white; border:1px solid #ddd5cb; border-radius:16px; padding:12px; margin:0; }}
img {{ width:100%; height:auto; border-radius:10px; display:block; }}
figcaption {{ color:#655e57; font-size:13px; margin-top:8px; }}
.note {{ font-size:13px; color:#625b54; line-height:1.55; }}
</style>
</head>
<body>
<div class=\"hero\">
  <div class=\"kicker\">EquiMap v1.3.1</div>
  <h1>{title}</h1>
  <div class=\"subtitle\">Spatial inequality intelligence report for {label_y} versus {label_x}</div>
</div>
<section>
  <div class=\"card\">
    <h2>Executive Metrics</h2>
    <table>{priority_rows}</table>
  </div>
  <div class=\"card note\">
    <b>Interpretation note.</b> Priority score treats Field 1/Y as pressure, need, intensity, or exposure, while Field 2/X is treated as service, capacity, support, or balancing factor. Adjust the interpretation according to the selected indicators. Priority ranking is available in <b>{priority_csv}</b>.
  </div>
  <div class=\"grid\">{imgs}</div>
</section>
</body>
</html>""".format(title=esc(title), label_y=esc(label_y), label_x=esc(label_x), priority_rows=priority_rows, priority_csv=esc(priority_csv_name), imgs=imgs)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(html_text)


class BivariatePosterMapAlgorithm(QgsProcessingAlgorithm):

    INPUT = 'INPUT'
    FIELD1 = 'FIELD1'
    FIELD2 = 'FIELD2'
    LABEL1 = 'LABEL1'
    LABEL2 = 'LABEL2'
    NUM_CLASSES = 'NUM_CLASSES'
    COLOR_SCHEME = 'COLOR_SCHEME'
    MAP_RENDER_MODE = 'MAP_RENDER_MODE'

    INPUT_DEM = 'INPUT_DEM'

    MAP_TITLE = 'MAP_TITLE'
    MAP_SUBTITLE = 'MAP_SUBTITLE'
    FOOTER_TEXT = 'FOOTER_TEXT'

    MAX_RASTER_DIM = 'MAX_RASTER_DIM'
    HILLSHADE_AZIMUTH = 'HILLSHADE_AZIMUTH'
    HILLSHADE_ALTITUDE = 'HILLSHADE_ALTITUDE'
    RELIEF_Z_FACTOR = 'RELIEF_Z_FACTOR'
    DRAPE_STRENGTH = 'DRAPE_STRENGTH'
    SHADE_STRENGTH = 'SHADE_STRENGTH'
    SHADOW_ALPHA = 'SHADOW_ALPHA'
    SHOW_BOUNDARIES = 'SHOW_BOUNDARIES'
    ALL_TOUCHED = 'ALL_TOUCHED'

    RUN_INEQUALITY = 'RUN_INEQUALITY'
    DO_SPATIAL = 'DO_SPATIAL'
    WEIGHT_MODE = 'WEIGHT_MODE'
    KNN_K = 'KNN_K'
    PERMUTATIONS = 'PERMUTATIONS'
    ALPHA = 'ALPHA'
    OUTPUT_ANALYTICS_FOLDER = 'OUTPUT_ANALYTICS_FOLDER'

    OUTPUT = 'OUTPUT'
    QML = 'QML'
    LEGEND = 'LEGEND'
    MAP = 'MAP'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return BivariatePosterMapAlgorithm()

    def name(self):
        return 'equimap_sit'

    def displayName(self):
        return self.tr('EquiMap: Spatial Inequality Intelligence')

    def shortHelpString(self):
        return self.tr(
            "<p><b>Created By: Firman Afrianto, Maya Safira</b></p>"

            "<p>"
            "<b>EquiMap: Spatial Inequality Intelligence</b> is a QGIS Processing tool for producing "
            "a modern bivariate poster map and spatial inequality analysis based on two numeric polygon attributes. "
            "It combines bivariate choropleth mapping, optional relief-draped mapping, distributional inequality metrics, "
            "Global Moran's I, Local Moran/LISA, Bivariate Moran, BiLISA, priority scoring, policy typology, "
            "modern PNG outputs, CSV tables, and an HTML report."
            "</p>"

            "<p><b>Core EquiMap Components</b></p>"
            "<ul>"
            "<li><b>Bivariate poster mapping</b> with quantile classes and multiple bivariate palette options.</li>"
            "<li><b>Dynamic layout</b> automatically adapts the poster to landscape, portrait, or balanced map shapes.</li>"
            "<li><b>Relief-draped poster</b> optionally uses a DEM and hillshade for presentation-ready terrain-based visualization.</li>"
            "<li><b>Inequality analytics</b> including Gini, Theil T, Hoover, Lorenz curve, and concentration curve.</li>"
            "<li><b>Spatial autocorrelation</b> including Global Moran's I, Local Moran/LISA, Bivariate Moran, and BiLISA using Queen, Rook, or KNN weights.</li>"
            "<li><b>Planning intelligence</b> through mismatch scoring, priority intervention ranking, and automatic policy typology.</li>"
            "</ul>"

            "<p><b>Variable Selection Rule</b></p>"
            "<p>"
            "EquiMap uses a directional interpretation. <b>Field 1 is the Y variable</b> and "
            "<b>Field 2 is the X variable</b>. The Y variable increases upward in the bivariate matrix, "
            "while the X variable increases to the right. Because priority scoring and mismatch typology are directional, "
            "the order of the two variables matters."
            "</p>"

            "<p><b>Recommended Interpretation of Field 1 / Y</b></p>"
            "<ul>"
            "<li>Use <b>Field 1 / Y</b> for variables representing <b>need, pressure, demand, exposure, vulnerability, risk, or problem intensity</b>.</li>"
            "<li>Examples: vulnerable population, poverty, population density, flood risk, land surface temperature, built-up pressure, service demand, emissions, congestion, or disaster exposure.</li>"
            "</ul>"

            "<p><b>Recommended Interpretation of Field 2 / X</b></p>"
            "<ul>"
            "<li>Use <b>Field 2 / X</b> for variables representing <b>service capacity, support, supply, access, infrastructure, environmental buffer, or resilience factor</b>.</li>"
            "<li>Examples: health facilities, schools, transport access, green open space, evacuation facilities, road capacity, clean water network, public service coverage, or infrastructure investment.</li>"
            "</ul>"

            "<p><b>Why Variable Order Matters</b></p>"
            "<ul>"
            "<li><b>High Y and Low X</b> is interpreted as a critical mismatch: high need or pressure with low support or capacity.</li>"
            "<li><b>High Y and High X</b> may indicate an intensive but relatively supported area.</li>"
            "<li><b>Low Y and High X</b> may indicate potential oversupply, spare capacity, or low-demand high-support area.</li>"
            "<li><b>Low Y and Low X</b> may indicate low demand and low service, or a low-intensity area requiring monitoring.</li>"
            "</ul>"

            "<p><b>Examples of Good Y-X Pairings</b></p>"
            "<ul>"
            "<li><b>Y = vulnerable population</b>, <b>X = health facilities</b>: identifies underserved vulnerable areas.</li>"
            "<li><b>Y = population density</b>, <b>X = green open space</b>: identifies urban pressure with insufficient environmental balancing capacity.</li>"
            "<li><b>Y = flood risk</b>, <b>X = evacuation facilities</b>: identifies disaster-risk areas with weak evacuation capacity.</li>"
            "<li><b>Y = poverty</b>, <b>X = transport access</b>: identifies mobility deprivation and accessibility inequality.</li>"
            "<li><b>Y = land surface temperature</b>, <b>X = vegetation or green cover</b>: identifies environmental heat inequality.</li>"
            "</ul>"

            "<p><b>Special Cases</b></p>"
            "<ul>"
            "<li>If both variables are negative indicators, such as poverty and unemployment, interpret <b>High Y and High X</b> as a <b>double-burden area</b>, not as a service mismatch.</li>"
            "<li>If both variables are positive indicators, such as income and accessibility, interpret the map as a co-advantage or development structure, not automatically as deprivation.</li>"
            "<li>If the goal is service equity, the safest setup is: <b>Y = demand, need, risk, or vulnerability</b>; <b>X = supply, service, capacity, access, or resilience factor</b>.</li>"
            "</ul>"

            "<p><b>LISA, BiLISA, and Moran Interpretation</b></p>"
            "<ul>"
            "<li><b>Univariate LISA for Y</b>: compares Y at location i with neighboring Y values.</li>"
            "<li><b>Univariate LISA for X</b>: compares X at location i with neighboring X values.</li>"
            "<li><b>BiLISA Y vs lag(X)</b>: compares Y at location i with neighboring X values. This is useful for identifying high-need areas surrounded by low-support environments.</li>"
            "<li><b>BiLISA X vs lag(Y)</b>: compares X at location i with neighboring Y values. This is useful for reading the spatial relationship between support capacity and surrounding need.</li>"
            "<li><b>Bivariate Moran</b> is the global counterpart of BiLISA and summarizes the overall spatial cross-association between one local variable and the spatial lag of the other variable.</li>"
            "</ul>"

            "<p><b>Map Render Modes</b></p>"
            "<ul>"
            "<li><b>Flat 2D Poster</b>: a polygon-based 2D bivariate map without DEM.</li>"
            "<li><b>Relief Draped Poster</b>: bivariate classes are rasterized and draped over a shaded-relief DEM.</li>"
            "</ul>"

            "<p><b>Main Outputs</b></p>"
            "<ul>"
            "<li><b>Classified output layer</b> with bivariate class fields: biv_y_cls, biv_x_cls, biv_class, and biv_label.</li>"
            "<li><b>Extended analytics fields</b> for inequality, z-score, spatial lag, Moran, LISA, Bivariate Moran, BiLISA, priority score, policy typology, and recommendation.</li>"
            "<li><b>QML style</b>, <b>bivariate legend PNG</b>, and <b>bivariate poster map PNG</b>.</li>"
            "<li><b>Analytical PNG outputs</b>: Lorenz curve, concentration curve, Moran scatterplot, Bivariate Moran scatterplot, LISA map, BiLISA map, and summary dashboard.</li>"
            "<li><b>CSV outputs</b>: summary metrics, Lorenz points, concentration points, and priority ranking.</li>"
            "<li><b>HTML report</b> combining metrics, charts, maps, priority interpretation, and recommendations.</li>"
            "</ul>"

            "<p><b>Important Notes</b></p>"
            "<ul>"
            "<li>The main input must be a <b>polygon layer</b>.</li>"
            "<li>Both selected fields must be <b>numeric</b>.</li>"
            "<li>Relief mode requires a <b>DEM</b> with the same CRS as the polygon layer.</li>"
            "<li>Moran, LISA, Bivariate Moran, and BiLISA require <b>libpysal</b>, <b>esda</b>, and <b>shapely</b>.</li>"
            "<li>For contiguity weights, use <b>Queen</b> for general polygon adjacency and <b>Rook</b> for stricter edge-sharing adjacency. Use <b>KNN</b> when polygons are disconnected or contain islands.</li>"
            "</ul>"
        )

    def initAlgorithm(self, config=None):

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input polygon layer'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )

        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD1,
                self.tr('Field 1 / Y axis'),
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric
            )
        )

        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD2,
                self.tr('Field 2 / X axis'),
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.LABEL1,
                self.tr('Label for Y axis'),
                defaultValue='Variable Y'
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.LABEL2,
                self.tr('Label for X axis'),
                defaultValue='Variable X'
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.NUM_CLASSES,
                self.tr('Number of classes'),
                options=['3', '4'],
                defaultValue=0
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.COLOR_SCHEME,
                self.tr('Bivariate color palette'),
                options=[
                    'BlueGill',
                    'BlueGold',
                    'BlueOr',
                    'BlueYl',
                    'Brown2',
                    'DkBlue2',
                    'DkCyan2',
                    'DkViolet2',
                    'GrPink2',
                    'PinkGrn',
                    'PurpleGrn',
                    'PurpleOr'
                ],
                defaultValue=0
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.MAP_RENDER_MODE,
                self.tr('Map render mode'),
                options=[
                    'Flat 2D Poster',
                    'Relief Draped Poster'
                ],
                defaultValue=0
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                self.tr('Input DEM raster, required only for Relief Draped Poster'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.MAP_TITLE,
                self.tr('Map title'),
                defaultValue='EquiMap: Spatial Inequality Intelligence'
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.MAP_SUBTITLE,
                self.tr('Map subtitle'),
                defaultValue=''
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.FOOTER_TEXT,
                self.tr('Footer text'),
                defaultValue='Generated by..............'
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_RASTER_DIM,
                self.tr('Maximum raster dimension for relief processing'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1800,
                minValue=300
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.HILLSHADE_AZIMUTH,
                self.tr('Hillshade azimuth'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=120.0,
                minValue=0.0,
                maxValue=360.0
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.HILLSHADE_ALTITUDE,
                self.tr('Hillshade altitude'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=45.0,
                minValue=1.0,
                maxValue=90.0
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.RELIEF_Z_FACTOR,
                self.tr('Relief vertical exaggeration factor'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=2.5,
                minValue=0.1,
                maxValue=10.0
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.DRAPE_STRENGTH,
                self.tr('Drape strength'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.75,
                minValue=0.0,
                maxValue=1.0
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHADE_STRENGTH,
                self.tr('Hillshade strength'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.65,
                minValue=0.0,
                maxValue=1.5
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHADOW_ALPHA,
                self.tr('Soft shadow opacity'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.15,
                minValue=0.0,
                maxValue=1.0
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SHOW_BOUNDARIES,
                self.tr('Draw polygon boundaries on top'),
                defaultValue=False
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ALL_TOUCHED,
                self.tr('Rasterize with all_touched, relief mode only'),
                defaultValue=True
            )
        )


        self.addParameter(
            QgsProcessingParameterBoolean(
                self.RUN_INEQUALITY,
                self.tr('Run inequality and concentration analysis'),
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.DO_SPATIAL,
                self.tr('Run Moran and LISA analysis'),
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_MODE,
                self.tr('Spatial weight mode'),
                options=['Queen', 'Rook', 'KNN'],
                defaultValue=0
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN_K,
                self.tr('K for KNN weights'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=8,
                minValue=1
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PERMUTATIONS,
                self.tr('Permutations for Moran and LISA'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=999,
                minValue=99
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.ALPHA,
                self.tr('Significance level alpha for LISA'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.05,
                minValue=0.001,
                maxValue=0.20
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_ANALYTICS_FOLDER,
                self.tr('Main output folder for QML, legend, poster map, analytics PNG, CSV, and HTML report'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Classified output layer')
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUTPUT_ANALYTICS_FOLDER,
                self.tr('Main output folder')
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.QML,
                self.tr('QML style file')
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.LEGEND,
                self.tr('Legend PNG')
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.MAP,
                self.tr('Poster map PNG')
            )
        )

    # =========================================================
    # General helpers
    # =========================================================

    def _safe_float(self, value):
        try:
            if value is None:
                return None
            f = float(value)
            if math.isfinite(f):
                return f
            return None
        except Exception:
            return None

    def _hex_to_rgb(self, value):
        value = value.lstrip('#')
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(self, rgb):
        r = max(0, min(255, int(round(rgb[0]))))
        g = max(0, min(255, int(round(rgb[1]))))
        b = max(0, min(255, int(round(rgb[2]))))
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def _lerp_rgb(self, rgb1, rgb2, t):
        return (
            self._lerp(rgb1[0], rgb2[0], t),
            self._lerp(rgb1[1], rgb2[1], t),
            self._lerp(rgb1[2], rgb2[2], t)
        )

    def _bilinear_color(self, top_left, top_right, bottom_left, bottom_right, tx, ty):
        bottom = self._lerp_rgb(bottom_left, bottom_right, tx)
        top = self._lerp_rgb(top_left, top_right, tx)
        return self._lerp_rgb(bottom, top, ty)

    def _format_breaks(self, breaks):
        if not breaks:
            return 'No breaks — all valid values are identical'
        return ', '.join(['{:.6g}'.format(float(x)) for x in breaks])

    def _axis_labels(self, ncls):
        if ncls == 3:
            return ['Low', 'Med', 'High']
        return ['Low', 'Low-Med', 'High-Med', 'High']

    def _quantile_breaks(self, values, k):
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            raise QgsProcessingException('No valid numeric values were found for classification.')

        if np.all(arr == arr[0]):
            return []

        quantiles = [i / float(k) for i in range(1, k)]
        return list(np.quantile(arr, quantiles))

    def _classify(self, value, breaks):
        """
        [FIX] Added guard for empty breaks list (all-identical values case).
        Returns class 0 when there are no break boundaries.
        """
        if value is None:
            return None

        # Empty breaks means all values are identical — assign class 0
        if not breaks:
            return 0

        for i, brk in enumerate(breaks):
            if value <= brk:
                return i

        return len(breaks)

    def _push_file_link(self, feedback, label, path):
        try:
            if path:
                p = os.path.abspath(path)
                feedback.pushInfo(f'{label}: <a href="file:///{p.replace(os.sep, "/")}">{p}</a>')
        except Exception:
            pass

    def _geom_as_json(self, geom):
        """
        [FIX] Robust geometry-to-JSON conversion.
        QGIS 3.x: asJson() is fine. Older QGIS versions may need exportToGeoJSON().
        Falls back gracefully.
        """
        try:
            return json.loads(geom.asJson())
        except AttributeError:
            try:
                return json.loads(geom.exportToGeoJSON())
            except Exception:
                return None

    # =========================================================
    # Palette
    # =========================================================

    def _palette_anchors(self):
        return {
            'BlueGill': {
                'tl': '#a56c21',
                'tr': '#6a641d',
                'bl': '#d9d9d9',
                'br': '#77a8a2'
            },
            'BlueGold': {
                'tl': '#dfa600',
                'tr': '#5e7f00',
                'bl': '#d9d9d9',
                'br': '#4d91b0'
            },
            'BlueOr': {
                'tl': '#25a7d8',
                'tr': '#155d2c',
                'bl': '#d9d9d9',
                'br': '#dd6f2b'
            },
            'BlueYl': {
                'tl': '#198bd0',
                'tr': '#009300',
                'bl': '#d9d9d9',
                'br': '#d9c000'
            },
            'Brown2': {
                'tl': '#8a6da1',
                'tr': '#7d5943',
                'bl': '#d9d9d9',
                'br': '#b3a24f'
            },
            'DkBlue2': {
                'tl': '#a35ea0',
                'tr': '#4c5897',
                'bl': '#d9d9d9',
                'br': '#57b3b1'
            },
            'DkCyan2': {
                'tl': '#74a678',
                'tr': '#35595d',
                'bl': '#d9d9d9',
                'br': '#687fb0'
            },
            'DkViolet2': {
                'tl': '#4b7eb6',
                'tr': '#341f45',
                'bl': '#d9d9d9',
                'br': '#a6394a'
            },
            'GrPink2': {
                'tl': '#5f9cad',
                'tr': '#574650',
                'bl': '#d9d9d9',
                'br': '#b85757'
            },
            'PinkGrn': {
                'tl': '#c22082',
                'tr': '#520916',
                'bl': '#d9d9d9',
                'br': '#49a01f'
            },
            'PurpleGrn': {
                'tl': '#72348f',
                'tr': '#002a2d',
                'bl': '#d9d9d9',
                'br': '#0a8b30'
            },
            'PurpleOr': {
                'tl': '#62409a',
                'tr': '#5d1800',
                'bl': '#d9d9d9',
                'br': '#d96b00'
            }
        }

    def gen_colors(self, scheme_name, ncls):
        anchors = self._palette_anchors()

        if scheme_name not in anchors:
            raise QgsProcessingException('Unknown palette name: {}'.format(scheme_name))

        tl = self._hex_to_rgb(anchors[scheme_name]['tl'])
        tr = self._hex_to_rgb(anchors[scheme_name]['tr'])
        bl = self._hex_to_rgb(anchors[scheme_name]['bl'])
        br = self._hex_to_rgb(anchors[scheme_name]['br'])

        colors = {}

        for ycls in range(ncls):
            for xcls in range(ncls):
                tx = 0.0 if ncls == 1 else xcls / float(ncls - 1)
                ty = 0.0 if ncls == 1 else ycls / float(ncls - 1)
                rgb = self._bilinear_color(tl, tr, bl, br, tx, ty)
                idx = ycls * ncls + xcls
                colors[idx] = self._rgb_to_hex(rgb)

        return colors

    # =========================================================
    # Dynamic layout
    # =========================================================

    def _get_dynamic_layout(self, features_info):
        """
        Returns layout settings based on map aspect ratio.
        aspect = width / height
        """

        xmin, xmax, ymin, ymax = self._extent_padded_from_features(features_info)

        width = max(1e-9, xmax - xmin)
        height = max(1e-9, ymax - ymin)
        aspect = width / height

        if aspect < 0.72:
            return {
                'layout_name': 'portrait',
                'figsize': (10, 14),
                'map_ax': [0.10, 0.245, 0.80, 0.615],
                'legend': {
                    'card_left': 0.060,
                    'card_bottom': 0.045,
                    'card_width': 0.360,
                    'card_height': 0.155,
                    'grid_left_offset': 0.150,
                    'grid_bottom_offset': 0.058,
                    'grid_size': 0.085,
                    'label_offset': 0.043,
                    'tick_font': 6.8,
                    'axis_font': 8.2,
                    'card_alpha': 0.88
                },
                'title_y': 0.945,
                'subtitle_y': 0.915,
                'footer_y': 0.065,
                'title_size_base': 28,
                'subtitle_size_base': 16
            }

        elif aspect > 1.35:
            return {
                'layout_name': 'landscape',
                'figsize': (14, 10),
                'map_ax': [0.025, 0.245, 0.95, 0.635],
                'legend': {
                    'card_left': 0.035,
                    'card_bottom': 0.035,
                    'card_width': 0.250,
                    'card_height': 0.180,
                    'grid_left_offset': 0.105,
                    'grid_bottom_offset': 0.070,
                    'grid_size': 0.095,
                    'label_offset': 0.047,
                    'tick_font': 7.2,
                    'axis_font': 8.8,
                    'card_alpha': 0.88
                },
                'title_y': 0.935,
                'subtitle_y': 0.895,
                'footer_y': 0.055,
                'title_size_base': 28,
                'subtitle_size_base': 17
            }

        else:
            return {
                'layout_name': 'balanced',
                'figsize': (11.5, 11.5),
                'map_ax': [0.060, 0.245, 0.88, 0.635],
                'legend': {
                    'card_left': 0.050,
                    'card_bottom': 0.040,
                    'card_width': 0.290,
                    'card_height': 0.170,
                    'grid_left_offset': 0.120,
                    'grid_bottom_offset': 0.064,
                    'grid_size': 0.090,
                    'label_offset': 0.045,
                    'tick_font': 7.0,
                    'axis_font': 8.5,
                    'card_alpha': 0.88
                },
                'title_y': 0.935,
                'subtitle_y': 0.900,
                'footer_y': 0.060,
                'title_size_base': 28,
                'subtitle_size_base': 16
            }

    # =========================================================
    # QML
    # =========================================================

    def _apply_and_save_qml(self, dest_id, context, colors, label_map, qml_path, feedback):
        out_layer = QgsProcessingUtils.mapLayerFromString(dest_id, context)

        if out_layer is None:
            feedback.pushWarning('Could not access the output layer for QML export.')
            return

        categories = []

        for idx in sorted(colors.keys()):
            label = label_map[idx]
            symbol = QgsSymbol.defaultSymbol(out_layer.geometryType())

            if symbol is None:
                continue

            symbol.setColor(QColor(colors[idx]))

            try:
                symbol.symbolLayer(0).setStrokeColor(QColor('#e8e8e8'))
                symbol.symbolLayer(0).setStrokeWidth(0.15)
            except Exception:
                pass

            categories.append(QgsRendererCategory(label, symbol, label))

        symbol_nd = QgsSymbol.defaultSymbol(out_layer.geometryType())

        if symbol_nd is not None:
            symbol_nd.setColor(QColor('#efefef'))

            try:
                symbol_nd.symbolLayer(0).setStrokeColor(QColor('#d9d9d9'))
                symbol_nd.symbolLayer(0).setStrokeWidth(0.15)
            except Exception:
                pass

            categories.append(QgsRendererCategory('No Data', symbol_nd, 'No Data'))

        renderer = QgsCategorizedSymbolRenderer('biv_label', categories)
        out_layer.setRenderer(renderer)
        out_layer.triggerRepaint()

        ok, msg = out_layer.saveNamedStyle(qml_path)

        if not ok:
            feedback.pushWarning('QML export failed: {}'.format(msg))

    # =========================================================
    # Geometry helpers
    # =========================================================

    def _extent_padded_from_features(self, features_info):
        xmin, ymin, xmax, ymax = None, None, None, None

        for item in features_info:
            geom = item['geom']

            if geom is None or geom.isEmpty():
                continue

            bbox = geom.boundingBox()

            if xmin is None:
                xmin = bbox.xMinimum()
                ymin = bbox.yMinimum()
                xmax = bbox.xMaximum()
                ymax = bbox.yMaximum()
            else:
                xmin = min(xmin, bbox.xMinimum())
                ymin = min(ymin, bbox.yMinimum())
                xmax = max(xmax, bbox.xMaximum())
                ymax = max(ymax, bbox.yMaximum())

        if xmin is None:
            return (0, 1, 0, 1)

        padx = (xmax - xmin) * 0.006 if xmax > xmin else 1
        pady = (ymax - ymin) * 0.006 if ymax > ymin else 1

        return (xmin - padx, xmax + padx, ymin - pady, ymax + pady)

    def _draw_polygon_geom(self, ax, geom, facecolor, edgecolor='#e8e8e8', lw=0.30, alpha=1.0, zorder=2):
        """
        [IMPROVE] Default edge color updated to soft #e8e8e8 for cleaner cartographic look.
        """
        if geom is None or geom.isEmpty():
            return

        try:
            if geom.isMultipart():
                multi_poly = geom.asMultiPolygon()

                for poly in multi_poly:
                    if not poly:
                        continue

                    outer = poly[0]

                    if outer:
                        xs = [pt.x() for pt in outer]
                        ys = [pt.y() for pt in outer]

                        ax.fill(
                            xs, ys,
                            facecolor=facecolor,
                            edgecolor=edgecolor,
                            linewidth=lw,
                            alpha=alpha,
                            zorder=zorder,
                            joinstyle='round'
                        )

                    for hole in poly[1:]:
                        if hole:
                            xs = [pt.x() for pt in hole]
                            ys = [pt.y() for pt in hole]

                            ax.fill(
                                xs, ys,
                                facecolor='#f7f5f2',
                                edgecolor=edgecolor,
                                linewidth=lw,
                                alpha=1.0,
                                zorder=zorder + 1,
                                joinstyle='round'
                            )
            else:
                poly = geom.asPolygon()

                if not poly:
                    return

                outer = poly[0]

                if outer:
                    xs = [pt.x() for pt in outer]
                    ys = [pt.y() for pt in outer]

                    ax.fill(
                        xs, ys,
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        linewidth=lw,
                        alpha=alpha,
                        zorder=zorder,
                        joinstyle='round'
                    )

                for hole in poly[1:]:
                    if hole:
                        xs = [pt.x() for pt in hole]
                        ys = [pt.y() for pt in hole]

                        ax.fill(
                            xs, ys,
                            facecolor='#f7f5f2',
                            edgecolor=edgecolor,
                            linewidth=lw,
                            alpha=1.0,
                            zorder=zorder + 1,
                            joinstyle='round'
                        )

        except Exception:
            pass

    def _draw_boundaries(self, ax, features_info, color='#cccccc', lw=0.22, alpha=0.40):
        """
        [IMPROVE] Boundary color softened to #cccccc, thinner stroke.
        """
        for item in features_info:
            geom = item['geom']

            if geom is None or geom.isEmpty():
                continue

            try:
                if geom.isMultipart():
                    multi_poly = geom.asMultiPolygon()

                    for poly in multi_poly:
                        if not poly:
                            continue

                        outer = poly[0]
                        xs = [pt.x() for pt in outer]
                        ys = [pt.y() for pt in outer]

                        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, zorder=5)
                else:
                    poly = geom.asPolygon()

                    if not poly:
                        continue

                    outer = poly[0]
                    xs = [pt.x() for pt in outer]
                    ys = [pt.y() for pt in outer]

                    ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, zorder=5)
            except Exception:
                pass

    # =========================================================
    # DEM and raster helpers
    # =========================================================

    def _resolve_raster_path(self, qgs_raster_layer):
        path = qgs_raster_layer.source()

        if '|' in path:
            path = path.split('|')[0]

        return path

    def _read_dem(self, dem_layer, max_dim, feedback):
        """
        [FIX] Affine import made robust.
        rasterio >= 1.0 bundles its own Affine; importing from rasterio.transform
        is preferred. The standalone 'affine' package is used as fallback.
        """
        try:
            import rasterio
            from rasterio.enums import Resampling
        except Exception as e:
            raise QgsProcessingException(
                "rasterio is not available in the QGIS Python environment. "
                "Please install rasterio first. Details: {}".format(e)
            )

        # [FIX] Robust Affine import
        try:
            from rasterio.transform import Affine
        except ImportError:
            try:
                from affine import Affine
            except ImportError:
                raise QgsProcessingException(
                    "The affine transform library is not available. "
                    "Make sure rasterio is installed correctly."
                )

        path = self._resolve_raster_path(dem_layer)

        with rasterio.open(path) as src:
            h = src.height
            w = src.width

            if max(h, w) > max_dim:
                scale = float(max(h, w)) / float(max_dim)
                new_h = max(1, int(round(h / scale)))
                new_w = max(1, int(round(w / scale)))

                arr = src.read(
                    1,
                    out_shape=(new_h, new_w),
                    resampling=Resampling.bilinear
                ).astype(np.float32)

                transform = src.transform * Affine.scale(
                    src.width / float(new_w),
                    src.height / float(new_h)
                )
            else:
                arr = src.read(1).astype(np.float32)
                transform = src.transform

            nodata = src.nodata

            if nodata is not None:
                arr[arr == nodata] = np.nan

            left = transform.c
            top = transform.f
            right = left + arr.shape[1] * transform.a
            bottom = top + arr.shape[0] * transform.e

            extent = (left, right, bottom, top)

            return arr, transform, extent

    def _smooth_array(self, arr, iterations=1):
        a = np.array(arr, dtype=float)

        finite = np.isfinite(a)

        if finite.any():
            fill = np.nanmedian(a[finite])
            a = np.where(np.isfinite(a), a, fill)

        for _ in range(iterations):
            p = np.pad(a, 1, mode='edge')

            a = (
                p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
                p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
            ) / 9.0

        return a

    def _single_hillshade(self, arr_smooth, azimuth, altitude, z_factor):
        """
        Compute one hillshade pass from a pre-smoothed DEM array.
        Returns raw shaded values in [0, 1] with full dynamic range.
        """
        dy, dx = np.gradient(arr_smooth)

        dx = dx * z_factor
        dy = dy * z_factor

        slope  = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
        aspect = np.arctan2(-dx, dy)

        az  = np.deg2rad(azimuth)
        alt = np.deg2rad(altitude)

        shaded = (
            np.sin(alt) * np.sin(slope) +
            np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
        )

        # Stretch to full 0-1 range preserving all contrast
        lo, hi = np.nanmin(shaded), np.nanmax(shaded)
        shaded = (shaded - lo) / (hi - lo + 1e-9)

        return np.clip(shaded, 0.0, 1.0)

    def _make_hillshade(self, dem, azimuth=315.0, altitude=45.0, z_factor=1.5):
        """
        Multi-directional hillshade (MDH) blending:
        - Primary direction   : user-defined azimuth, full weight
        - Secondary direction : +90° rotation, reduced weight
        - Tertiary direction  : −90° rotation, reduced weight
        This eliminates the "flat face" artifact of single-direction shading
        and ensures ridges/valleys are visible from all orientations.

        The final result is gamma-corrected to lift mid-tones without
        washing out bright peaks and dark valleys, giving a more tactile
        topographic feel that works well under color draping.
        """
        arr = np.array(dem, dtype=float)

        valid = np.isfinite(arr)

        if not valid.any():
            return None

        fill = np.nanmedian(arr[valid])
        arr  = np.where(np.isfinite(arr), arr, fill)

        # Light smoothing to reduce sensor noise without losing ridges
        arr_s = self._smooth_array(arr, iterations=1)

        # Three-direction blend: primary 60%, left 20%, right 20%
        hs_main  = self._single_hillshade(arr_s, azimuth,        altitude, z_factor)
        hs_left  = self._single_hillshade(arr_s, (azimuth + 90)  % 360, altitude * 0.85, z_factor)
        hs_right = self._single_hillshade(arr_s, (azimuth - 90)  % 360, altitude * 0.85, z_factor)

        blended = 0.60 * hs_main + 0.20 * hs_left + 0.20 * hs_right

        # Re-normalise after blending
        lo, hi  = blended.min(), blended.max()
        blended = (blended - lo) / (hi - lo + 1e-9)

        # Gamma correction: gamma < 1 lifts shadows, > 1 deepens them.
        # 0.72 gives good mid-tone lift while keeping valley darkness.
        gamma   = 0.72
        blended = np.power(blended, gamma)

        # Ensure a comfortable brightness floor so no area goes fully black
        blended = 0.08 + blended * 0.92

        return np.clip(blended, 0.0, 1.0)

    def _rasterize_classes(self, features_info, out_shape, transform, all_touched=True):
        """
        [FIX] Uses _geom_as_json() for robust geometry serialization.
        Skips features where geometry JSON cannot be produced.
        """
        try:
            from rasterio.features import rasterize
        except Exception as e:
            raise QgsProcessingException(
                "rasterio.features is not available. Details: {}".format(e)
            )

        class_shapes = []
        mask_shapes = []

        for item in features_info:
            geom = item['geom']
            biv = item['biv']

            geom_json = self._geom_as_json(geom)

            if geom_json is None:
                continue

            class_val = -1 if biv is None else int(biv)

            class_shapes.append((geom_json, class_val))
            mask_shapes.append((geom_json, 1))

        class_arr = rasterize(
            class_shapes,
            out_shape=out_shape,
            transform=transform,
            fill=-9999,
            dtype='int16',
            all_touched=all_touched
        )

        mask_arr = rasterize(
            mask_shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype='uint8',
            all_touched=all_touched
        )

        return class_arr, mask_arr

    def _simple_blur(self, arr, iterations=4):
        a = arr.astype(float)

        for _ in range(iterations):
            p = np.pad(a, 1, mode='edge')

            a = (
                p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
                p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
            ) / 9.0

        return a

    def _build_shadow_mask(self, mask_arr, alpha=0.12):
        """
        Improved outer shadow / vignette:
        - Erode (shrink) the mask slightly before blurring so the shadow
          starts just outside the data boundary, not inside it.
        - Use a two-stage blur: tight first pass for the sharp inner edge,
          wide second pass for the soft halo.
        - Apply a sqrt ramp so the shadow falls off more naturally.
        """
        m = mask_arr.astype(float)

        # Stage 1: tight blur creates a sharp inner falloff at the edge
        tight = self._simple_blur(m, iterations=2)

        # Stage 2: wide blur creates the soft outer halo
        wide  = self._simple_blur(m, iterations=8)

        # Combine: tight edge preserves crispness, wide halo adds depth
        combined = np.clip(0.55 * tight + 0.45 * wide, 0.0, 1.0)

        # Invert: shadow appears outside the data area, not inside it.
        # Values near 0 (outside) → high shadow; values near 1 (inside) → no shadow.
        shadow = 1.0 - combined

        # Only keep the fringe: suppress far-outside background
        shadow = np.where(shadow > 0.85, 0.0, shadow)
        shadow = np.clip(shadow / 0.85, 0.0, 1.0)

        # Sqrt ramp for a softer falloff curve
        shadow = np.sqrt(shadow)

        return shadow * alpha

    def _build_draped_rgb(self, class_arr, mask_arr, colors, hillshade, drape_strength=0.80, shade_strength=0.55):
        """
        Improved relief drape pipeline:

        1. Base colour layer  — bivariate class colours at full saturation
        2. Luminance modulation — hillshade applied in luminance space (HSV),
           not as a simple multiply. This preserves hue/saturation while
           modulating only brightness, giving rich colours in valleys and
           bright peaks that still read as their correct class colour.
        3. Screen-blend highlights — a subtle additive pass brightens only
           the very highest hillshade values (ridgelines, peaks), simulating
           specular light without blowing out to white.
        4. Multiply-blend shadows — darkens the lowest hillshade values
           (valley floors, steep shadow faces) more aggressively than the
           linear factor used previously.
        5. Outer boundary stroke — a thin 1-pixel bright edge is painted
           along the mask boundary (via morphological difference) so the
           data extent is clearly readable even against a light background.
        """
        h, w = class_arr.shape

        # -----------------------------------------------------------------
        # Background (outside data area) — warm off-white matching poster
        # -----------------------------------------------------------------
        base_rgb = np.full((h, w, 3), fill_value=0.0, dtype=float)
        base_rgb[:, :, 0] = 0.969
        base_rgb[:, :, 1] = 0.961
        base_rgb[:, :, 2] = 0.949

        valid_inside   = mask_arr > 0
        nodata_inside  = (mask_arr > 0) & (class_arr == -1)

        for idx, hexc in colors.items():
            sel = (class_arr == idx)

            if np.any(sel):
                r, g, b = self._hex_to_rgb(hexc)
                base_rgb[sel, 0] = r / 255.0
                base_rgb[sel, 1] = g / 255.0
                base_rgb[sel, 2] = b / 255.0

        if np.any(nodata_inside):
            base_rgb[nodata_inside, 0] = 0.88
            base_rgb[nodata_inside, 1] = 0.87
            base_rgb[nodata_inside, 2] = 0.86

        hs = np.array(hillshade, dtype=float)          # [0, 1]

        # -----------------------------------------------------------------
        # Luminance-space modulation
        # Convert base colour to approximate luminance, compute modulation
        # factor in that space, then reapply.
        # Factor range is wider than before: 0.40 (deep shadow) → 1.30
        # (bright ridge) — giving much stronger topographic contrast.
        # -----------------------------------------------------------------
        # shade_strength controls the contrast amplitude.
        # Default 0.55 → contrast_range ≈ 1.65 (factor 0.45..1.38)
        contrast_range = shade_strength * 1.65 + 0.65
        relief_lo = 2.0 - contrast_range          # dark shadow floor
        relief_hi = contrast_range                 # bright highlight ceiling

        relief_factor = relief_lo + (relief_hi - relief_lo) * hs
        relief_factor = np.clip(relief_factor, 0.35, 1.40)[:, :, None]

        relief_rgb = np.clip(base_rgb * relief_factor, 0.0, 1.0)

        # -----------------------------------------------------------------
        # Screen-blend highlight pass (ridgelines, sun-facing peaks)
        # Screen: out = 1 - (1-a)(1-b), brightens without blowing out
        # Only applied where hs > 0.82 (top 18% of brightness)
        # -----------------------------------------------------------------
        highlight_mask = np.clip((hs - 0.82) / 0.18, 0.0, 1.0)[:, :, None]
        highlight_rgb  = 1.0 - (1.0 - relief_rgb) * (1.0 - 0.18 * highlight_mask)

        # -----------------------------------------------------------------
        # Multiply-blend shadow deepening (valley floors, steep shadow faces)
        # Multiply: out = a*b. Only applied where hs < 0.30 (darkest 30%)
        # -----------------------------------------------------------------
        shadow_deepen  = np.clip((0.30 - hs) / 0.30, 0.0, 1.0)[:, :, None]
        darkened_rgb   = relief_rgb * (1.0 - 0.28 * shadow_deepen)

        # Blend all three passes
        draped = (
            (1.0 - highlight_mask) * darkened_rgb +
            highlight_mask          * highlight_rgb
        )
        draped = np.clip(draped, 0.0, 1.0)

        # Final lerp: flat base ↔ full relief
        draped = np.clip(
            base_rgb * (1.0 - drape_strength) + draped * drape_strength,
            0.0, 1.0
        )

        # -----------------------------------------------------------------
        # Outer boundary edge stroke
        # Morphological erosion (shrink mask by 1px), XOR with original
        # mask → gives a 1-pixel ring at the data perimeter.
        # Painted as a semi-transparent dark stroke for clean separation.
        # -----------------------------------------------------------------
        eroded = self._simple_blur(mask_arr.astype(float), iterations=1) > 0.98
        boundary_ring = valid_inside & (~eroded)

        # Dark charcoal stroke — visible but not harsh on light theme
        stroke_col = np.array([0.20, 0.18, 0.16])
        draped[boundary_ring] = (
            0.45 * draped[boundary_ring] + 0.55 * stroke_col
        )

        # Alpha: fully opaque inside data area
        alpha = np.zeros((h, w), dtype=float)
        alpha[valid_inside] = 1.0

        return draped, alpha

    # =========================================================
    # Legend
    # =========================================================

    def _save_legend_png(self, path_png, colors, ncls, axis_lbls, y_label, x_label):
        """
        [FIX] Standalone legend spacing refined so the Y variable label does not overlap
        with the Low/Med/High tick labels.
        [FIX] Y-axis arrow corrected from ← to ↑.
        [IMPROVE] Modern light-theme styling applied to standalone legend.
        """
        with matplotlib.rc_context(_POSTER_STYLE):
            fig = plt.figure(figsize=(3.9, 3.4), facecolor='white')
            ax = fig.add_axes([0, 0, 1, 1], facecolor='white')
            ax.set_axis_off()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            border = FancyBboxPatch(
                (0.01, 0.01), 0.98, 0.98,
                boxstyle='round,pad=0.01',
                linewidth=0.6,
                edgecolor='#d4d0cb',
                facecolor='white',
                transform=ax.transAxes,
                zorder=0
            )
            ax.add_patch(border)

            # Shift grid slightly to the right to create more space for the Y axis label.
            grid_ax = fig.add_axes([0.32, 0.22, 0.51, 0.58], facecolor='none')

            for ycls in range(ncls):
                row_display = ycls
                for xcls in range(ncls):
                    idx = ycls * ncls + xcls
                    grid_ax.add_patch(
                        Rectangle(
                            (xcls, row_display), 1, 1,
                            facecolor=colors[idx],
                            edgecolor='white',
                            linewidth=1.6
                        )
                    )

            grid_ax.set_xlim(0, ncls)
            grid_ax.set_ylim(0, ncls)
            grid_ax.set_xticks([i + 0.5 for i in range(ncls)])
            grid_ax.set_yticks([i + 0.5 for i in range(ncls)])
            grid_ax.set_xticklabels(axis_lbls, fontsize=8, color='#333333')
            grid_ax.set_yticklabels(axis_lbls, fontsize=8, color='#333333')
            grid_ax.tick_params(length=0, pad=3)
            grid_ax.set_aspect('equal')

            for spine in grid_ax.spines.values():
                spine.set_visible(False)

            ax.text(
                0.58, 0.10,
                f'{x_label}  →',
                ha='center', va='center',
                fontsize=9, color='#222222',
                fontstyle='italic'
            )

            # Separate arrow and label so the Y variable no longer overlaps tick labels.
            ax.text(
                0.050, 0.50,
                '↑',
                ha='center', va='center',
                fontsize=10.5, color='#222222'
            )
            ax.text(
                0.082, 0.50,
                y_label,
                ha='center', va='center',
                fontsize=8.6, color='#222222',
                fontstyle='italic', rotation=90
            )

            fig.savefig(
                path_png, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor()
            )
            plt.close(fig)

    def _draw_inset_legend(self, fig, colors, ncls, axis_lbls, y_label, x_label, layout):
        """
        [FIX] Y-axis labels now correctly ordered bottom-to-top (same as standalone legend fix).
        [FIX] Y-axis arrow corrected from ← to ↑.
        [IMPROVE] Legend card uses thin border for refined frosted-glass look.
        [IMPROVE] Tick and axis label colors use #2a2a2a for readability on light bg.
        """
        cfg = layout['legend']

        card_left = cfg['card_left']
        card_bottom = cfg['card_bottom']
        card_width = cfg['card_width']
        card_height = cfg['card_height']

        # Frosted card background
        card_ax = fig.add_axes(
            [card_left, card_bottom, card_width, card_height],
            facecolor='none'
        )
        card_ax.set_axis_off()

        # White card with thin border
        card = Rectangle(
            (0.0, 0.0), 1.0, 1.0,
            transform=card_ax.transAxes,
            facecolor='white',
            edgecolor='#ccc8c0',
            linewidth=0.5,
            alpha=cfg['card_alpha'],
            zorder=0
        )
        card_ax.add_patch(card)

        grid_left = card_left + cfg['grid_left_offset']
        grid_bottom = card_bottom + cfg['grid_bottom_offset']
        grid_size = cfg['grid_size']

        grid_ax = fig.add_axes(
            [grid_left, grid_bottom, grid_size, grid_size],
            facecolor='none'
        )

        # ycls 0 at bottom, ycls ncls-1 at top (correct mathematical orientation)
        for ycls in range(ncls):
            row_display = ycls

            for xcls in range(ncls):
                idx = ycls * ncls + xcls

                grid_ax.add_patch(
                    Rectangle(
                        (xcls, row_display), 1, 1,
                        facecolor=colors[idx],
                        edgecolor='white',
                        linewidth=0.75
                    )
                )

        grid_ax.set_xlim(0, ncls)
        grid_ax.set_ylim(0, ncls)
        grid_ax.set_xticks([])
        grid_ax.set_yticks([])
        grid_ax.set_aspect('equal')

        for spine in grid_ax.spines.values():
            spine.set_visible(False)

        # X-axis tick labels
        for i, lab in enumerate(axis_lbls):
            fig.text(
                grid_left + grid_size * ((i + 0.5) / ncls),
                grid_bottom - 0.014,
                lab,
                ha='center', va='top',
                fontsize=cfg['tick_font'],
                color='#2a2a2a'
            )

        # Y-axis tick labels — same order as axis_lbls (low at bottom, high at top)
        for i, lab in enumerate(axis_lbls):
            fig.text(
                grid_left - 0.008,
                grid_bottom + grid_size * ((i + 0.5) / ncls),
                lab,
                ha='right', va='center',
                fontsize=cfg['tick_font'],
                color='#2a2a2a'
            )

        variable_offset = cfg['label_offset']

        # X arrow → correct
        fig.text(
            grid_left + grid_size / 2.0,
            grid_bottom - variable_offset,
            f'{x_label}  →',
            ha='center', va='center',
            fontsize=cfg['axis_font'],
            color='#1a1a1a',
            fontstyle='italic'
        )

        # Separate arrow and label to avoid overlap with Y tick labels.
        fig.text(
            grid_left - variable_offset - 0.004,
            grid_bottom + grid_size / 2.0,
            '↑',
            ha='center', va='center',
            fontsize=max(8, cfg['axis_font'] - 0.5),
            color='#1a1a1a'
        )
        fig.text(
            grid_left - variable_offset + 0.018,
            grid_bottom + grid_size / 2.0,
            y_label,
            ha='center', va='center',
            fontsize=max(7.7, cfg['axis_font'] - 0.9),
            color='#1a1a1a',
            fontstyle='italic',
            rotation=90
        )

    # =========================================================
    # Poster typography helpers
    # =========================================================

    def _draw_title_block(self, fig, title, subtitle, layout):
        """
        Draw the bivariate poster header using the same visual language as the
        LISA analytics outputs: small EQUIMAP kicker at the top-left, large
        left-aligned title, and italic subtitle below it.
        """
        title_size = layout['title_size_base']
        subtitle_size = layout['subtitle_size_base']

        if len(title) > 42:
            title_size -= 4
        elif len(title) > 30:
            title_size -= 2

        if len(subtitle) > 60:
            subtitle_size -= 2

        kicker_x = 0.055
        title_x = 0.055
        kicker_y = min(0.977, layout['title_y'] + 0.032)
        title_y = layout['title_y']
        subtitle_y = layout['subtitle_y']

        fig.text(
            kicker_x, kicker_y,
            'EQUIMAP',
            ha='left', va='top',
            fontsize=12.5,
            fontweight='bold',
            color='#1f7a8c',
            family='sans-serif'
        )

        fig.text(
            title_x, title_y,
            title,
            ha='left', va='top',
            fontsize=max(20, title_size - 4),
            fontweight='bold',
            color='#111111',
            family='sans-serif'
        )

        if subtitle:
            fig.text(
                title_x, subtitle_y,
                subtitle,
                ha='left', va='top',
                fontsize=max(11.5, subtitle_size - 4),
                color='#6b645d',
                family='serif',
                fontstyle='italic'
            )

    def _draw_footer(self, fig, footer_text, layout):
        """
        Footer with thin separator rule above it.
        """
        sep_y = layout['footer_y'] + 0.022

        sep_line = matplotlib.lines.Line2D(
            [0.30, 0.70], [sep_y, sep_y],
            transform=fig.transFigure,
            color='#c8c2ba',
            linewidth=0.4,
            linestyle='-'
        )
        fig.add_artist(sep_line)

        fig.text(
            0.5, layout['footer_y'],
            footer_text,
            ha='center', va='center',
            fontsize=8.5,
            color='#6a6560',
            family='serif',
            fontstyle='italic'
        )

    # =========================================================
    # Poster 2D
    # =========================================================

    def _save_flat_2d_poster_map_png(
        self,
        path_png,
        features_info,
        colors,
        ncls,
        axis_lbls,
        label1,
        label2,
        title,
        subtitle,
        footer_text,
        shadow_alpha,
        show_boundaries
    ):
        map_xlim = self._extent_padded_from_features(features_info)
        layout = self._get_dynamic_layout(features_info)

        with matplotlib.rc_context(_POSTER_STYLE):
            fig = plt.figure(figsize=layout['figsize'], facecolor='#f7f5f2')

            ax = fig.add_axes(layout['map_ax'], facecolor='none')
            ax.set_axis_off()
            ax.set_aspect('equal')

            xmin, xmax, ymin, ymax = map_xlim
            # [IMPROVE] Softer shadow: smaller offset, lower alpha
            dx = (xmax - xmin) * 0.006
            dy = -(ymax - ymin) * 0.007

            # Drop shadow pass
            for item in features_info:
                geom = item['geom']

                if geom is None or geom.isEmpty():
                    continue

                try:
                    if geom.isMultipart():
                        for poly in geom.asMultiPolygon():
                            if poly:
                                outer = poly[0]
                                xs = [pt.x() + dx for pt in outer]
                                ys = [pt.y() + dy for pt in outer]
                                ax.fill(xs, ys, facecolor='#222222', edgecolor='none',
                                        alpha=shadow_alpha * 0.7, zorder=0)
                    else:
                        poly = geom.asPolygon()
                        if poly:
                            outer = poly[0]
                            xs = [pt.x() + dx for pt in outer]
                            ys = [pt.y() + dy for pt in outer]
                            ax.fill(xs, ys, facecolor='#222222', edgecolor='none',
                                    alpha=shadow_alpha * 0.7, zorder=0)
                except Exception:
                    pass

            # Main polygon pass
            for item in features_info:
                geom = item['geom']
                biv = item['biv']

                color = '#eeebe6' if biv is None else colors[biv]

                self._draw_polygon_geom(
                    ax, geom,
                    facecolor=color,
                    edgecolor='#e0dbd4',
                    lw=0.30,
                    alpha=1.0,
                    zorder=2
                )

            if show_boundaries:
                self._draw_boundaries(ax, features_info, color='#b8b2aa', lw=0.22, alpha=0.40)

            ax.set_xlim(map_xlim[0], map_xlim[1])
            ax.set_ylim(map_xlim[2], map_xlim[3])

            self._draw_title_block(fig, title, subtitle, layout)

            self._draw_inset_legend(
                fig, colors, ncls, axis_lbls, label1, label2, layout
            )

            self._draw_footer(fig, footer_text, layout)

            fig.savefig(
                path_png, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor()
            )
            plt.close(fig)

    def _draw_outer_boundary(self, ax, features_info, color='#2a2520', lw=0.9, alpha=0.55, zorder=10):
        """
        Draw only the outer perimeter (dissolve-like effect):
        plots all polygon outer rings with a slightly heavier stroke and
        higher zorder so they appear above the draped raster image.
        For relief mode this gives a clean cartographic boundary that
        separates the data area from the poster background.
        """
        for item in features_info:
            geom = item['geom']

            if geom is None or geom.isEmpty():
                continue

            try:
                if geom.isMultipart():
                    for poly in geom.asMultiPolygon():
                        if poly:
                            outer = poly[0]
                            xs = [pt.x() for pt in outer] + [poly[0][0].x()]
                            ys = [pt.y() for pt in outer] + [poly[0][0].y()]
                            ax.plot(xs, ys, color=color, linewidth=lw,
                                    alpha=alpha, zorder=zorder,
                                    solid_capstyle='round', solid_joinstyle='round')
                else:
                    poly = geom.asPolygon()

                    if poly:
                        outer = poly[0]
                        xs = [pt.x() for pt in outer] + [poly[0][0].x()]
                        ys = [pt.y() for pt in outer] + [poly[0][0].y()]
                        ax.plot(xs, ys, color=color, linewidth=lw,
                                alpha=alpha, zorder=zorder,
                                solid_capstyle='round', solid_joinstyle='round')
            except Exception:
                pass

    # =========================================================
    # Poster Relief Draped
    # =========================================================

    def _save_relief_draped_poster_map_png(
        self,
        path_png,
        features_info,
        colors,
        ncls,
        axis_lbls,
        label1,
        label2,
        title,
        subtitle,
        footer_text,
        dem_extent,
        class_arr,
        mask_arr,
        hillshade,
        drape_strength,
        shade_strength,
        shadow_alpha,
        show_boundaries
    ):
        draped_rgb, alpha_mask = self._build_draped_rgb(
            class_arr, mask_arr, colors, hillshade,
            drape_strength=drape_strength,
            shade_strength=shade_strength
        )

        shadow_alpha_arr = self._build_shadow_mask(mask_arr, alpha=shadow_alpha)

        rgba = np.dstack([draped_rgb, alpha_mask])

        # Shadow RGBA — dark warm tone for natural paper shadow
        shadow_rgba = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=float)
        shadow_rgba[:, :, 0] = 0.10
        shadow_rgba[:, :, 1] = 0.08
        shadow_rgba[:, :, 2] = 0.06
        shadow_rgba[:, :, 3] = shadow_alpha_arr

        map_xlim = self._extent_padded_from_features(features_info)
        layout   = self._get_dynamic_layout(features_info)

        with matplotlib.rc_context(_POSTER_STYLE):
            fig = plt.figure(figsize=layout['figsize'], facecolor='#f7f5f2')

            ax = fig.add_axes(layout['map_ax'], facecolor='#ece8e0')
            ax.set_axis_off()
            ax.set_aspect('equal')

            # Shadow — slightly offset south-east (classic cartographic convention)
            dem_w  = dem_extent[1] - dem_extent[0]
            dem_h  = dem_extent[3] - dem_extent[2]
            sdx    = dem_w * 0.005
            sdy    = -dem_h * 0.006

            ax.imshow(
                shadow_rgba,
                extent=(
                    dem_extent[0] + sdx, dem_extent[1] + sdx,
                    dem_extent[2] + sdy, dem_extent[3] + sdy
                ),
                origin='upper',
                interpolation='bilinear',
                zorder=0
            )

            # Main draped raster
            ax.imshow(
                rgba,
                extent=dem_extent,
                origin='upper',
                interpolation='bilinear',
                zorder=1
            )

            # Outer boundary polygon strokes — always drawn on relief mode
            # for clear perimeter definition
            self._draw_outer_boundary(
                ax, features_info,
                color='#1e1b18', lw=0.75, alpha=0.50, zorder=10
            )

            # Optional inner feature boundaries
            if show_boundaries:
                self._draw_boundaries(
                    ax, features_info,
                    color='rgba(255,255,255,0)',   # invisible placeholder
                    lw=0.0, alpha=0.0
                )
                # Draw semi-transparent white lines on top of the draped image
                self._draw_boundaries(
                    ax, features_info,
                    color='#ffffff', lw=0.28, alpha=0.28
                )

            ax.set_xlim(map_xlim[0], map_xlim[1])
            ax.set_ylim(map_xlim[2], map_xlim[3])

            self._draw_title_block(fig, title, subtitle, layout)

            self._draw_inset_legend(
                fig, colors, ncls, axis_lbls, label1, label2, layout
            )

            self._draw_footer(fig, footer_text, layout)

            fig.savefig(
                path_png, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor()
            )
            plt.close(fig)

    # =========================================================
    # Main process
    # =========================================================

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)

        if source is None:
            raise QgsProcessingException('Invalid input polygon layer.')

        field1 = self.parameterAsString(parameters, self.FIELD1, context)
        field2 = self.parameterAsString(parameters, self.FIELD2, context)

        label1 = self.parameterAsString(parameters, self.LABEL1, context).strip() or 'Variable Y'
        label2 = self.parameterAsString(parameters, self.LABEL2, context).strip() or 'Variable X'

        ncls = [3, 4][self.parameterAsEnum(parameters, self.NUM_CLASSES, context)]

        scheme_name = [
            'BlueGill', 'BlueGold', 'BlueOr', 'BlueYl', 'Brown2',
            'DkBlue2', 'DkCyan2', 'DkViolet2', 'GrPink2', 'PinkGrn',
            'PurpleGrn', 'PurpleOr'
        ][self.parameterAsEnum(parameters, self.COLOR_SCHEME, context)]

        map_render_mode = [
            'Flat 2D Poster',
            'Relief Draped Poster'
        ][self.parameterAsEnum(parameters, self.MAP_RENDER_MODE, context)]

        map_title = self.parameterAsString(parameters, self.MAP_TITLE, context).strip() or 'EquiMap: Spatial Inequality Intelligence'

        map_subtitle = self.parameterAsString(parameters, self.MAP_SUBTITLE, context).strip()

        if not map_subtitle:
            if map_render_mode == 'Relief Draped Poster':
                map_subtitle = '{} and {} over shaded relief'.format(label1, label2)
            else:
                map_subtitle = '{} and {} bivariate quantile map'.format(label1, label2)

        footer_text = (
            self.parameterAsString(parameters, self.FOOTER_TEXT, context).strip()
            or '@Firman Afrianto and Maya Safira'
        )

        dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)

        max_raster_dim     = self.parameterAsInt(parameters, self.MAX_RASTER_DIM, context)
        hillshade_azimuth  = self.parameterAsDouble(parameters, self.HILLSHADE_AZIMUTH, context)
        hillshade_altitude = self.parameterAsDouble(parameters, self.HILLSHADE_ALTITUDE, context)
        relief_z_factor    = self.parameterAsDouble(parameters, self.RELIEF_Z_FACTOR, context)
        drape_strength     = self.parameterAsDouble(parameters, self.DRAPE_STRENGTH, context)
        shade_strength     = self.parameterAsDouble(parameters, self.SHADE_STRENGTH, context)
        shadow_alpha       = self.parameterAsDouble(parameters, self.SHADOW_ALPHA, context)
        show_boundaries    = self.parameterAsBool(parameters, self.SHOW_BOUNDARIES, context)
        all_touched        = self.parameterAsBool(parameters, self.ALL_TOUCHED, context)

        run_inequality = self.parameterAsBool(parameters, self.RUN_INEQUALITY, context)
        do_spatial     = self.parameterAsBool(parameters, self.DO_SPATIAL, context)
        weight_mode    = ['queen', 'rook', 'knn'][self.parameterAsEnum(parameters, self.WEIGHT_MODE, context)]
        knn_k          = int(self.parameterAsInt(parameters, self.KNN_K, context))
        permutations   = int(self.parameterAsInt(parameters, self.PERMUTATIONS, context))
        alpha          = float(self.parameterAsDouble(parameters, self.ALPHA, context))
        analytics_folder = self.parameterAsString(parameters, self.OUTPUT_ANALYTICS_FOLDER, context).strip()
        if not analytics_folder:
            analytics_folder = os.path.join(
                QgsProcessingUtils.tempFolder(),
                'equimap_spatial_inequality_intelligence_outputs'
            )
        analytics_folder = os.path.abspath(analytics_folder)
        os.makedirs(analytics_folder, exist_ok=True)

        # All cartographic and analytical files are stored in one output folder.
        qml_path    = os.path.join(analytics_folder, 'equimap_bivariate_style.qml')
        legend_path = os.path.join(analytics_folder, 'equimap_bivariate_legend.png')
        map_path    = os.path.join(analytics_folder, 'equimap_bivariate_poster_map.png')

        if map_render_mode == 'Relief Draped Poster':
            if dem_layer is None:
                raise QgsProcessingException(
                    'DEM raster is required for Relief Draped Poster mode.'
                )

            if source.sourceCrs() != dem_layer.crs():
                raise QgsProcessingException(
                    'Input polygon layer and DEM must use the same CRS for Relief Draped Poster mode. '
                    'Please reproject one of them first.'
                )

        features = list(source.getFeatures())

        if len(features) == 0:
            raise QgsProcessingException('The input layer contains no features.')

        values1_all, values2_all = [], []
        vals1, vals2 = [], []

        for feat in features:
            v1 = self._safe_float(feat[field1])
            v2 = self._safe_float(feat[field2])
            values1_all.append(np.nan if v1 is None else v1)
            values2_all.append(np.nan if v2 is None else v2)

            if v1 is not None:
                vals1.append(v1)
            if v2 is not None:
                vals2.append(v2)

        if len(vals1) == 0 or len(vals2) == 0:
            raise QgsProcessingException('One or both selected fields contain no valid numeric values.')

        # The main output folder has already been created above. Analytics files,
        # QML, legend, and poster map are all written to this same directory.

        breaks1 = self._quantile_breaks(vals1, ncls)
        breaks2 = self._quantile_breaks(vals2, ncls)

        axis_lbls = self._axis_labels(ncls)
        colors = self.gen_colors(scheme_name, ncls)

        label_map = {}

        for ycls in range(ncls):
            for xcls in range(ncls):
                idx = ycls * ncls + xcls
                label_map[idx] = '{}-{}'.format(axis_lbls[ycls], axis_lbls[xcls])

        # =====================================================
        # Inequality, concentration, Moran, and LISA analytics
        # =====================================================
        n_features = len(features)
        desc1 = desc2 = {}
        gini1 = gini2 = theil1 = theil2 = hoover1 = hoover2 = np.nan
        pearson12 = spearman12 = np.nan
        metrics1 = metrics2 = None
        idx_sorted1 = idx_sorted2 = []
        vals_sorted1 = vals_sorted2 = []
        cp1 = cv1 = cp2 = cv2 = None
        ci_2_given_1 = ci_1_given_2 = np.nan
        cp_21 = cy_21 = cp_12 = cy_12 = None
        cum_21_x = [None] * n_features
        cum_21_y = [None] * n_features
        cum_12_x = [None] * n_features
        cum_12_y = [None] * n_features
        z1 = _ia_zscore(values1_all)
        z2 = _ia_zscore(values2_all)
        spatial1 = _ia_default_spatial_result(n_features)
        spatial2 = _ia_default_spatial_result(n_features)
        bilisa_yx = _ia_default_spatial_result(n_features)
        bilisa_xy = _ia_default_spatial_result(n_features)

        if run_inequality:
            desc1 = _ia_describe_nonnegative(values1_all)
            desc2 = _ia_describe_nonnegative(values2_all)

            gini1 = _ia_gini(values1_all)
            theil1 = _ia_theil_t(values1_all)
            hoover1 = _ia_hoover(values1_all)

            gini2 = _ia_gini(values2_all)
            theil2 = _ia_theil_t(values2_all)
            hoover2 = _ia_hoover(values2_all)

            pearson12 = _ia_pearson_corr(values1_all, values2_all)
            spearman12 = _ia_spearman_corr(values1_all, values2_all)

            metrics1, idx_sorted1, vals_sorted1, cp1, cv1 = _ia_build_feature_metrics(values1_all)
            metrics2, idx_sorted2, vals_sorted2, cp2, cv2 = _ia_build_feature_metrics(values2_all)

            if len(idx_sorted1) > 0:
                x_sorted_1 = [max(0.0, float(values1_all[i])) for i in idx_sorted1]
                y_sorted_2_by_1 = [max(0.0, _ia_safe_float(values2_all[i])) if np.isfinite(_ia_safe_float(values2_all[i])) else 0.0 for i in idx_sorted1]
                cp_21, cy_21, ci_2_given_1 = _ia_concentration_index(x_sorted_1, y_sorted_2_by_1)
                for j, original_i in enumerate(idx_sorted1):
                    cum_21_x[original_i] = float(cp_21[j + 1])
                    cum_21_y[original_i] = float(cy_21[j + 1])

            if len(idx_sorted2) > 0:
                x_sorted_2 = [max(0.0, float(values2_all[i])) for i in idx_sorted2]
                y_sorted_1_by_2 = [max(0.0, _ia_safe_float(values1_all[i])) if np.isfinite(_ia_safe_float(values1_all[i])) else 0.0 for i in idx_sorted2]
                cp_12, cy_12, ci_1_given_2 = _ia_concentration_index(x_sorted_2, y_sorted_1_by_2)
                for j, original_i in enumerate(idx_sorted2):
                    cum_12_x[original_i] = float(cp_12[j + 1])
                    cum_12_y[original_i] = float(cy_12[j + 1])

            if do_spatial:
                if not _HAS_PYSAL:
                    feedback.pushWarning('libpysal/esda is not available. Moran and LISA are skipped.')
                elif not _HAS_SHAPELY:
                    feedback.pushWarning('shapely is not available. Moran and LISA are skipped.')
                else:
                    spatial1 = _ia_run_global_local_moran_from_features(
                        features, values1_all,
                        weight_mode=weight_mode,
                        knn_k=knn_k,
                        permutations=permutations,
                        alpha=alpha
                    )
                    spatial2 = _ia_run_global_local_moran_from_features(
                        features, values2_all,
                        weight_mode=weight_mode,
                        knn_k=knn_k,
                        permutations=permutations,
                        alpha=alpha
                    )
                    bilisa_yx = _ia_run_bivariate_moran_from_features(
                        features, values1_all, values2_all,
                        weight_mode=weight_mode,
                        knn_k=knn_k,
                        permutations=permutations,
                        alpha=alpha
                    )
                    bilisa_xy = _ia_run_bivariate_moran_from_features(
                        features, values2_all, values1_all,
                        weight_mode=weight_mode,
                        knn_k=knn_k,
                        permutations=permutations,
                        alpha=alpha
                    )

            # -------------------------------------------------
            # Save analytics CSV and PNG
            # -------------------------------------------------
            p_summary = os.path.join(analytics_folder, 'equimap_summary_inequality_moran_lisa.csv')
            p_lorenz_1_png = os.path.join(analytics_folder, 'equimap_lorenz_field1_y_axis.png')
            p_lorenz_2_png = os.path.join(analytics_folder, 'equimap_lorenz_field2_x_axis.png')
            p_conc_21_png = os.path.join(analytics_folder, 'equimap_concentration_field2_given_field1.png')
            p_conc_12_png = os.path.join(analytics_folder, 'equimap_concentration_field1_given_field2.png')
            p_moran_1_png = os.path.join(analytics_folder, 'equimap_moran_scatter_field1_y_axis.png')
            p_moran_2_png = os.path.join(analytics_folder, 'equimap_moran_scatter_field2_x_axis.png')
            p_lisa_1_png = os.path.join(analytics_folder, 'equimap_lisa_map_field1_y_axis.png')
            p_lisa_2_png = os.path.join(analytics_folder, 'equimap_lisa_map_field2_x_axis.png')
            p_bilisa_yx_png = os.path.join(analytics_folder, 'equimap_bilisa_y_vs_lag_x.png')
            p_bilisa_xy_png = os.path.join(analytics_folder, 'equimap_bilisa_x_vs_lag_y.png')
            p_bimoran_yx_png = os.path.join(analytics_folder, 'equimap_bivariate_moran_y_vs_lag_x.png')
            p_bimoran_xy_png = os.path.join(analytics_folder, 'equimap_bivariate_moran_x_vs_lag_y.png')
            p_lorenz_1_csv = os.path.join(analytics_folder, 'equimap_lorenz_points_field1_y_axis.csv')
            p_lorenz_2_csv = os.path.join(analytics_folder, 'equimap_lorenz_points_field2_x_axis.csv')
            p_conc_21_csv = os.path.join(analytics_folder, 'equimap_concentration_points_field2_given_field1.csv')
            p_conc_12_csv = os.path.join(analytics_folder, 'equimap_concentration_points_field1_given_field2.csv')
            p_priority_csv = os.path.join(analytics_folder, 'equimap_priority_ranking.csv')
            p_dashboard_png = os.path.join(analytics_folder, 'equimap_summary_dashboard.png')
            p_report_html = os.path.join(analytics_folder, 'equimap_spatial_inequality_report.html')

            names_all = [str(feat.id()) for feat in features]

            priority = _ia_build_equimap_priority(
                values1_all, values2_all,
                [self._classify(v if np.isfinite(v) else None, breaks1) for v in values1_all],
                [self._classify(v if np.isfinite(v) else None, breaks2) for v in values2_all],
                metrics1, metrics2, spatial1, spatial2
            )
            try:
                _ia_write_priority_ranking_csv(p_priority_csv, names_all, label1, label2, values1_all, values2_all, priority)
            except Exception as e:
                feedback.pushWarning('Failed to create priority ranking CSV: {}'.format(e))

            try:
                if cp1 is not None:
                    _ia_save_curve_plot(
                        'Lorenz Curve: {}'.format(label1),
                        'Cumulative unit share',
                        'Cumulative {} share'.format(label1),
                        cp1, cv1, p_lorenz_1_png,
                        subtitle='Gini={:.4f} | Theil={:.4f} | Hoover={:.4f}'.format(gini1, theil1, hoover1)
                    )
                    _ia_write_curve_csv(p_lorenz_1_csv, [names_all[i] for i in idx_sorted1], vals_sorted1, cp1, cv1)

                if cp2 is not None:
                    _ia_save_curve_plot(
                        'Lorenz Curve: {}'.format(label2),
                        'Cumulative unit share',
                        'Cumulative {} share'.format(label2),
                        cp2, cv2, p_lorenz_2_png,
                        subtitle='Gini={:.4f} | Theil={:.4f} | Hoover={:.4f}'.format(gini2, theil2, hoover2)
                    )
                    _ia_write_curve_csv(p_lorenz_2_csv, [names_all[i] for i in idx_sorted2], vals_sorted2, cp2, cv2)

                if cp_21 is not None:
                    _ia_save_curve_plot(
                        'Concentration Curve: {} | {}'.format(label2, label1),
                        'Cumulative {} share'.format(label1),
                        'Cumulative {} share'.format(label2),
                        cp_21, cy_21, p_conc_21_png,
                        subtitle='CI={:.4f}'.format(ci_2_given_1)
                    )
                    _ia_write_concentration_csv(
                        p_conc_21_csv,
                        [names_all[i] for i in idx_sorted1],
                        [max(0.0, float(values1_all[i])) for i in idx_sorted1],
                        [max(0.0, _ia_safe_float(values2_all[i])) if np.isfinite(_ia_safe_float(values2_all[i])) else 0.0 for i in idx_sorted1],
                        cp_21, cy_21, label1, label2
                    )

                if cp_12 is not None:
                    _ia_save_curve_plot(
                        'Concentration Curve: {} | {}'.format(label1, label2),
                        'Cumulative {} share'.format(label2),
                        'Cumulative {} share'.format(label1),
                        cp_12, cy_12, p_conc_12_png,
                        subtitle='CI={:.4f}'.format(ci_1_given_2)
                    )
                    _ia_write_concentration_csv(
                        p_conc_12_csv,
                        [names_all[i] for i in idx_sorted2],
                        [max(0.0, float(values2_all[i])) for i in idx_sorted2],
                        [max(0.0, _ia_safe_float(values1_all[i])) if np.isfinite(_ia_safe_float(values1_all[i])) else 0.0 for i in idx_sorted2],
                        cp_12, cy_12, label2, label1
                    )

                if np.isfinite(spatial1['global_I']):
                    _ia_save_moran_scatter_png(spatial1['z'], spatial1['wz'], label1, spatial1['global_I'], spatial1['global_p'], p_moran_1_png)
                    _ia_save_lisa_map_png(features, spatial1['cluster_code'], 'LISA Map: {}'.format(label1), p_lisa_1_png)

                if np.isfinite(spatial2['global_I']):
                    _ia_save_moran_scatter_png(spatial2['z'], spatial2['wz'], label2, spatial2['global_I'], spatial2['global_p'], p_moran_2_png)
                    _ia_save_lisa_map_png(features, spatial2['cluster_code'], 'LISA Map: {}'.format(label2), p_lisa_2_png)

                # BiLISA / Bivariate Moran: directional cross-variable spatial association
                if np.isfinite(bilisa_yx['global_I']):
                    _ia_save_bivariate_moran_scatter_png(
                        bilisa_yx['z'], bilisa_yx['wz'], label1, label2,
                        bilisa_yx['global_I'], bilisa_yx['global_p'], p_bimoran_yx_png
                    )
                    _ia_save_lisa_map_png(
                        features, bilisa_yx['cluster_code'],
                        'BiLISA Map: {} vs lag({})'.format(label1, label2),
                        p_bilisa_yx_png
                    )

                if np.isfinite(bilisa_xy['global_I']):
                    _ia_save_bivariate_moran_scatter_png(
                        bilisa_xy['z'], bilisa_xy['wz'], label2, label1,
                        bilisa_xy['global_I'], bilisa_xy['global_p'], p_bimoran_xy_png
                    )
                    _ia_save_lisa_map_png(
                        features, bilisa_xy['cluster_code'],
                        'BiLISA Map: {} vs lag({})'.format(label2, label1),
                        p_bilisa_xy_png
                    )

                _ia_save_summary_dashboard_png(
                    p_dashboard_png, label1, label2,
                    gini1, gini2, theil1, theil2, hoover1, hoover2,
                    pearson12, spearman12, spatial1, spatial2, priority
                )
                report_metrics = {
                    'Gini {}'.format(label1): '{:.4f}'.format(gini1),
                    'Gini {}'.format(label2): '{:.4f}'.format(gini2),
                    'Theil {}'.format(label1): '{:.4f}'.format(theil1),
                    'Theil {}'.format(label2): '{:.4f}'.format(theil2),
                    'Hoover {}'.format(label1): '{:.4f}'.format(hoover1),
                    'Hoover {}'.format(label2): '{:.4f}'.format(hoover2),
                    'Pearson correlation': '{:.4f}'.format(pearson12) if np.isfinite(pearson12) else 'NA',
                    'Spearman correlation': '{:.4f}'.format(spearman12) if np.isfinite(spearman12) else 'NA',
                    "Moran's I {}".format(label1): '{:.4f}'.format(spatial1['global_I']) if np.isfinite(spatial1['global_I']) else 'NA',
                    "Moran's I {}".format(label2): '{:.4f}'.format(spatial2['global_I']) if np.isfinite(spatial2['global_I']) else 'NA',
                    "Bivariate Moran I {} vs lag({})".format(label1, label2): '{:.4f}'.format(bilisa_yx['global_I']) if np.isfinite(bilisa_yx['global_I']) else 'NA',
                    "Bivariate Moran I {} vs lag({})".format(label2, label1): '{:.4f}'.format(bilisa_xy['global_I']) if np.isfinite(bilisa_xy['global_I']) else 'NA',
                    'Very High priority features': sum(1 for v in priority['level'] if v == 'Very High'),
                    'High priority features': sum(1 for v in priority['level'] if v == 'High')
                }
                report_images = [
                    (p_dashboard_png, 'EquiMap summary dashboard'),
                    (p_lorenz_1_png, 'Lorenz curve {}'.format(label1)),
                    (p_lorenz_2_png, 'Lorenz curve {}'.format(label2)),
                    (p_conc_21_png, 'Concentration curve {} given {}'.format(label2, label1)),
                    (p_conc_12_png, 'Concentration curve {} given {}'.format(label1, label2)),
                    (p_moran_1_png, 'Moran scatter {}'.format(label1)),
                    (p_moran_2_png, 'Moran scatter {}'.format(label2)),
                    (p_lisa_1_png, 'LISA map {}'.format(label1)),
                    (p_lisa_2_png, 'LISA map {}'.format(label2)),
                    (p_bimoran_yx_png, 'Bivariate Moran scatter {} vs lag({})'.format(label1, label2)),
                    (p_bimoran_xy_png, 'Bivariate Moran scatter {} vs lag({})'.format(label2, label1)),
                    (p_bilisa_yx_png, 'BiLISA map {} vs lag({})'.format(label1, label2)),
                    (p_bilisa_xy_png, 'BiLISA map {} vs lag({})'.format(label2, label1)),
                ]
                _ia_write_html_report(
                    p_report_html, 'EquiMap: Spatial Inequality Intelligence',
                    label1, label2, report_metrics, report_images, os.path.basename(p_priority_csv)
                )
            except Exception as e:
                feedback.pushWarning('Some analytical PNG/CSV outputs could not be created: {}'.format(e))

            with open(p_summary, 'w', newline='', encoding='utf-8') as fh:
                w = csv.writer(fh)
                w.writerow(['section', 'metric', 'variable', 'value'])

                for nm, desc in [(label1, desc1), (label2, desc2)]:
                    w.writerow(['descriptive', 'n_valid', nm, desc.get('n_valid', '')])
                    w.writerow(['descriptive', 'total', nm, desc.get('total', '')])
                    w.writerow(['descriptive', 'mean', nm, desc.get('mean', '')])
                    w.writerow(['descriptive', 'median', nm, desc.get('median', '')])
                    w.writerow(['descriptive', 'std', nm, desc.get('std', '')])
                    w.writerow(['descriptive', 'min', nm, desc.get('min', '')])
                    w.writerow(['descriptive', 'max', nm, desc.get('max', '')])
                    w.writerow(['descriptive', 'cv', nm, desc.get('cv', '')])

                w.writerow(['inequality', 'gini', label1, gini1])
                w.writerow(['inequality', 'theil_t', label1, theil1])
                w.writerow(['inequality', 'hoover', label1, hoover1])
                w.writerow(['inequality', 'gini', label2, gini2])
                w.writerow(['inequality', 'theil_t', label2, theil2])
                w.writerow(['inequality', 'hoover', label2, hoover2])
                w.writerow(['concentration', 'CI_field2_given_field1', '{}|{}'.format(label2, label1), ci_2_given_1])
                w.writerow(['concentration', 'CI_field1_given_field2', '{}|{}'.format(label1, label2), ci_1_given_2])
                w.writerow(['correlation', 'pearson', '{} vs {}'.format(label1, label2), pearson12])
                w.writerow(['correlation', 'spearman', '{} vs {}'.format(label1, label2), spearman12])
                w.writerow(['moran', 'global_I', label1, spatial1['global_I']])
                w.writerow(['moran', 'global_p', label1, spatial1['global_p']])
                w.writerow(['moran', 'global_I', label2, spatial2['global_I']])
                w.writerow(['moran', 'global_p', label2, spatial2['global_p']])
                w.writerow(['bivariate_moran', 'global_I', '{} vs lag({})'.format(label1, label2), bilisa_yx['global_I']])
                w.writerow(['bivariate_moran', 'global_p', '{} vs lag({})'.format(label1, label2), bilisa_yx['global_p']])
                w.writerow(['bivariate_moran', 'global_I', '{} vs lag({})'.format(label2, label1), bilisa_xy['global_I']])
                w.writerow(['bivariate_moran', 'global_p', '{} vs lag({})'.format(label2, label1), bilisa_xy['global_p']])

                for code, lab in [(0, 'NS'), (1, 'HH'), (2, 'LL'), (3, 'LH'), (4, 'HL')]:
                    w.writerow(['lisa_count', lab, label1, sum([1 for v in spatial1['cluster_code'] if v == code])])
                    w.writerow(['lisa_count', lab, label2, sum([1 for v in spatial2['cluster_code'] if v == code])])
                    w.writerow(['bilisa_count', lab, '{} vs lag({})'.format(label1, label2), sum([1 for v in bilisa_yx['cluster_code'] if v == code])])
                    w.writerow(['bilisa_count', lab, '{} vs lag({})'.format(label2, label1), sum([1 for v in bilisa_xy['cluster_code'] if v == code])])

                for lev in ['Very High', 'High', 'Moderate', 'Low', 'No Data']:
                    w.writerow(['priority_count', lev, 'EquiMap priority', sum([1 for v in priority['level'] if v == lev])])
                typ_counts = {}
                for t in priority['typology']:
                    typ_counts[t] = typ_counts.get(t, 0) + 1
                for t, c in sorted(typ_counts.items(), key=lambda x: x[1], reverse=True):
                    w.writerow(['typology_count', t, 'EquiMap typology', c])
                w.writerow(['output', 'priority_ranking_csv', 'EquiMap', p_priority_csv])
                w.writerow(['output', 'summary_dashboard_png', 'EquiMap', p_dashboard_png])
                w.writerow(['output', 'html_report', 'EquiMap', p_report_html])

        else:
            metrics1, _, _, _, _ = _ia_build_feature_metrics(values1_all)
            metrics2, _, _, _, _ = _ia_build_feature_metrics(values2_all)

        # =====================================================
        # Output layer fields
        # =====================================================
        fields = QgsFields()

        for fld in source.fields():
            fields.append(fld)

        fields.append(QgsField('biv_y_cls', QVariant.Int))
        fields.append(QgsField('biv_x_cls', QVariant.Int))
        fields.append(QgsField('biv_class', QVariant.Int))
        fields.append(QgsField('biv_label', QVariant.String))

        fields.append(QgsField('Y_VAL', QVariant.Double))
        fields.append(QgsField('Y_RANK', QVariant.Int))
        fields.append(QgsField('Y_SHARE', QVariant.Double))
        fields.append(QgsField('Y_CUMUNIT', QVariant.Double))
        fields.append(QgsField('Y_CUMVAL', QVariant.Double))
        fields.append(QgsField('Y_DEVEQ', QVariant.Double))
        fields.append(QgsField('Y_THEILTRM', QVariant.Double))

        fields.append(QgsField('X_VAL', QVariant.Double))
        fields.append(QgsField('X_RANK', QVariant.Int))
        fields.append(QgsField('X_SHARE', QVariant.Double))
        fields.append(QgsField('X_CUMUNIT', QVariant.Double))
        fields.append(QgsField('X_CUMVAL', QVariant.Double))
        fields.append(QgsField('X_DEVEQ', QVariant.Double))
        fields.append(QgsField('X_THEILTRM', QVariant.Double))

        fields.append(QgsField('X_Y_CUMX', QVariant.Double))
        fields.append(QgsField('X_Y_CUMY', QVariant.Double))
        fields.append(QgsField('Y_X_CUMX', QVariant.Double))
        fields.append(QgsField('Y_X_CUMY', QVariant.Double))

        fields.append(QgsField('Y_Z', QVariant.Double))
        fields.append(QgsField('X_Z', QVariant.Double))

        fields.append(QgsField('Y_WZ', QVariant.Double))
        fields.append(QgsField('Y_MORAN_I', QVariant.Double))
        fields.append(QgsField('Y_MORAN_P', QVariant.Double))
        fields.append(QgsField('Y_LISA_I', QVariant.Double))
        fields.append(QgsField('Y_LISA_P', QVariant.Double))
        fields.append(QgsField('Y_LISA_C', QVariant.Int))
        fields.append(QgsField('Y_LISA_L', QVariant.String))

        fields.append(QgsField('X_WZ', QVariant.Double))
        fields.append(QgsField('X_MORAN_I', QVariant.Double))
        fields.append(QgsField('X_MORAN_P', QVariant.Double))
        fields.append(QgsField('X_LISA_I', QVariant.Double))
        fields.append(QgsField('X_LISA_P', QVariant.Double))
        fields.append(QgsField('X_LISA_C', QVariant.Int))
        fields.append(QgsField('X_LISA_L', QVariant.String))

        fields.append(QgsField('BYX_WZ', QVariant.Double))
        fields.append(QgsField('BYX_I', QVariant.Double))
        fields.append(QgsField('BYX_P', QVariant.Double))
        fields.append(QgsField('BYX_LI', QVariant.Double))
        fields.append(QgsField('BYX_LP', QVariant.Double))
        fields.append(QgsField('BYX_C', QVariant.Int))
        fields.append(QgsField('BYX_L', QVariant.String))

        fields.append(QgsField('BXY_WZ', QVariant.Double))
        fields.append(QgsField('BXY_I', QVariant.Double))
        fields.append(QgsField('BXY_P', QVariant.Double))
        fields.append(QgsField('BXY_LI', QVariant.Double))
        fields.append(QgsField('BXY_LP', QVariant.Double))
        fields.append(QgsField('BXY_C', QVariant.Int))
        fields.append(QgsField('BXY_L', QVariant.String))

        fields.append(QgsField('EQ_PRIO', QVariant.Double))
        fields.append(QgsField('EQ_LEVEL', QVariant.String))
        fields.append(QgsField('EQ_TYPE', QVariant.String))
        fields.append(QgsField('EQ_MISMATCH', QVariant.Double))
        fields.append(QgsField('EQ_RECOMM', QVariant.String))
        fields.append(QgsField('ANALYT_DIR', QVariant.String))

        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            fields, source.wkbType(), source.sourceCrs()
        )

        if sink is None:
            raise QgsProcessingException('Failed to create the output sink.')

        features_info = []
        nodata_count = 0
        total = len(features)

        for i, feat in enumerate(features):
            if feedback.isCanceled():
                break

            new_feat = QgsFeature(fields)
            new_feat.setGeometry(feat.geometry())

            v1 = self._safe_float(feat[field1])
            v2 = self._safe_float(feat[field2])

            if v1 is None or v2 is None:
                ycls = None
                xcls = None
                biv = None
                label = 'No Data'
                nodata_count += 1
            else:
                ycls = self._classify(v1, breaks1)
                xcls = self._classify(v2, breaks2)
                biv = ycls * ncls + xcls
                label = label_map[biv]

            attrs = feat.attributes() + [ycls, xcls, biv, label]

            attrs += [
                metrics1['VAL'][i], metrics1['RANK'][i], metrics1['SHARE'][i],
                metrics1['CUMUNIT'][i], metrics1['CUMVAL'][i], metrics1['DEVEQ'][i],
                metrics1['THEILTERM'][i]
            ]
            attrs += [
                metrics2['VAL'][i], metrics2['RANK'][i], metrics2['SHARE'][i],
                metrics2['CUMUNIT'][i], metrics2['CUMVAL'][i], metrics2['DEVEQ'][i],
                metrics2['THEILTERM'][i]
            ]
            attrs += [cum_21_x[i], cum_21_y[i], cum_12_x[i], cum_12_y[i]]
            attrs += [z1[i], z2[i]]

            attrs += [
                spatial1['wz'][i], spatial1['global_I'], spatial1['global_p'],
                spatial1['local_I'][i], spatial1['local_p'][i],
                spatial1['cluster_code'][i], spatial1['cluster_label'][i]
            ]
            attrs += [
                spatial2['wz'][i], spatial2['global_I'], spatial2['global_p'],
                spatial2['local_I'][i], spatial2['local_p'][i],
                spatial2['cluster_code'][i], spatial2['cluster_label'][i]
            ]

            attrs += [
                bilisa_yx['wz'][i], bilisa_yx['global_I'], bilisa_yx['global_p'],
                bilisa_yx['local_I'][i], bilisa_yx['local_p'][i],
                bilisa_yx['cluster_code'][i], bilisa_yx['cluster_label'][i]
            ]
            attrs += [
                bilisa_xy['wz'][i], bilisa_xy['global_I'], bilisa_xy['global_p'],
                bilisa_xy['local_I'][i], bilisa_xy['local_p'][i],
                bilisa_xy['cluster_code'][i], bilisa_xy['cluster_label'][i]
            ]

            attrs += [
                priority['score'][i],
                priority['level'][i],
                priority['typology'][i],
                priority['mismatch'][i],
                priority['recommendation'][i]
            ]
            attrs += [analytics_folder if run_inequality else '']

            new_feat.setAttributes(attrs)
            sink.addFeature(new_feat, QgsFeatureSink.FastInsert)

            features_info.append({'geom': feat.geometry(), 'biv': biv})

            feedback.setProgress(int(((i + 1) / total) * 100))

        layout_used = self._get_dynamic_layout(features_info)

        self._apply_and_save_qml(dest_id, context, colors, label_map, qml_path, feedback)

        self._save_legend_png(legend_path, colors, ncls, axis_lbls, label1, label2)

        if map_render_mode == 'Flat 2D Poster':
            self._save_flat_2d_poster_map_png(
                map_path, features_info, colors, ncls, axis_lbls,
                label1, label2, map_title, map_subtitle, footer_text,
                shadow_alpha, show_boundaries
            )
        else:
            dem_arr, dem_transform, dem_extent = self._read_dem(dem_layer, max_raster_dim, feedback)

            hillshade = self._make_hillshade(
                dem_arr,
                azimuth=hillshade_azimuth,
                altitude=hillshade_altitude,
                z_factor=relief_z_factor
            )

            if hillshade is None:
                raise QgsProcessingException('Failed to build hillshade from DEM.')

            class_arr, mask_arr = self._rasterize_classes(
                features_info, dem_arr.shape, dem_transform, all_touched=all_touched
            )

            self._save_relief_draped_poster_map_png(
                map_path, features_info, colors, ncls, axis_lbls,
                label1, label2, map_title, map_subtitle, footer_text,
                dem_extent, class_arr, mask_arr, hillshade,
                drape_strength, shade_strength, shadow_alpha, show_boundaries
            )

        feedback.pushInfo('=' * 54)
        feedback.pushInfo('EquiMap: Spatial Inequality Intelligence v1.3.1')
        feedback.pushInfo('=' * 54)
        feedback.pushInfo('Map render mode    = {}'.format(map_render_mode))
        feedback.pushInfo('Dynamic layout     = {}'.format(layout_used.get('layout_name')))
        feedback.pushInfo('Figure size        = {}'.format(layout_used.get('figsize')))
        feedback.pushInfo('Map axis           = {}'.format(layout_used.get('map_ax')))
        feedback.pushInfo('Y-axis field       = {}'.format(field1))
        feedback.pushInfo('X-axis field       = {}'.format(field2))
        feedback.pushInfo('Y-axis label       = {}'.format(label1))
        feedback.pushInfo('X-axis label       = {}'.format(label2))
        feedback.pushInfo('Y-axis breaks      = {}'.format(self._format_breaks(breaks1)))
        feedback.pushInfo('X-axis breaks      = {}'.format(self._format_breaks(breaks2)))
        feedback.pushInfo('Classes            = {} × {}'.format(ncls, ncls))
        feedback.pushInfo('Palette            = {}'.format(scheme_name))
        feedback.pushInfo('Map title          = {}'.format(map_title))
        feedback.pushInfo('Map subtitle       = {}'.format(map_subtitle))
        feedback.pushInfo('Max raster dim     = {}'.format(max_raster_dim))
        feedback.pushInfo('Hillshade azimuth  = {}'.format(hillshade_azimuth))
        feedback.pushInfo('Hillshade altitude = {}'.format(hillshade_altitude))
        feedback.pushInfo('Relief z-factor    = {}'.format(relief_z_factor))
        feedback.pushInfo('Drape strength     = {}'.format(drape_strength))
        feedback.pushInfo('Shade strength     = {}'.format(shade_strength))
        feedback.pushInfo('Shadow alpha       = {}'.format(shadow_alpha))
        feedback.pushInfo('Show boundaries    = {}'.format(show_boundaries))
        feedback.pushInfo('All touched        = {}'.format(all_touched))
        feedback.pushInfo('Run inequality     = {}'.format(run_inequality))
        feedback.pushInfo('Run Moran/LISA     = {}'.format(do_spatial))
        feedback.pushInfo('Weight mode        = {}'.format(weight_mode))
        feedback.pushInfo('Permutations       = {}'.format(permutations))
        feedback.pushInfo('Alpha              = {}'.format(alpha))
        if run_inequality:
            feedback.pushInfo('Gini Y             = {:.6f}'.format(gini1))
            feedback.pushInfo('Gini X             = {:.6f}'.format(gini2))
            feedback.pushInfo("Moran's I Y        = {}".format('{:.6f}'.format(spatial1['global_I']) if np.isfinite(spatial1['global_I']) else 'not available'))
            feedback.pushInfo("Moran's I X        = {}".format('{:.6f}'.format(spatial2['global_I']) if np.isfinite(spatial2['global_I']) else 'not available'))
            feedback.pushInfo("BiMoran Y~lagX     = {}".format('{:.6f}'.format(bilisa_yx['global_I']) if np.isfinite(bilisa_yx['global_I']) else 'not available'))
            feedback.pushInfo("BiMoran X~lagY     = {}".format('{:.6f}'.format(bilisa_xy['global_I']) if np.isfinite(bilisa_xy['global_I']) else 'not available'))
            feedback.pushInfo('Analytics folder   = {}'.format(analytics_folder))
        feedback.pushInfo('Features total     = {}'.format(len(features)))
        feedback.pushInfo('No Data features   = {}'.format(nodata_count))
        feedback.pushInfo('=' * 54)

        self._push_file_link(feedback, 'QML style',    qml_path)
        self._push_file_link(feedback, 'Legend PNG',   legend_path)
        self._push_file_link(feedback, 'Poster map PNG', map_path)
        if run_inequality:
            self._push_file_link(feedback, 'Analytics folder', analytics_folder)
            self._push_file_link(feedback, 'Priority ranking CSV', os.path.join(analytics_folder, 'equimap_priority_ranking.csv'))
            self._push_file_link(feedback, 'Summary dashboard PNG', os.path.join(analytics_folder, 'equimap_summary_dashboard.png'))
            self._push_file_link(feedback, 'BiLISA Y vs lag X PNG', os.path.join(analytics_folder, 'equimap_bilisa_y_vs_lag_x.png'))
            self._push_file_link(feedback, 'BiLISA X vs lag Y PNG', os.path.join(analytics_folder, 'equimap_bilisa_x_vs_lag_y.png'))
            self._push_file_link(feedback, 'Bivariate Moran Y vs lag X PNG', os.path.join(analytics_folder, 'equimap_bivariate_moran_y_vs_lag_x.png'))
            self._push_file_link(feedback, 'Bivariate Moran X vs lag Y PNG', os.path.join(analytics_folder, 'equimap_bivariate_moran_x_vs_lag_y.png'))
            self._push_file_link(feedback, 'HTML report', os.path.join(analytics_folder, 'equimap_spatial_inequality_report.html'))

        return {
            self.OUTPUT: dest_id,
            self.QML:    qml_path,
            self.LEGEND: legend_path,
            self.MAP:    map_path,
            self.OUTPUT_ANALYTICS_FOLDER: analytics_folder
        }
