from pathlib import Path
from collections import OrderedDict, Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.colors import (to_rgba, Normalize, ListedColormap, LinearSegmentedColormap, is_color_like) 
import gc
from typing import Optional, List, Tuple, Dict, Sequence, Union

from sklearn.cluster import KMeans
from rastermap import Rastermap

# IBL / atlas
from iblatlas.regions import BrainRegions

# --------------------------------------------------
# Globals required by regional_group
# --------------------------------------------------
br = BrainRegions()

# Cache path (repo-local)
REPO_DIR = Path(__file__).resolve().parent
pth_dmn = REPO_DIR
pth_dmn.mkdir(parents=True, exist_ok=True)


T_BIN = 0.0125  # bin size [sec] for neural binning
sts = 0.002  # stride size in [sec] for overlapping bins

# conversion divident to get bins in seconds 
# (taking striding into account)
c_sec =  1.0 / (T_BIN / int(T_BIN // sts))


# --------------------------------------------------
# Allen color palette
# --------------------------------------------------
alleninfo_path = REPO_DIR / "alleninfo.npy"
if not alleninfo_path.exists():
    # backward-compatible fallback
    alleninfo_path = pth_dmn / "alleninfo.npy"
pal = np.load(alleninfo_path, allow_pickle=True).flat[0]

# --------------------------------------------------
# PETH label dictionary (needed by regional_group)
# --------------------------------------------------
tts__ = [
    'inter_trial',
    'blockL',
    'blockR',
    'quiescence',
    'block_change_s',
    'stimLbLcL',
    'stimLbRcL',
    'stimRbRcR',
    'stimRbLcR',
    'mistake_s',
    'motor_init',
    'block_change_m',
    'sLbLchoiceL',
    'sLbRchoiceL',
    'sRbRchoiceR',
    'sRbLchoiceR',
    'mistake_m',
    'choiceL',
    'choiceR',
    'fback1',
    'fback0',
]

# LaTeX typeset labels (Matplotlib mathtext)
peth_ila = [
    r"$\mathrm{rest}$",
    r"$\mathrm{L_b}$",
    r"$\mathrm{R_b}$",
    r"$\mathrm{quies}$",
    r"$\mathrm{change_b, s}$",
    r"$\mathrm{L_sL_cL_b, s}$",
    r"$\mathrm{L_sL_cR_b, s}$",
    r"$\mathrm{R_sR_cR_b, s}$",
    r"$\mathrm{R_sR_cL_b, s}$",
    r"$\mathrm{mistake, s}$",
    r"$\mathrm{m}$",
    r"$\mathrm{change_b, m}$",
    r"$\mathrm{L_sL_cL_b, m}$",
    r"$\mathrm{L_sL_cR_b, m}$",
    r"$\mathrm{R_sR_cR_b, m}$",
    r"$\mathrm{R_sR_cL_b, m}$",
    r"$\mathrm{mistake, m}$",
    r"$\mathrm{L_{move}}$",
    r"$\mathrm{R_{move}}$",
    r"$\mathrm{feedbk1}$",
    r"$\mathrm{feedbk0}$",
]

peth_dictm = dict(zip(tts__, peth_ila))


def regional_group(
    mapping,
    vers: str = "concat",
    ephys: bool = False,
    grid_upsample: int = 0,
    nclus: int = 25,
    nclus_rm: int = 100,
    nclus_s: int = 20,          
    cv: bool = True,
    locality: float = 0.75,
    time_lag_window: int = 5,
    symmetric: bool = False,
    rerun: bool = False,
    synthetic: bool = False,
    syn_control: bool = False,
    zsc: bool = True,         
):
    """

    Conventions
    -----------
    - zsc=True  -> base feature is 'concat_z'
    - zsc=False -> base feature is 'concat' (requires cv=False)


    """
    pth_res = Path(pth_dmn, "res")
    pth_res.mkdir(parents=True, exist_ok=True)

    if synthetic:
        cv = False

    nclus = int(nclus)
    nclus_rm = int(nclus_rm)
    nclus_s = int(nclus_s)

    # ---------------- cache paths ----------------
    def _cache_path(kind: str) -> Path:
        synth_tag = f"_synthetic1_nsb{nclus_s}_synctrl{int(bool(syn_control))}" if synthetic else ""
        ztag = f"_zsc{int(bool(zsc))}"  # NEW: used for ALL caches that depend on feature-space

        if kind == "stack":
            base = f"{vers}"
            base += f"_cv{cv}"
            base += f"_ephys{ephys}"
            return pth_res / (base + ".npy")

        if kind == "rm":
            base = f"{kind}_{vers}"
            base += f"_cv{cv}"
            base += f"_ephys{ephys}"
            base += f"_nclusrm{nclus_rm}"
            base += ztag
            base += synth_tag
            return pth_res / (base + ".npy")

        if kind == "kmeans":
            base = f"{kind}_{vers}"
            base += f"_cv{cv}"
            base += f"_ephys{ephys}"
            base += f"_n{nclus}"
            base += f"_nclusrm{nclus_rm}"
            base += ztag
            base += synth_tag
            return pth_res / (base + ".npy")

        if kind == "synthetic":
            # NEW: include ztag so zsc=False synthetic cache doesn't collide with zsc=True synthetic cache
            base = f"{kind}_{vers}"
            base += f"_cv{cv}"
            base += f"_ephys{ephys}"
            base += ztag
            base += f"_nsb{nclus_s}"
            base += f"_synctrl{int(bool(syn_control))}"
            base += "_margHist"
            base += "_seed0"
            return pth_res / (base + ".npy")

        raise ValueError(f"Unknown cache kind: {kind}")

    # ---------------- load stack ----------------
    stack_path = _cache_path("stack")
    if not stack_path.is_file():
        raise FileNotFoundError(
            f"Stack file not found: {stack_path}\n"
            "Expected stack caches to depend only on vers/cv/ephys (not rm hyperparams)."
        )

    r = np.load(stack_path, allow_pickle=True).flat[0]
    print(
        f"mapping {mapping}, vers {vers}, ephys {ephys}, "
        f"nclus {nclus}, nclus_rm {nclus_rm}, nclus_s {nclus_s}, "
        f"rerun {rerun}, cv {cv}, synthetic {synthetic}, syn_control {syn_control}, zsc {zsc}, "
        f"{len(r['ids'])} neurons loaded."
    )

    r["len"] = OrderedDict((k, int(r["len"][k])) for k in r["ttypes"])

    if "xyz" not in r:
        raise KeyError("Saved stack lacks 'xyz'.")
    r["nums"] = np.arange(r["xyz"].shape[0], dtype=int)

    # base feature key (real data feature space)
    base_feat = "concat_z" if zsc else "concat"
    if base_feat not in r:
        raise KeyError(f"Saved stack lacks '{base_feat}' (zsc={zsc}).")

    r["_order_signature"] = (
        "|".join(f"{k}:{r['len'][k]}" for k in r["ttypes"])
        + f"|shape:{np.asarray(r[base_feat]).shape}"
        + f"|zsc:{int(bool(zsc))}"
    )

    r["peth_dict"] = {x: peth_dictm[x] for x in r["ttypes"]}

    # ---------------- helpers (RM) ----------------
    def _load_rm_cache(rm_cache_path: Path, n_rows: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        if rerun or (not rm_cache_path.is_file()):
            return None, None
        try:
            cached = np.load(rm_cache_path, allow_pickle=True).flat[0]
            if not (
                isinstance(cached, dict)
                and cached.get("order_sig") == r["_order_signature"]
                and cached.get("nclus_rm") == nclus_rm
                and cached.get("synthetic") == bool(synthetic)
                and cached.get("zsc") == bool(zsc)
                and (cached.get("nclus_s") == nclus_s if synthetic else True)
                and (cached.get("syn_control") == bool(syn_control) if synthetic else True)
                and "rm_labels" in cached
                and "isort" in cached
            ):
                return None, None

            labels = np.asarray(cached["rm_labels"], dtype=int).reshape(-1)
            isort = np.asarray(cached["isort"], dtype=int).reshape(-1)
            if labels.shape[0] != n_rows or isort.shape[0] != n_rows:
                return None, None
            print(f"[rm] using cached labels/isort ({rm_cache_path.name})")
            return labels, isort
        except Exception as e:
            print(f"[rm] cache read error; will recompute Rastermap: {e}")
            return None, None

    def _compute_and_save_rm(rm_cache_path: Path, feat_used: str, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
        if feat_used not in r:
            raise KeyError(f"Feature '{feat_used}' not found in stack.")

        print(f"[rm] computing Rastermap (n_clusters={nclus_rm}) on {feat_used}")
        model = Rastermap(
            n_PCs=200,
            n_clusters=nclus_rm,
            grid_upsample=grid_upsample,
            locality=locality,
            time_lag_window=time_lag_window,
            bin_size=1,
            symmetric=symmetric,
        ).fit(r[feat_used])

        labels = np.asarray(model.embedding_clust, dtype=int)
        if labels.ndim > 1:
            labels = labels[:, 0]
        isort = np.asarray(model.isort, dtype=int).reshape(-1)

        if labels.shape[0] != n_rows or isort.shape[0] != n_rows:
            raise ValueError("Rastermap outputs do not match data length.")

        np.save(
            rm_cache_path,
            {
                "rm_labels": labels,
                "isort": isort,
                "order_sig": r["_order_signature"],
                "nclus_rm": nclus_rm,
                "synthetic": bool(synthetic),
                "zsc": bool(zsc),
                "nclus_s": int(nclus_s) if synthetic else None,
                "syn_control": bool(syn_control) if synthetic else None,
            },
            allow_pickle=True,
        )
        print(f"[rm] wrote cache ({rm_cache_path.name})")
        return labels, isort

    # ---------------- synthetic generation (UPDATED for zsc) ----------------
    def _compute_and_cache_synthetic(synth_feat_key: str, src_feat: str):
        synth_path = _cache_path("synthetic")

        if (not rerun) and synth_path.is_file():
            try:
                cached = np.load(synth_path, allow_pickle=True).flat[0]
                want_control = bool(syn_control)

                # NEW: require zsc + synth_feat_key + src_feat match
                if want_control:
                    ok = (
                        isinstance(cached, dict)
                        and cached.get("order_sig") == r["_order_signature"]
                        and cached.get("synthetic") is True
                        and cached.get("syn_control") is True
                        and cached.get("zsc") == bool(zsc)
                        and cached.get("synth_feat_key") == synth_feat_key
                        and cached.get("src_feat") == src_feat
                        and cached.get("nclus_basis") == nclus_s
                        and synth_feat_key in cached
                        and "V" in cached
                        and "C" in cached
                    )
                else:
                    ok = (
                        isinstance(cached, dict)
                        and cached.get("order_sig") == r["_order_signature"]
                        and cached.get("synthetic") is True
                        and cached.get("syn_control") is False
                        and cached.get("zsc") == bool(zsc)
                        and cached.get("synth_feat_key") == synth_feat_key
                        and cached.get("src_feat") == src_feat
                        and cached.get("nclus_basis") == nclus_s
                        and synth_feat_key in cached
                        and "V" in cached
                        and "C" in cached
                        and "B" in cached
                        and "marginals" in cached
                    )

                if ok:
                    r[synth_feat_key] = np.asarray(cached[synth_feat_key], dtype=float)
                    r["V"] = np.asarray(cached["V"], dtype=float)
                    r["C"] = np.asarray(cached["C"], dtype=float)
                    if not want_control:
                        r["B"] = np.asarray(cached["B"], dtype=float)
                        r["marginals"] = cached["marginals"]
                    else:
                        r["B"] = None
                        r["marginals"] = None

                    r["kmeans_basis_labels"] = np.asarray(cached.get("kmeans_basis_labels"))
                    r["kmeans_basis_counts"] = np.asarray(cached.get("kmeans_basis_counts"))
                    print(f"[synthetic] using cached synthetic data ({synth_path.name})")
                    return
            except Exception as e:
                print(f"[synthetic] cache read error; recomputing synthetic: {e}")

        seed = 0
        n_bins = 200
        rng = np.random.default_rng(seed)

        X = np.asarray(r[src_feat], dtype=float)  # (N,T)   <-- UPDATED: concat or concat_z
        N, T = X.shape
        M = int(nclus_s)

        print(f"[synthetic] fitting KMeans basis (n={M}) on {src_feat} (zsc={zsc})")
        km = KMeans(n_clusters=M, random_state=0)
        km.fit(X)
        labs = km.predict(X).astype(int)

        V = np.zeros((M, T), dtype=float)
        counts = np.zeros(M, dtype=int)
        for a in range(M):
            idx = labs == a
            counts[a] = int(np.sum(idx))
            if counts[a] > 0:
                V[a, :] = np.mean(X[idx, :], axis=0)

        V_pinv = np.linalg.pinv(V)
        X = X[r['isort']]
        C = X @ V.T  # (N,M)

        if syn_control:
            Xs = C @ V_pinv.T
            B = None
            marginals = None
            method = "kmeans_basis_reconstruct_CV"
        else:
            marginals = []
            B = np.zeros((N, M), dtype=float)
            for a in range(M):
                col = C[:, a]
                lo, hi = np.nanpercentile(col, [0.5, 99.5])
                if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo == hi):
                    lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
                    if lo == hi:
                        lo, hi = lo - 1.0, hi + 1.0

                h, edges = np.histogram(col, bins=int(n_bins), range=(float(lo), float(hi)), density=False)
                h = h.astype(float)
                s = h.sum()
                probs = (h / s) if s > 0 else np.ones_like(h) / len(h)
                centers = 0.5 * (edges[:-1] + edges[1:])

                B[:, a] = rng.choice(centers, size=N, replace=True, p=probs)
                marginals.append(
                    {"alpha": int(a), "edges": edges, "centers": centers, "probs": probs, "n_bins": int(n_bins)}
                )

            Xs = B @ V
            method = "kmeans_basis_iid_hist_marginals"

        # write into r using the correct synthetic key
        r[synth_feat_key] = Xs
        r["V"] = V
        r["C"] = C
        r["B"] = B
        r["marginals"] = marginals
        r["kmeans_basis_labels"] = labs
        r["kmeans_basis_counts"] = counts

        payload = {
            "synthetic": True,
            "syn_control": bool(syn_control),
            "zsc": bool(zsc),
            "order_sig": r["_order_signature"],
            "nclus_basis": int(M),
            "seed": int(seed),
            "method": method,
            "src_feat": src_feat,
            "synth_feat_key": synth_feat_key,
            synth_feat_key: Xs,        # <-- stores concat_zs OR concat_s
            "V": V,
            "C": C,
            "kmeans_basis_labels": labs,
            "kmeans_basis_counts": counts,
        }
        if not syn_control:
            payload["B"] = B
            payload["marginals"] = marginals

        np.save(synth_path, payload, allow_pickle=True)
        print(f"[synthetic] wrote cache ({synth_path.name})")

    # ---------------- feature selection for mapping ----------------
    if synthetic:
        if mapping not in ("kmeans", "rm"):
            raise ValueError("In synthetic=True mode, only mapping='kmeans' or mapping='rm' is supported.")

        # NEW: synthetic uses the same source feature as real mapping would (concat_z or concat)
        src_feat = base_feat
        synth_feat_key = "concat_zs" if zsc else "concat_s"  # NEW: avoid misleading name when not zsc
        _compute_and_cache_synthetic(synth_feat_key=synth_feat_key, src_feat=src_feat)
        feat_key_map = synth_feat_key
    else:
        feat_key_map = base_feat

    n_rows = np.asarray(r[feat_key_map]).shape[0]

    # ---------------- mapping ----------------
    if mapping == "rm":
        rm_cache_path = _cache_path("rm")
        labels, isort = _load_rm_cache(rm_cache_path, n_rows)

        if labels is None or isort is None:
            # with zsc=False, cv is guaranteed False, so this reduces cleanly.
            feat_used = feat_key_map if synthetic else ("concat_z_train" if cv else feat_key_map)
            labels, isort = _compute_and_save_rm(rm_cache_path, feat_used, n_rows)

        clusters = labels.astype(int, copy=False)
        unique_sorted = np.sort(np.unique(clusters))
        cmap = mpl.colormaps["tab20"]
        u_to_idx = {u: (i % 20) for i, u in enumerate(unique_sorted)}
        color_map = {u: cmap(u_to_idx[u]) for u in unique_sorted}
        cols = np.array([color_map[c] for c in clusters])

        r["els"] = [Line2D([0], [0], color=color_map[reg], lw=4, label=f"{reg}") for reg in unique_sorted]
        r["Beryl"] = np.array(br.id2acronym(r["ids"], mapping="Beryl"))
        r["acs"] = labels
        r["cols"] = cols
        r["isort"] = isort

    elif mapping == "kmeans":
        feat_fit = feat_key_map if synthetic else ("concat_z_train" if cv else feat_key_map)
        if feat_fit not in r:
            raise KeyError(f"Feature '{feat_fit}' not found in stack.")

        kmeans_cache_path = _cache_path("kmeans")
        clusters = None

        if (not rerun) and kmeans_cache_path.is_file():
            try:
                cached = np.load(kmeans_cache_path, allow_pickle=True).flat[0]
                if (
                    isinstance(cached, dict)
                    and cached.get("order_sig") == r["_order_signature"]
                    and cached.get("feat_fit") == feat_fit
                    and cached.get("nclus") == nclus
                    and cached.get("synthetic") == bool(synthetic)
                    and cached.get("zsc") == bool(zsc)
                    and (cached.get("nclus_s") == nclus_s if synthetic else True)
                    and (cached.get("syn_control") == bool(syn_control) if synthetic else True)
                    and "kmeans_labels" in cached
                ):
                    clusters = np.asarray(cached["kmeans_labels"], dtype=int).reshape(-1)
                    if clusters.shape[0] != n_rows:
                        clusters = None
                    else:
                        print(f"[kmeans] using cached labels ({kmeans_cache_path.name})")
            except Exception as e:
                print(f"[kmeans] cache read error; recomputing KMeans: {e}")
                clusters = None

        if clusters is None:
            print(
                f"[kmeans] fitting (n={nclus}) on {feat_fit} "
                f"(cv={cv}, synthetic={synthetic}, nclus_s={nclus_s}, syn_control={syn_control}, zsc={zsc})"
            )
            km = KMeans(n_clusters=nclus, random_state=0)
            km.fit(np.asarray(r[feat_fit], dtype=float))
            clusters = km.predict(np.asarray(r[feat_key_map], dtype=float)).astype(int)

            if clusters.shape[0] != n_rows:
                raise ValueError("KMeans labels do not match data length.")

            np.save(
                kmeans_cache_path,
                {
                    "kmeans_labels": clusters,
                    "order_sig": r["_order_signature"],
                    "feat_fit": feat_fit,
                    "nclus": nclus,
                    "synthetic": bool(synthetic),
                    "zsc": bool(zsc),
                    "nclus_s": int(nclus_s) if synthetic else None,
                    "syn_control": bool(syn_control) if synthetic else None,
                },
                allow_pickle=True,
            )
            print(f"[kmeans] wrote cache ({kmeans_cache_path.name})")

        clusters = clusters.astype(int, copy=False)
        unique_sorted = np.sort(np.unique(clusters))
        cmap = mpl.colormaps["tab20"]
        u_to_idx = {u: (i % 20) for i, u in enumerate(unique_sorted)}
        color_map = {u: cmap(u_to_idx[u]) for u in unique_sorted}
        cols = np.array([color_map[c] for c in clusters])

        r["els"] = [Line2D([0], [0], color=color_map[reg], lw=4, label=f"{reg + 1}") for reg in unique_sorted]
        r["Beryl"] = np.array(br.id2acronym(r["ids"], mapping="Beryl"))
        r["acs"] = clusters
        r["cols"] = cols

        # If synthetic and mapping == 'kmeans', also run Rastermap on synthetic data and attach isort/rm_labels.
        if synthetic:
            rm_cache_path = _cache_path("rm")
            labels_rm, isort_rm = _load_rm_cache(rm_cache_path, n_rows)
            if labels_rm is None or isort_rm is None:
                labels_rm, isort_rm = _compute_and_save_rm(rm_cache_path, feat_key_map, n_rows)
            r["isort"] = isort_rm
            r["rm_labels"] = labels_rm

    else:
        acs = np.array(br.id2acronym(r["ids"], mapping=mapping))
        cols = np.array([pal[reg] for reg in acs])
        r["acs"] = acs
        r["cols"] = cols

    # ---------------- attach/ensure Rastermap isort for non-rm mappings when requested ----------------
    if mapping != "rm" and nclus_rm != 100:
        rm_cache_path = _cache_path("rm")
        labels_rm, isort_rm = _load_rm_cache(rm_cache_path, n_rows)
        if labels_rm is None or isort_rm is None:
            feat_used = feat_key_map if synthetic else ("concat_z_train" if cv else feat_key_map)
            labels_rm, isort_rm = _compute_and_save_rm(rm_cache_path, feat_used, n_rows)
        r["isort"] = isort_rm

    r["_feat_map"] = feat_key_map
    r["_zsc"] = bool(zsc)
    return r


def _draw_peth_boundaries(ax, r, vers, yy_max, c_sec):
    """Add vertical window boundaries and labels, matching plot_single_feature."""
   

    d2 = {sec: r['len'][sec] for sec in r['ttypes']}
    h = 0
    for sec in d2:
        xv = d2[sec] + h
        ax.axvline(xv / c_sec, linestyle='--', linewidth=1, color='grey')
        ax.text(xv / c_sec - d2[sec] / (2 * c_sec), yy_max,
                '   ' + r['peth_dict'][sec], rotation=90, color='k',
                fontsize=10, ha='center', va='bottom')
        h += d2[sec]


def plot_example_neurons(
        n: int = 20,
        vers: str = 'concat',
        mapping: str = 'kmeans',
        seed: Optional[int] = None,
        max_categories: int = 101,
        offset_scale: float = 4.0,
        linewidth: float = 1.3,
        savefig: bool = True,
        save_formats: tuple = ('png',),
        dpi: int = 200,
        show: bool = True,
        annotate: bool = True,
        label_key: str = 'Beryl',
        label_fontsize: int = 8,
        label_pad_frac: float = 0.01,
        nclus: int = 25,
        cv: bool = False,
        sing_clus: Union[bool, int] = False,
        no_filts=False,
        min_max_fr: Optional[Tuple[float, float]] = (0.1, 100),
        min_max_lz: Optional[Tuple[float, float]] = (0.0, 0.6),
):
    """
    Plot example neurons per cluster (mapping categories).

    If sing_clus is an int, only that cluster ID is plotted.
    If sing_clus is False (default), all clusters are plotted.
    """

    if no_filts:
        min_max_fr = None
        min_max_lz = None

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    feat = 'concat_z'
    r = regional_group(mapping, vers=vers, cv=cv, nclus=nclus)
    if feat not in r:
        raise KeyError(f"Feature '{feat}' not in r. Available: {list(r.keys())}")

    if 'fr' not in r:
        raise KeyError("'fr' (firing rate) not found in r.")
    if 'lz' not in r:
        raise KeyError("'lz' (Lempel–Ziv complexity) not found in r.")

    if 'ids' not in r:
        raise KeyError("'ids' not found in r (needed for Cosmos acronym mapping).")

    # Cosmos acronyms for all neurons (as per your convention)
    try:
        from iblatlas.regions import BrainRegions
        br = BrainRegions()
        cosm_all = br.id2acronym(r['ids'], 'Cosmos')
        cosm_all = np.asarray(cosm_all, dtype=object)
    except Exception:
        cosm_all = None

    acs_vals = np.asarray(r['acs'])
    cats = np.unique(acs_vals)
    if cats.size >= max_categories:
        print(f"[info] Found {cats.size} 'acs' categories (>= {max_categories}). Skipping.")
        return

    # restrict to single cluster, if requested
    if sing_clus is not False:
        sing_clus_int = int(sing_clus)
        if sing_clus_int not in cats:
            print(f"[info] sing_clus={sing_clus_int} not present in categories {cats}. Nothing to plot.")
            return
        cats = np.array([sing_clus_int])
    else:
        show = False  # only show when single cluster

    # saving
    if savefig:
        save_dir = Path(pth_dmn, 'figs')
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] Figures will be saved to {save_dir}")

    xx = np.arange(r[feat].shape[1]) / c_sec  # seconds

    for cat in cats:
        # -----------------------------
        # filtering by FR and LZ
        # -----------------------------
        idx_all = np.where(acs_vals == cat)[0]

        if min_max_fr is not None:
            lo, hi = min_max_fr
            idx_all = idx_all[(r['fr'][idx_all] >= lo) & (r['fr'][idx_all] <= hi)]

        if min_max_lz is not None:
            lo, hi = min_max_lz
            idx_all = idx_all[(r['lz'][idx_all] >= lo) & (r['lz'][idx_all] <= hi)]

        if idx_all.size == 0:
            continue

        k = min(n, idx_all.size)
        if idx_all.size >= k:
            samp = np.random.choice(idx_all, size=k, replace=False)
        else:
            samp = np.array(random.choices(idx_all, k=k))

        fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
        try:
            stds = [np.nanstd(r[feat][i]) for i in samp]
            base_off = 2.0 * (np.nanmedian(stds) if len(stds) else 1.0)
            off = base_off * offset_scale

            y_max_seen = -np.inf
            for j, i in enumerate(samp):
                yi = r[feat][i]
                yy = yi + j * off
                lbl = str(r[label_key][i])
                color = pal[lbl]

                ax.plot(xx, yy, linewidth=linewidth, color='black', alpha=0.9)
                y_max_seen = max(y_max_seen, np.nanmax(yy))

                if annotate:
                    x_offset = xx[0] - 0.02 * (xx[-1] - xx[0])
                    prefix = max(1, int(0.02 * yi.size))
                    y0 = np.nanmedian(yi[:prefix]) + j * off

                    ax.text(x_offset, y0, lbl,
                            fontsize=label_fontsize,
                            va='center', ha='right',
                            color=color,
                            alpha=0.95,
                            clip_on=False)

                    fr_val = float(r['fr'][i])
                    lz_val = float(r['lz'][i])
                    info_txt = f"{fr_val:.2f}, {lz_val:.2f}"
                    y1 = y0 - 0.3 * off
                    ax.text(x_offset, y1, info_txt,
                            fontsize=label_fontsize - 1,
                            va='center', ha='right',
                            color=color,
                            alpha=0.9,
                            clip_on=False)

            _draw_peth_boundaries(ax, r, vers, y_max_seen, c_sec)

            ax.set_xlabel("time [s]")
            ax.set_ylabel("z-scored firing rate (stacked)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks([])

            ax.set_xlim(-0.5, xx[-1])

            # --- NEW: top-3 Cosmos regions (by fraction among shown example neurons), text in Cosmos colors ---
            if cosm_all is not None:
                cosm_s = np.asarray(cosm_all[samp], dtype=object)
                # guard: drop empty/nan-like labels
                cosm_s = cosm_s[(cosm_s != None) & (cosm_s != '')]  # noqa: E711
                if cosm_s.size > 0:
                    u, cts = np.unique(cosm_s, return_counts=True)
                    frac = cts / float(cosm_s.size)

                    # sort by decreasing fraction, then name for stability
                    order = np.lexsort((u.astype(str), -frac))
                    u, frac = u[order], frac[order]

                    top = min(3, u.size)
                    x0, y0 = 0.0, 1.1
                    dy = 0.04  # axes fraction
                    fs = max(6, label_fontsize - 1)

                    for t in range(top):
                        name = str(u[t])
                        pct = int(np.round(100.0 * float(frac[t])))
                        txt = f"{name} ({pct}%)"
                        col = pal[name] if (name in pal) else 'black'
                        ax.text(x0, y0 - t * dy, txt,
                                transform=ax.transAxes,
                                va='top', ha='left',
                                fontsize=fs,
                                color=col,
                                alpha=0.95)

            # counts
            N0 = np.sum(acs_vals == cat)
            N1 = len(idx_all)
            if min_max_fr is None:
                N_fr = N0
            else:
                lo, hi = min_max_fr
                fr_mask = (r['fr'][acs_vals == cat] >= lo) & (r['fr'][acs_vals == cat] <= hi)
                N_fr = int(np.sum(fr_mask))

            title = f"{mapping} = {cat} of {nclus}, (n={k} of {N1} neurons)"
            if min_max_fr is not None:
                lo, hi = min_max_fr
                title += f"\nFR ∈ [{lo:.2f}, {hi:.2f}]   ({N0} → {N_fr})"
            if min_max_lz is not None:
                lo, hi = min_max_lz
                title += f"\nLZ ∈ [{lo:.2f}, {hi:.2f}]   ({N_fr} → {N1})"

            fig.suptitle(title, fontsize=12, weight='bold')
            fig.subplots_adjust(left=0.12, top=0.90)

            if savefig:
                for fmt in save_formats:
                    fn = save_dir / f"{mapping}_{cat}_of{nclus}_n{n}_cv{int(cv)}.{fmt}"
                    fig.savefig(fn, dpi=dpi, bbox_inches='tight')
                print(f"  saved: {mapping}_{cat}_of{nclus}_n{n}_cv{cv}.{save_formats[0]}")

            if show:
                plt.show(block=False)

        finally:
            gc.collect()


def plot_cluster_mean_PETHs(
    r, mapping, feat, vers='concat',
    axx=None, alone=True,
    extraclus=None,  # NEW: list of cluster IDs to overlay
cv=True):
    """
    Plot mean PETH per cluster using the segment order/labels from the data file.

    Parameters
    ----------
    extraclus : list[int] or None
        - If [] or None: keep old behavior (one axis per cluster).
        - If non-empty list: overlay those cluster mean traces on ONE axis.
          Cluster IDs must be valid values in r['acs'] (typically 0..nclus-1).
    """

    if feat not in r:
        raise KeyError(f"Feature '{feat}' not in result dict.")
    if 'acs' not in r or 'cols' not in r:
        raise KeyError("Result dict must contain 'acs' and 'cols'.")
    if 'len' not in r or not isinstance(r['len'], dict) or len(r['len']) == 0:
        raise KeyError("r['len'] (segment lengths) missing or empty.")
    if 'peth_dict' not in r:
        r['peth_dict'] = {k: k for k in r['len'].keys()}

    # normalize extraclus
    if extraclus is None:
        extraclus = []
    if not isinstance(extraclus, (list, tuple, np.ndarray)):
        raise TypeError("extraclus must be a list/tuple/array of integers (or empty).")
    extraclus = [int(x) for x in extraclus]

    clu_vals = np.array(sorted(np.unique(r['acs'])))
    n_clu = len(clu_vals)
    if n_clu > 50 and len(extraclus) == 0:
        print('too many (>50) line plots!')
        return

    n_bins = r[feat].shape[1]
    xx = np.arange(n_bins) / c_sec

    ordered_segments = list(r['len'].keys())
    seg_lengths = [r['len'][s] for s in ordered_segments]
    if sum(seg_lengths) != n_bins:
        print(f"[warn] sum(r['len'])={sum(seg_lengths)} != n_bins={n_bins}")

    # -------------------------------
    # NEW MODE: overlay selected clusters on ONE axis
    # -------------------------------
    if len(extraclus) > 0:
        # Validate requested clusters exist
        valid = set(int(c) for c in clu_vals.tolist())
        bad = [c for c in extraclus if c not in valid]
        if bad:
            raise ValueError(f"extraclus contains invalid cluster IDs {bad}. Valid: {sorted(valid)}")

        # Prepare single axis
        if axx is None:
            fg, ax = plt.subplots(figsize=(6, 3))
        else:
            ax = axx if not isinstance(axx, (list, np.ndarray)) else axx[0]

        ymax_global = 0.0

        # Plot each requested cluster on the same axis
        for clu in extraclus:
            idx = np.where(r['acs'] == clu)[0]
            if idx.size == 0:
                continue
            yy = np.mean(r[feat][idx, :], axis=0)
            col = r['cols'][idx[0]]
            ax.plot(xx, yy, color=col, linewidth=2, label=str(clu))
            if yy.size:
                ymax_global = max(ymax_global, float(np.max(yy)))

        # Segment boundaries + labels (draw once)
        h = 0
        for s in ordered_segments:
            seg_len = r['len'][s]
            xv_bins = h + seg_len
            if xv_bins > n_bins:
                break
            ax.axvline(xv_bins / c_sec, linestyle='--', linewidth=1, color='grey')

            seg_mid = h + seg_len / 2.0
            ax.text(
                seg_mid / c_sec,
                ymax_global,
                '   ' + r['peth_dict'].get(s, s),
                rotation=90, color='k', fontsize=10, ha='center'
            )
            h += seg_len

        ax.set_xlim(0, n_bins / c_sec)
        ax.set_xlabel('time [sec]')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend showing cluster IDs
        ax.legend(title="cluster", frameon=False, loc="best")

        if alone:
            plt.tight_layout()

        return  # important: do not run the old multi-axes path

    # -------------------------------
    # OLD MODE: one axis per cluster (unchanged behavior)
    # -------------------------------
    if axx is None:
        fg, axx = plt.subplots(nrows=n_clu, sharex=True, sharey=False, figsize=(6, 8))
    if not isinstance(axx, (list, np.ndarray)):
        axx = [axx]
    if len(axx) != n_clu:
        raise ValueError(f"Expected {n_clu} axes, got {len(axx)}.")

    for k, clu in enumerate(clu_vals):
        idx = np.where(r['acs'] == clu)[0]

        axx[k].spines['top'].set_visible(False)
        axx[k].spines['right'].set_visible(False)
        axx[k].spines['left'].set_visible(False)
        axx[k].tick_params(left=False, labelleft=False)
        axx[k].set_ylabel(clu,rotation=0, labelpad=10)

        yy = np.mean(r[feat][idx, :], axis=0)
        col = r['cols'][idx[0]]
        axx[k].plot(xx, yy, color=col, linewidth=2)

        if k != (n_clu - 1):
            axx[k].spines['bottom'].set_visible(False)
            axx[k].tick_params(bottom=False, labelbottom=False)
        else:
            # keep bottom axis ticks only for the last panel
            axx[k].spines['bottom'].set_visible(True)
            axx[k].tick_params(bottom=True, labelbottom=True)

        h = 0
        ymax = float(np.max(yy)) if yy.size else 0.0
        for s in ordered_segments:
            seg_len = r['len'][s]
            xv_bins = h + seg_len
            if xv_bins > n_bins:
                break
            axx[k].axvline(xv_bins / c_sec, linestyle='--', linewidth=1, color='grey')
            if k == 0:
                seg_mid = h + seg_len / 2.0
                axx[k].text(seg_mid / c_sec, ymax,
                            '   ' + r['peth_dict'].get(s, s),
                            rotation=90, color='k', fontsize=10, ha='center')
            h += seg_len

        axx[k].set_xlim(0, n_bins / c_sec)

    axx[-1].set_xlabel('time [sec]')

    if alone:
        plt.tight_layout()

    # add information in figure window title
    fg = plt.gcf()
    fg.canvas.manager.set_window_title(
        f"{feat} | {mapping} | nclus={n_clu} | cv={int(cv)} | {vers}")

    plt.show()


def plot_rastermap(
    vers="concat",
    feat="concat_z",
    regex="ECT",
    exa=False,
    mapping="rm",
    bg="#c2a37a",  
    img_only=False,
    interp="antialiased",
    single_reg=False,
    cv=True,
    bg_bright=0.99, # put to 1 for bg constant
    vmax=2,
    rerun=False,
    sort_method="rastermap",
    nclus=25,
    nclus_rm=100,
    nclus_s=25,
    clsfig=False,
    bounds=False,
    grid_upsample=0,
    zsc=True,
    locality=0.75,
    time_lag_window=5,
    symmetric=False,
    clabels="all",
    synthetic: bool = False,
    syn_control: bool = False,
):
    """
    Updated for zsc=False:
      - real: use 'concat' instead of 'concat_z'
      - synthetic: use 'concat_s' instead of 'concat_zs'
    """

    r = regional_group(
        mapping,
        vers=vers,
        ephys=False,
        nclus=nclus,
        nclus_rm=nclus_rm,
        rerun=rerun,
        cv=cv,
        grid_upsample=grid_upsample,
        nclus_s=nclus_s,
        locality=locality,
        time_lag_window=time_lag_window,
        symmetric=symmetric,
        synthetic=synthetic,
        syn_control=syn_control,
        zsc=zsc,
    )

    # ---------------- choose correct default feature keys ----------------
    # base feature for real
    base_feat = "concat_z" if zsc else "concat"
    # base synthetic feature
    base_feat_syn = "concat_zs" if zsc else "concat_s"

    # If user passed the "z-scored name" while zsc=False, transparently map it.
    # Likewise for synthetic.
    if synthetic:
        # normalize feat request to synthetic keyspace
        if feat in ("concat_z", "concat"):
            feat = base_feat_syn
        elif feat == "single_reg":
            feat = base_feat_syn
        elif feat == "concat_zs" and (not zsc):
            feat = "concat_s"
        elif feat == "concat_s" and zsc:
            feat = "concat_zs"
    else:
        # normalize feat request to real keyspace
        if feat == "concat_z" and (not zsc):
            feat = "concat"
        elif feat == "concat" and zsc:
            feat = "concat_z"

    if exa:
        plot_cluster_mean_PETHs(r, mapping, feat, cv=cv)

    plt.ion()
    if clsfig:
        plt.ioff()

    # ---------------- choose which feature matrix to plot ----------------
    feat_plot = feat

    # keep old behavior for 'single_reg': just means filtering by Beryl, matrix remains base_feat/base_feat_syn
    if feat_plot == "single_reg":
        feat_plot = base_feat_syn if synthetic else base_feat

    if feat_plot not in r:
        raise KeyError(
            f"Feature '{feat_plot}' not found in results dict. "
            f"(synthetic={synthetic}, zsc={zsc}). Available keys: {list(r.keys())[:40]} ..."
        )

    spks = r[feat_plot]

    # ---------------- choose sorting algorithm ----------------
    if sort_method == "rastermap":
        isort = r["isort" if feat_plot != "ephysTF" else "isort_e"]

    elif sort_method == "umap":
        # embeddings are stack-derived; if you need them to respect zsc=False you must add separate embedding caches upstream
        assert "umap_z" in r, "r['umap_z'] not found in results."
        isort = np.argsort(r["umap_z"][:, 0])

    elif sort_method == "pca":
        assert "pca_z" in r, "r['pca_z'] not found in results."
        isort = np.argsort(r["pca_z"][:, 0])

    elif sort_method == "acs":
        if "acs" not in r:
            raise KeyError("r['acs'] not found in results.")

        acs_arr = np.asarray(r["acs"])

        if mapping == "Cosmos":
            regs_can = ["Isocortex", "OLF", "HPF", "CTXsp", "CNU", "TH", "HY", "MB", "HB", "CB", "void", "root"]
            rank = {reg: i for i, reg in enumerate(regs_can)}
            unk = len(regs_can) + 1
            keys = np.array([rank.get(str(a), unk) for a in acs_arr], dtype=int)
            isort = np.argsort(keys, kind="stable")

        elif mapping == "Beryl":
            p = Path(iblatlas.__file__).parent / "beryl.npy"
            regs_can = br.id2acronym(np.load(p), mapping="Beryl")
            rank = {reg: i for i, reg in enumerate(regs_can)}
            unk = len(regs_can) + 1
            keys = np.array([rank.get(str(a), unk) for a in acs_arr], dtype=int)
            isort = np.argsort(keys, kind="stable")

        else:
            try:
                isort = np.argsort(acs_arr, kind="stable")
            except TypeError:
                isort = np.argsort(acs_arr.astype(str), kind="stable")

    else:
        raise ValueError(f"Unknown sort_method: {sort_method}")

    data = spks[isort]
    row_colors = np.array(r["cols"])[isort]
    clus_sorted = np.asarray(r["acs"])[isort]

    del spks
    gc.collect()

    if single_reg:
        # filter by Beryl labels (works for both real and synthetic)
        acsB = np.array(r["Beryl"])[isort]
        n = len(acsB)
        n_ex = int(np.sum(acsB == regex))
        print("number of cells in example region:", n_ex)

        data = data[acsB == regex]
        row_colors = row_colors[acsB == regex]
        clus_sorted = clus_sorted[acsB == regex]
        print(f"filtering rastermap for {regex} cells only")

        del acsB
        gc.collect()

    n_rows, n_cols = data.shape

    fig, ax = plt.subplots(figsize=(6, 8))

    # ---------------- display scaling (FIX for zsc=False) ----------------
    data = np.asarray(data, dtype=float)
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("Raster data contains no finite values (all NaN/inf).")

    if zsc:
        # legacy behavior for z-scored features
        vmin = 0.0
        vmax_ = float(vmax)
        if (not np.isfinite(vmax_)) or (vmax_ <= vmin):
            vmax_ = vmin + 1.0
    else:
        # robust scaling for raw features (concat / concat_s)
        vmin, vmax_ = np.nanpercentile(data[finite], [1.0, 99.0])
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax_)) or (vmin == vmax_):
            vmin = float(np.nanmin(data[finite]))
            vmax_ = float(np.nanmax(data[finite]))
            if vmin == vmax_:
                vmin -= 1.0
                vmax_ += 1.0

    data_clipped = np.clip(data, vmin, vmax_)
    denom = (vmax_ - vmin) if (vmax_ > vmin) else 1.0
    gray_scaled = (data_clipped - vmin) / denom

    # ---------------- background handling ----------------
    if bg:
        # Case 1: constant background color
        if (not isinstance(bg, bool)) and is_color_like(bg):
            bg_rgb = np.array(to_rgba(bg), dtype=np.float32)[:3]

            # brighten toward white exactly like the row-color branch
            bg_rgb = bg_rgb * bg_bright + (1.0 - bg_bright)

            rgba = np.empty((*gray_scaled.shape, 4), dtype=np.float32)
            rgba[..., :3] = bg_rgb[None, None, :]
            rgba[..., 3] = 1.0

            # ink strength (optional but useful)
            ink = gray_scaled.astype(np.float32)
            ink = gray_scaled ** 1.25
            # ink = ink ** 0.8  # optional: boosts faint structure

            for c in range(3):
                rgba[..., c] *= (1.0 - ink)

            ax.imshow(rgba, aspect="auto", interpolation=interp, zorder=1)
            del rgba
            gc.collect()

        # Case 2: legacy per-row colored background
        else:
            rgba_bg = np.array([to_rgba(c) for c in row_colors], dtype=np.float32)
            rgba_bg = np.broadcast_to(
                rgba_bg[:, np.newaxis, :], (*data.shape, 4)
            ).copy()

            rgba_bg[..., :3] = rgba_bg[..., :3] * bg_bright + (1 - bg_bright)

            alpha_overlay = gray_scaled
            for c in range(3):
                rgba_bg[..., c] *= (1 - alpha_overlay)

            rgba_bg[..., 3] = 1.0
            ax.imshow(rgba_bg, aspect="auto", interpolation=interp, zorder=1)
            del rgba_bg
            gc.collect()

    # Default: white background
    else:
        rgba_overlay = np.zeros((*gray_scaled.shape, 4), dtype=np.float32)
        inv_gray = 1.0 - gray_scaled
        rgba_overlay[..., :3] = inv_gray[..., np.newaxis]
        rgba_overlay[..., 3] = 1.0
        ax.imshow(rgba_overlay, aspect="auto", interpolation=interp, zorder=2)
        del rgba_overlay
        gc.collect()

    if grid_upsample > 0:
        bounds = False

    if bounds:
        rc = np.asarray(clus_sorted)
        cluster_changes = rc[1:] != rc[:-1]
        boundaries = np.where(cluster_changes)[0] + 0.5

        for y in boundaries:
            ax.axhline(y, color="k", linewidth=0.6, zorder=5)

        trans_right = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        n_rows_ = data.shape[0]
        edges = np.concatenate(([0.5], boundaries, [n_rows_ - 0.5]))
        n_segments = len(edges) - 1
        fontsize = np.clip(300 / max(int(nclus), 1), 5, mpl.rcParams["font.size"])

        if clabels == "all":
            label_idxs = np.arange(n_segments)
        elif isinstance(clabels, int):
            label_idxs = (
                np.array([], dtype=int) if clabels < 1 else
                np.array([0]) if clabels == 1 else
                np.linspace(0, n_segments - 1, clabels, dtype=int)
            )
        else:
            raise ValueError("clabels must be 'all' or a positive integer")

        for i in label_idxs:
            y0, y1 = edges[i], edges[i + 1]
            mid_y = 0.5 * (y0 + y1)
            row_idx = int(np.clip(np.floor(mid_y), 0, n_rows_ - 1))

            if sort_method == "acs" and mapping in ("Beryl", "Cosmos"):
                label = str(clus_sorted[row_idx])
            else:
                try:
                    label = str(int(clus_sorted[row_idx]))
                except Exception:
                    label = str(clus_sorted[row_idx])

            ax.text(
                1.01, mid_y, label,
                transform=trans_right,
                va="center", ha="left",
                fontsize=fontsize,
                color="k",
                clip_on=False,
            )

        ax.text(
            1.08, 0.5, "clusters",
            transform=ax.transAxes,
            rotation=90,
            va="center", ha="center",
            fontsize=mpl.rcParams["axes.labelsize"],
            color="k",
            clip_on=False,
        )

    if feat_plot != "ephysTF":
        if "len" not in r or not isinstance(r["len"], dict) or len(r["len"]) == 0:
            raise KeyError("Segment lengths r['len'] missing or empty; cannot draw boundaries/labels.")

        ordered_segments = list(r["len"].keys())
        labels = r.get("peth_dict", {})

        if data.shape[1] != sum(r["len"].values()):
            print(f"[warn] data.shape[1] ({data.shape[1]}) != sum(len) ({sum(r['len'].values())})")

        trans_top = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)

        h = 0
        for seg in ordered_segments:
            seg_len = r["len"][seg]
            xv = h + seg_len
            if xv > n_cols:
                break
            ax.axvline(xv, linestyle="--", linewidth=1, color="grey")

            midpoint = h + seg_len / 2.0
            if not img_only:
                ax.text(
                    midpoint, 1.02,
                    labels.get(seg, seg),
                    rotation=90,
                    color="k",
                    fontsize=10,
                    ha="center",
                    va="bottom",
                    transform=trans_top,
                    clip_on=False,
                )
            h += seg_len

        x_ticks = np.arange(0, n_cols, c_sec)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{int(tick / c_sec)}" for tick in x_ticks])
    else:
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(r["fts"], rotation=90)

    ax.set_xlabel("time [sec]")
    ax.set_ylabel(f"cells in {regex}" if feat == "single_reg" else "cells")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if img_only:
        ax.axis("off")
        if single_reg:
            ax.set_title(f"{regex} ({n_ex})", fontsize=25, color=pal[regex], transform=ax.transAxes)

    ax.set_xlim(0, n_cols)
    plt.tight_layout()

    descriptor = (
        f"map_{mapping}"
        f"_cv_{int(cv)}"
        f"_zsc_{int(bool(zsc))}"                 # NEW
        + (f"_nclus_{nclus}" if mapping == "kmeans" else "")
        + f"_nclus_rm_{nclus_rm}"
        + (f"_nclus_s_{nclus_s}" if synthetic else "")
        + f"_sort_{sort_method}"
    )

    if single_reg:
        descriptor += f"_reg_{regex}"
    fname = descriptor + ".svg"

    try:
        fig.canvas.manager.set_window_title(f"{descriptor}")
    except Exception:
        pass

    if clsfig:
        plt.close(fig)

    for v in ("sig", "img_array", "isort", "r"):
        if v in locals():
            del locals()[v]

