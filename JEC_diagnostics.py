#!/usr/bin/env python3
"""
Compares corrected vs uncorrected FatJet pT and MET distributions across
background and signal samples, with cross-section weighting.

Produces: pT distributions with gen ratio, MET distributions, MET closure
(reco - gen), delta-phi(MET_reco, MET_gen), MET resolution vs gen MET, and
mean jet response / resolution vs pT.  Three MET variants are compared:
uncorrected, official-JEC-only, and all corrections applied.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import uproot
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
from pathlib import Path
from tqdm import tqdm

# Set CMS style
plt.style.use(hep.style.CMS)

# Base path for the datasets
#BASE_PATH = "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_JEC/2017/s_channel_scouting_pre_selection_with_custom_JEC_inclusive/nominal"
BASE_PATH = "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_JEC/2017/s_channel_scouting_pre_selection_with_custom_JEC_lepton_veto/nominal"

# Datasets organised by group — processed one group at a time to keep memory low
DATASET_GROUPS = {
    "QCD": {
        "QCD_HT300to500":  f"{BASE_PATH}/QCD_HT300to500",
        "QCD_HT500to700":  f"{BASE_PATH}/QCD_HT500to700",
        "QCD_HT700to1000": f"{BASE_PATH}/QCD_HT700to1000",
        "QCD_HT1000to1500": f"{BASE_PATH}/QCD_HT1000to1500",
        "QCD_HT1500to2000": f"{BASE_PATH}/QCD_HT1500to2000",
        "QCD_HT2000toInf": f"{BASE_PATH}/QCD_HT2000toInf",
    },
    "TTJets": {
        #"TTJets_DiLept":            f"{BASE_PATH}/TTJets_DiLept",
        "TTJets_HT-1200to2500":     f"{BASE_PATH}/TTJets_HT-1200to2500",
        "TTJets_HT-2500toInf":      f"{BASE_PATH}/TTJets_HT-2500toInf",
        "TTJets_HT-600to800":       f"{BASE_PATH}/TTJets_HT-600to800",
        "TTJets_HT-800to1200":      f"{BASE_PATH}/TTJets_HT-800to1200",
        #"TTJets_SingleLeptFromT":    f"{BASE_PATH}/TTJets_SingleLeptFromT",
        #"TTJets_SingleLeptFromTbar": f"{BASE_PATH}/TTJets_SingleLeptFromTbar",
        #"TTJets_TuneCP5":           f"{BASE_PATH}/TTJets_TuneCP5",
    },
    "WJets": {
        "WJetsToLNu_HT-1200To2500": f"{BASE_PATH}/WJetsToLNu_HT-1200To2500",
        "WJetsToLNu_HT-2500ToInf":  f"{BASE_PATH}/WJetsToLNu_HT-2500ToInf",
        "WJetsToLNu_HT-400To600":   f"{BASE_PATH}/WJetsToLNu_HT-400To600",
        "WJetsToLNu_HT-600To800":   f"{BASE_PATH}/WJetsToLNu_HT-600To800",
        "WJetsToLNu_HT-800To1200":  f"{BASE_PATH}/WJetsToLNu_HT-800To1200",
    },
    "ZJets": {
        "ZJetsToNuNu_HT-1200To2500": f"{BASE_PATH}/ZJetsToNuNu_HT-1200To2500",
        "ZJetsToNuNu_HT-2500ToInf":  f"{BASE_PATH}/ZJetsToNuNu_HT-2500ToInf",
        "ZJetsToNuNu_HT-400To600":   f"{BASE_PATH}/ZJetsToNuNu_HT-400To600",
        "ZJetsToNuNu_HT-600To800":   f"{BASE_PATH}/ZJetsToNuNu_HT-600To800",
        "ZJetsToNuNu_HT-800To1200":  f"{BASE_PATH}/ZJetsToNuNu_HT-800To1200",
    },
    # "Signal": {
    #     "Signal_mMed-1100_mDark-20_rinv-0.3": f"{BASE_PATH}/s-channel_mMed-1100_mDark-20_rinv-0.3",
    # },
}

# Integrated luminosity for 2017 (fb^-1)
LUMI = 41.53  # fb^-1


# Max parallel workers for file-level and dataset-level reading
FILE_WORKERS    = 8   # concurrent files within one dataset
DATASET_WORKERS = 4   # concurrent datasets within one group


def get_file_list(dataset_path):
    """Get list of ROOT files in the dataset directory via xrootd."""
    # For xrootd paths, we need to list files using uproot or subprocess
    # This assumes files follow a pattern like *.root
    import subprocess
    
    try:
        # Try to list files using xrdfs
        cmd = f"xrdfs cmsdcache-kit-disk.gridka.de:1094 ls {dataset_path.replace('root://cmsdcache-kit-disk.gridka.de:1094/', '')}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        files = [f"root://cmsdcache-kit-disk.gridka.de:1094/{line.strip()}" 
                 for line in result.stdout.split('\n') 
                 if line.strip().endswith('.root')]
        return files
    except Exception as e:
        print(f"Warning: Could not list files for {dataset_path}: {e}")
        return []


def _read_single_file(file_path):
    """Read one ROOT file; returns raw arrays + metadata, or None on failure."""
    name = Path(file_path).name
    try:
        with uproot.open(file_path) as f:
            try:
                initial_events = f["CutFlow"]["Initial"].array(library="np")[0]
            except Exception as e:
                tqdm.write(f"    Warning: CutFlow missing in {name}: {e}")
                return None
            try:
                xsec = f["Metadata"]["GenCrossSection"].array(library="np")[0]
            except Exception as e:
                tqdm.write(f"    Warning: GenCrossSection missing in {name}: {e}")
                xsec = None
            try:
                # Read all branches in a single round trip.
                # Variant branches are optional for backward compatibility.
                events = f["Events"]
                branch_names = set(events.keys())

                branches = [
                    "FatJet_pt", "FatJet_pt_uncorr", "GenFatJet_pt",
                    "FatJet_eta", "FatJet_phi", "FatJet_mass",
                    "GenFatJet_phi",
                    "FatJet_isGood",
                    "ScoutMET_pt", "ScoutMET_pt_uncorr", "genMET_pt",
                    "ScoutMET_phi", "ScoutMET_phi_uncorr", "genMET_phi",
                ]
                if "ScoutMET_pt_official_only" in branch_names:
                    branches.append("ScoutMET_pt_official_only")
                if "ScoutMET_pt_all_corr" in branch_names:
                    branches.append("ScoutMET_pt_all_corr")
                if "ScoutMET_phi_official_only" in branch_names:
                    branches.append("ScoutMET_phi_official_only")
                if "ScoutMET_phi_all_corr" in branch_names:
                    branches.append("ScoutMET_phi_all_corr")
                if "MT01FatJetMET" in branch_names:
                    branches.append("MT01FatJetMET")
                if "RTFatJet" in branch_names:
                    branches.append("RTFatJet")
                if "DeltaPhiMinFatJetMET" in branch_names:
                    branches.append("DeltaPhiMinFatJetMET")

                arrs = events.arrays(branches, library="ak")

                met_uncorr = arrs["ScoutMET_pt_uncorr"]
                met_official_only = (
                    arrs["ScoutMET_pt_official_only"]
                    if "ScoutMET_pt_official_only" in arrs.fields
                    else arrs["ScoutMET_pt"]
                )
                met_all_corr = (
                    arrs["ScoutMET_pt_all_corr"]
                    if "ScoutMET_pt_all_corr" in arrs.fields
                    else arrs["ScoutMET_pt"]
                )
                met_phi_uncorr = arrs["ScoutMET_phi_uncorr"]
                met_phi_official_only = (
                    arrs["ScoutMET_phi_official_only"]
                    if "ScoutMET_phi_official_only" in arrs.fields
                    else arrs["ScoutMET_phi"]
                )
                met_phi_all_corr = (
                    arrs["ScoutMET_phi_all_corr"]
                    if "ScoutMET_phi_all_corr" in arrs.fields
                    else arrs["ScoutMET_phi"]
                )
                # Pre-compute MT for each MET variant using the 2 leading corrected jets.
                # JEC does not change jet eta/phi, so the dijet direction is shared.
                def _calc_mt_arr(met_pt, met_phi):
                    j0 = arrs["FatJet_pt"][:, 0:1]; j1 = arrs["FatJet_pt"][:, 1:2]
                    j0_phi = arrs["FatJet_phi"][:, 0:1]; j1_phi = arrs["FatJet_phi"][:, 1:2]
                    j0_eta = arrs["FatJet_eta"][:, 0:1]; j1_eta = arrs["FatJet_eta"][:, 1:2]
                    j0_m = arrs["FatJet_mass"][:, 0:1]; j1_m = arrs["FatJet_mass"][:, 1:2]
                    # dijet Cartesian sum
                    j0_px = j0 * np.cos(j0_phi); j0_py = j0 * np.sin(j0_phi)
                    j1_px = j1 * np.cos(j1_phi); j1_py = j1 * np.sin(j1_phi)
                    jj_px = ak.flatten(j0_px + j1_px); jj_py = ak.flatten(j0_py + j1_py)
                    jj_pt = np.sqrt(jj_px**2 + jj_py**2)
                    jj_phi = np.arctan2(jj_py, jj_px)
                    # dijet invariant mass (scalar sum via 4-vec is more correct but
                    # requires eta; use E=sqrt(pt^2*cosh^2(eta)+m^2) per jet)
                    j0_E = np.sqrt((j0 * np.cosh(j0_eta))**2 + j0_m**2)
                    j1_E = np.sqrt((j1 * np.cosh(j1_eta))**2 + j1_m**2)
                    j0_pz = j0 * np.sinh(j0_eta); j1_pz = j1 * np.sinh(j1_eta)
                    jj_E = ak.flatten(j0_E + j1_E); jj_pz = ak.flatten(j0_pz + j1_pz)
                    jj_m2 = jj_E**2 - jj_px**2 - jj_py**2 - jj_pz**2
                    jj_m2 = ak.where(jj_m2 > 0, jj_m2, ak.zeros_like(jj_m2))
                    jj_m = np.sqrt(jj_m2)
                    dphi = np.arctan2(np.sin(met_phi - jj_phi), np.cos(met_phi - jj_phi))
                    Et_jj = np.sqrt(jj_m2 + jj_pt**2)
                    mt2 = jj_m2 + 2 * (Et_jj * met_pt - jj_pt * met_pt * np.cos(dphi))
                    mt2 = ak.where(mt2 > 0, mt2, ak.zeros_like(mt2))
                    return np.array(np.sqrt(mt2))

                # Pre-compute deltaphimin for each MET variant
                # Matches the skimmer's calculate_delta_phi_min function:
                # takes min of abs(delta_phi) between MET and ALL good jets (filtered by FatJet_isGood)
                def _calc_deltaphimin_arr_reco(met_phi, good_jet_mask):
                    # Get phi for ALL good jets (filtered by FatJet_isGood mask)
                    # The skimmer uses: events.FatJet_phi[events.FatJet_isGood]
                    good_jet_phis = arrs["FatJet_phi"][good_jet_mask]  # shape: (n_events, n_good_jets_per_event)
                    
                    # Broadcast met_phi to same shape as good_jet_phis for element-wise computation
                    met_phi_broadcast = ak.broadcast_arrays(met_phi, good_jet_phis)[0]
                    
                    # Compute delta_phi using proper wrapping: arctan2(sin(dphi), cos(dphi))
                    # This matches the LorentzVector.delta_phi() method
                    dphi = np.arctan2(np.sin(good_jet_phis - met_phi_broadcast), 
                                     np.cos(good_jet_phis - met_phi_broadcast))
                    
                    # Take absolute value and minimum across all good jets (axis=1)
                    # This matches: ak.min(abs(jets.delta_phi(met)), axis=1)
                    deltaphimin = ak.min(np.abs(dphi), axis=1)
                    
                    # Handle None values (events with no good jets) by filling with a sentinel value
                    deltaphimin = ak.fill_none(deltaphimin, -9999.0)
                    return np.array(deltaphimin)
                
                def _calc_deltaphimin_arr_gen(met_phi):
                    # For gen jets, use first 2 gen jets (no "good" jets filter for gen)
                    gen_jet_phis = arrs["GenFatJet_phi"][:, :2]
                    
                    # Broadcast met_phi to same shape
                    met_phi_broadcast = ak.broadcast_arrays(met_phi, gen_jet_phis)[0]
                    
                    # Compute delta_phi
                    dphi = np.arctan2(np.sin(gen_jet_phis - met_phi_broadcast), 
                                     np.cos(gen_jet_phis - met_phi_broadcast))
                    
                    # Take absolute value and minimum across jets (axis=1)
                    deltaphimin = ak.min(np.abs(dphi), axis=1)
                    
                    # Handle None values by filling with a sentinel value
                    deltaphimin = ak.fill_none(deltaphimin, -9999.0)
                    return np.array(deltaphimin)
                
                # Filter to only use events with at least 2 good jets (should already be the case in skimmed files)
                good_jet_mask = arrs["FatJet_isGood"]
                n_good_jets = ak.sum(good_jet_mask, axis=1)
                valid_events = n_good_jets >= 2
                
                return {
                    "initial_events": initial_events,
                    "xsec":      xsec,
                    "pt_corr":   ak.flatten(arrs["FatJet_pt"][valid_events][:, :2]),
                    "pt_uncorr": ak.flatten(arrs["FatJet_pt_uncorr"][valid_events][:, :2]),
                    "pt_gen":    ak.flatten(arrs["GenFatJet_pt"][valid_events][:, :2]),
                    "met_uncorr": met_uncorr[valid_events],
                    "met_official_only": met_official_only[valid_events],
                    "met_all_corr": met_all_corr[valid_events],
                    "met_gen":   arrs["genMET_pt"][valid_events],
                    "met_phi_uncorr": met_phi_uncorr[valid_events],
                    "met_phi_official_only": met_phi_official_only[valid_events],
                    "met_phi_all_corr": met_phi_all_corr[valid_events],
                    "met_phi_gen": arrs["genMET_phi"][valid_events],
                    "mt_uncorr":        _calc_mt_arr(met_uncorr[valid_events],        met_phi_uncorr[valid_events]),
                    "mt_official_only": _calc_mt_arr(met_official_only[valid_events], met_phi_official_only[valid_events]),
                    "mt_all_corr":      _calc_mt_arr(met_all_corr[valid_events],      met_phi_all_corr[valid_events]),
                    "mt_gen":           _calc_mt_arr(arrs["genMET_pt"][valid_events],  arrs["genMET_phi"][valid_events]),
                    "rt_uncorr":        np.array(met_uncorr[valid_events] / _calc_mt_arr(met_uncorr[valid_events], met_phi_uncorr[valid_events])),
                    "rt_official_only": np.array(met_official_only[valid_events] / _calc_mt_arr(met_official_only[valid_events], met_phi_official_only[valid_events])),
                    "rt_all_corr":      np.array(met_all_corr[valid_events] / _calc_mt_arr(met_all_corr[valid_events], met_phi_all_corr[valid_events])),
                    "rt_gen":           np.array(arrs["genMET_pt"][valid_events] / _calc_mt_arr(arrs["genMET_pt"][valid_events], arrs["genMET_phi"][valid_events])),
                    "deltaphimin_uncorr":        _calc_deltaphimin_arr_reco(met_phi_uncorr[valid_events], good_jet_mask[valid_events]),
                    "deltaphimin_official_only": _calc_deltaphimin_arr_reco(met_phi_official_only[valid_events], good_jet_mask[valid_events]),
                    "deltaphimin_all_corr":      _calc_deltaphimin_arr_reco(met_phi_all_corr[valid_events], good_jet_mask[valid_events]),
                    "deltaphimin_gen":           _calc_deltaphimin_arr_gen(arrs["genMET_phi"][valid_events]),
                    "_mt_check": (
                        _calc_mt_arr(met_all_corr[valid_events], met_phi_all_corr[valid_events]),
                        np.array(arrs["MT01FatJetMET"][valid_events]) if "MT01FatJetMET" in arrs.fields else None,
                    ),
                    "_rt_check": (
                        np.array(met_all_corr[valid_events] / _calc_mt_arr(met_all_corr[valid_events], met_phi_all_corr[valid_events])),
                        np.array(arrs["RTFatJet"][valid_events]) if "RTFatJet" in arrs.fields else None,
                    ),
                    "_deltaphimin_check": (
                        _calc_deltaphimin_arr_reco(met_phi_all_corr[valid_events], good_jet_mask[valid_events]),
                        np.array(arrs["DeltaPhiMinFatJetMET"][valid_events]) if "DeltaPhiMinFatJetMET" in arrs.fields else None,
                    ),
                }
            except Exception as e:
                tqdm.write(f"    Warning: Events unreadable in {name}: {e}")
                return None
    except Exception as e:
        tqdm.write(f"    Error opening {name}: {e}")
        return None


def read_dataset(dataset_name, dataset_path):
    """
    Read a dataset and return weighted histograms.

    Returns:
        dict with keys 'pt_corr', 'pt_uncorr', 'weight_sum'
    """
    print(f"\nProcessing {dataset_name}...")

    files = get_file_list(dataset_path)
    if not files:
        print(f"  No files found for {dataset_name}")
        return None

    print(f"  Found {len(files)} files")

    # Read all files in parallel (pure I/O-bound — threads are ideal)
    raw = [None] * len(files)
    errors = []
    with ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
        future_to_idx = {pool.submit(_read_single_file, fp): i
                         for i, fp in enumerate(files)}
        with tqdm(total=len(files), desc=f"  {dataset_name}", unit="file",
                  dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                raw[idx] = future.result()
                if raw[idx] is None:
                    errors.append(Path(files[idx]).name)
                pbar.update(1)
    if errors:
        print(f"  WARNING: {len(errors)} file(s) failed: {', '.join(errors[:5])}"
              + (" ..." if len(errors) > 5 else ""))

    # --- Safety check: recalculated MT(all_corr) vs saved MT01FatJetMET ---
    # Catches formula differences, wrong jet index selection, or skimmer logic changes.
    mt_diffs = []
    n_checked = 0
    for r in raw:
        if r is None or "_mt_check" not in r:
            continue
        mt_calc, mt_saved = r["_mt_check"]
        if mt_saved is None:
            continue
        n_checked += len(mt_calc)
        mt_diffs.append(np.abs(mt_calc - mt_saved[:len(mt_calc)]))
    if mt_diffs:
        all_diffs = np.concatenate(mt_diffs)
        max_diff  = np.max(all_diffs)
        mean_diff = np.mean(all_diffs)
        frac_large = np.mean(all_diffs > 1.0)  # fraction with >1 GeV discrepancy
        if max_diff < 1.0:
            print(f"  [CHECK] MT recalc == MT01FatJetMET ✓  "
                  f"(max |ΔMT|={max_diff:.3f} GeV, mean={mean_diff:.3f} GeV, n={n_checked})")
        else:
            print(f"  [CHECK] WARNING: MT recalc != MT01FatJetMET!  "
                  f"max |ΔMT|={max_diff:.1f} GeV, mean={mean_diff:.2f} GeV, "
                  f">1 GeV fraction={frac_large:.1%}, n={n_checked}")
            print(f"          => formula or jet selection mismatch between diagnostics and skimmer")
    else:
        print(f"  [CHECK] MT01FatJetMET branch not found — cannot validate MT recalculation")
    # -------------------------------------------------------------------

    # --- Safety check: recalculated RT(all_corr) vs saved RTFatJet ---
    rt_diffs = []
    n_checked_rt = 0
    for r in raw:
        if r is None or "_rt_check" not in r:
            continue
        rt_calc, rt_saved = r["_rt_check"]
        if rt_saved is None:
            continue
        n_checked_rt += len(rt_calc)
        rt_diffs.append(np.abs(rt_calc - rt_saved[:len(rt_calc)]))
    if rt_diffs:
        all_diffs = np.concatenate(rt_diffs)
        max_diff  = np.max(all_diffs)
        mean_diff = np.mean(all_diffs)
        frac_large = np.mean(all_diffs > 0.01)  # fraction with >0.01 discrepancy
        if max_diff < 0.01:
            print(f"  [CHECK] RT recalc == RTFatJet ✓  "
                  f"(max |ΔRT|={max_diff:.4f}, mean={mean_diff:.4f}, n={n_checked_rt})")
        else:
            print(f"  [CHECK] WARNING: RT recalc != RTFatJet!  "
                  f"max |ΔRT|={max_diff:.3f}, mean={mean_diff:.4f}, "
                  f">0.01 fraction={frac_large:.1%}, n={n_checked_rt}")
            print(f"          => formula or jet selection mismatch between diagnostics and skimmer")
    else:
        print(f"  [CHECK] RTFatJet branch not found — cannot validate RT recalculation")
    # -------------------------------------------------------------------

    # --- Safety check: recalculated deltaphimin(all_corr) vs saved DeltaPhiMinFatJetMET ---
    deltaphimin_diffs = []
    n_checked_dphi = 0
    for r in raw:
        if r is None or "_deltaphimin_check" not in r:
            continue
        dphi_calc, dphi_saved = r["_deltaphimin_check"]
        if dphi_saved is None:
            continue
        n_checked_dphi += len(dphi_calc)
        deltaphimin_diffs.append(np.abs(dphi_calc - dphi_saved[:len(dphi_calc)]))
    if deltaphimin_diffs:
        all_diffs = np.concatenate(deltaphimin_diffs)
        max_diff  = np.max(all_diffs)
        mean_diff = np.mean(all_diffs)
        frac_large = np.mean(all_diffs > 0.01)  # fraction with >0.01 rad discrepancy
        if max_diff < 0.01:
            print(f"  [CHECK] DeltaPhiMin recalc == DeltaPhiMinFatJetMET ✓  "
                  f"(max |Δφ|={max_diff:.4f} rad, mean={mean_diff:.4f} rad, n={n_checked_dphi})")
        else:
            print(f"  [CHECK] WARNING: DeltaPhiMin recalc != DeltaPhiMinFatJetMET!  "
                  f"max |Δφ|={max_diff:.3f} rad, mean={mean_diff:.4f} rad, "
                  f">0.01 rad fraction={frac_large:.1%}, n={n_checked_dphi}")
            print(f"          => formula or jet selection mismatch between diagnostics and skimmer")
    else:
        print(f"  [CHECK] DeltaPhiMinFatJetMET branch not found — cannot validate deltaphimin recalculation")
    # -------------------------------------------------------------------

    pt_corr_all = []; pt_uncorr_all = []; pt_gen_all = []
    met_uncorr_all = []; met_official_only_all = []; met_all_corr_all = []; met_gen_all = []
    met_phi_uncorr_all = []; met_phi_official_only_all = []; met_phi_all_corr_all = []; met_phi_gen_all = []
    mt_uncorr_all = []; mt_official_only_all = []; mt_all_corr_all = []; mt_gen_all = []
    rt_uncorr_all = []; rt_official_only_all = []; rt_all_corr_all = []; rt_gen_all = []
    deltaphimin_uncorr_all = []; deltaphimin_official_only_all = []; deltaphimin_all_corr_all = []; deltaphimin_gen_all = []
    total_initial_events = 0
    total_xsec = None

    for r in raw:
        if r is None:
            continue
        total_initial_events += r["initial_events"]
        if total_xsec is None and r["xsec"] is not None:
            total_xsec = r["xsec"]
        pt_corr_all.append(r["pt_corr"])
        pt_uncorr_all.append(r["pt_uncorr"])
        pt_gen_all.append(r["pt_gen"])
        met_uncorr_all.append(r["met_uncorr"])
        met_official_only_all.append(r["met_official_only"])
        met_all_corr_all.append(r["met_all_corr"])
        met_gen_all.append(r["met_gen"])
        met_phi_uncorr_all.append(r["met_phi_uncorr"])
        met_phi_official_only_all.append(r["met_phi_official_only"])
        met_phi_all_corr_all.append(r["met_phi_all_corr"])
        met_phi_gen_all.append(r["met_phi_gen"])
        mt_uncorr_all.append(r["mt_uncorr"])
        mt_official_only_all.append(r["mt_official_only"])
        mt_all_corr_all.append(r["mt_all_corr"])
        mt_gen_all.append(r["mt_gen"])
        rt_uncorr_all.append(r["rt_uncorr"])
        rt_official_only_all.append(r["rt_official_only"])
        rt_all_corr_all.append(r["rt_all_corr"])
        rt_gen_all.append(r["rt_gen"])
        deltaphimin_uncorr_all.append(r["deltaphimin_uncorr"])
        deltaphimin_official_only_all.append(r["deltaphimin_official_only"])
        deltaphimin_all_corr_all.append(r["deltaphimin_all_corr"])
        deltaphimin_gen_all.append(r["deltaphimin_gen"])

    if not pt_corr_all:
        print(f"  No data read for {dataset_name}")
        return None

    pt_corr_concat   = ak.concatenate(pt_corr_all)
    pt_uncorr_concat = ak.concatenate(pt_uncorr_all)
    pt_gen_concat    = ak.concatenate(pt_gen_all)
    met_uncorr_concat = ak.concatenate(met_uncorr_all)
    met_official_only_concat = ak.concatenate(met_official_only_all)
    met_all_corr_concat = ak.concatenate(met_all_corr_all)
    met_gen_concat   = ak.concatenate(met_gen_all)
    met_phi_uncorr_concat = ak.concatenate(met_phi_uncorr_all)
    met_phi_official_only_concat = ak.concatenate(met_phi_official_only_all)
    met_phi_all_corr_concat = ak.concatenate(met_phi_all_corr_all)
    met_phi_gen_concat = ak.concatenate(met_phi_gen_all)
    mt_uncorr_concat        = np.concatenate(mt_uncorr_all)
    mt_official_only_concat = np.concatenate(mt_official_only_all)
    mt_all_corr_concat      = np.concatenate(mt_all_corr_all)
    mt_gen_concat           = np.concatenate(mt_gen_all)
    rt_uncorr_concat        = np.concatenate(rt_uncorr_all)
    rt_official_only_concat = np.concatenate(rt_official_only_all)
    rt_all_corr_concat      = np.concatenate(rt_all_corr_all)
    rt_gen_concat           = np.concatenate(rt_gen_all)
    deltaphimin_uncorr_concat        = np.concatenate(deltaphimin_uncorr_all)
    deltaphimin_official_only_concat = np.concatenate(deltaphimin_official_only_all)
    deltaphimin_all_corr_concat      = np.concatenate(deltaphimin_all_corr_all)
    deltaphimin_gen_concat           = np.concatenate(deltaphimin_gen_all)

    if total_xsec is not None and total_initial_events > 0:
        weight_per_event = total_xsec / total_initial_events
        print(f"  {dataset_name}: xsec={total_xsec:.3g} pb, "
              f"N_init={total_initial_events}, w={weight_per_event:.3e}, "
              f"files={sum(r is not None for r in raw)}/{len(raw)}")
    else:
        weight_per_event = 1.0
        print(f"  {dataset_name}: unit weight (no xsec found)")

    weights_reco = np.full(len(pt_corr_concat),  weight_per_event)
    weights_gen  = np.full(len(pt_gen_concat),   weight_per_event)
    weights_met  = np.full(len(met_all_corr_concat), weight_per_event)

    return {
        "pt_corr":   np.array(pt_corr_concat),
        "pt_uncorr": np.array(pt_uncorr_concat),
        "pt_gen":    np.array(pt_gen_concat),
        "met_uncorr": np.array(met_uncorr_concat),
        "met_official_only": np.array(met_official_only_concat),
        "met_all_corr": np.array(met_all_corr_concat),
        "met_gen":    np.array(met_gen_concat),
        "met_phi_uncorr": np.array(met_phi_uncorr_concat),
        "met_phi_official_only": np.array(met_phi_official_only_concat),
        "met_phi_all_corr": np.array(met_phi_all_corr_concat),
        "met_phi_gen": np.array(met_phi_gen_concat),
        "mt_uncorr":        mt_uncorr_concat,
        "mt_official_only": mt_official_only_concat,
        "mt_all_corr":      mt_all_corr_concat,
        "mt_gen":           mt_gen_concat,
        "rt_uncorr":        rt_uncorr_concat,
        "rt_official_only": rt_official_only_concat,
        "rt_all_corr":      rt_all_corr_concat,
        "rt_gen":           rt_gen_concat,
        "deltaphimin_uncorr":        deltaphimin_uncorr_concat,
        "deltaphimin_official_only": deltaphimin_official_only_concat,
        "deltaphimin_all_corr":      deltaphimin_all_corr_concat,
        "deltaphimin_gen":           deltaphimin_gen_concat,
        "weights_reco": weights_reco,
        "weights_gen":  weights_gen,
        "weights_met":  weights_met,
        "n_jets_reco":     len(pt_corr_concat),
        "n_jets_gen":      len(pt_gen_concat),
        "n_events":        len(met_all_corr_concat),
        "weight_sum_reco": np.sum(weights_reco),
        "weight_sum_gen":  np.sum(weights_gen),
        "weight_sum_met":  np.sum(weights_met),
        "name":            dataset_name,
        "weight_per_event": weight_per_event,
        "n_files_ok":      sum(r is not None for r in raw),
        "n_files_total":   len(files),
    }


# ---------------------------------------------------------------------------
# Sample group definitions
# ---------------------------------------------------------------------------
SAMPLE_GROUPS = {
    "QCD":    {"color": "#3b5998", "legend_title": "QCD"},
    "TTJets": {"color": "#ff7f0e", "legend_title": "TTJets"},
    "WJets":  {"color": "#9467bd", "legend_title": "W+Jets"},
    "ZJets":  {"color": "#8c564b", "legend_title": "Z+Jets"},
    "Signal": {"color": "#d62728",
               "legend_title": (r"SVJ $m_{Z'}$ = 1.1 TeV" + "\n" +
                                r"$m_{\mathrm{Dark}}$ = 20 GeV, $r_{\mathrm{inv}}$ = 0.3")},
}


def _categorize_data(data_dict):
    """Categorize datasets into QCD / TTJets / WJets / ZJets / Signal groups."""
    categories = {key: [] for key in SAMPLE_GROUPS}
    for name, data in data_dict.items():
        if data is None:
            continue
        if "Signal" in name or "mMed" in name:
            categories["Signal"].append(data)
        elif name.startswith("TTJets"):
            categories["TTJets"].append(data)
        elif name.startswith("WJets"):
            categories["WJets"].append(data)
        elif name.startswith("ZJets"):
            categories["ZJets"].append(data)
        else:
            categories["QCD"].append(data)
    return categories


def _plot_per_sample_contributions(samples, group_name, output_prefix):
    """
    Diagnostic plot: corrected FatJet pT for each sample in the group as a
    separate line.  Reveals missing bins, wrong cross-section weights, or
    stitching discontinuities.
    """
    bins = np.linspace(0, 2000, 51)
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(13, 6))

    total = None
    for i, data in enumerate(samples):
        label = data.get("name", f"sample {i}")
        n_ok  = data.get("n_files_ok",    "?")
        n_tot = data.get("n_files_total", "?")
        w_ev  = data.get("weight_per_event", np.nan)
        h, _  = np.histogram(data["pt_corr"], bins=bins, weights=data["weights_reco"])
        ax.hist(bins[:-1], bins=bins, weights=h, histtype="step", lw=1.8,
                color=cmap(i % 10),
                label=f"{label}  [files {n_ok}/{n_tot}, w={w_ev:.2e}]")
        total = h if total is None else total + h

    if total is not None:
        ax.hist(bins[:-1], bins=bins, weights=total, histtype="step", lw=2.5,
                color="black", ls="--", label="Total")

    ax.set_yscale("log")
    ax.set_xlabel(r"FatJet $p_T$ (corrected) [GeV]", fontsize=13)
    ax.set_ylabel("Weighted entries", fontsize=13)
    ax.set_title(f"{group_name}: per-sample contributions (JEC diagnostic)", fontsize=14)
    ax.legend(fontsize=8, frameon=True, loc="upper right")
    ax.grid(True, alpha=0.2, which="both")

    # Print a summary table to stdout
    print(f"\n{'='*70}")
    print(f"Per-sample weight summary for {group_name}:")
    print(f"{'Dataset':<45} {'files ok/tot':>12} {'w/event':>12} {'weight sum':>12}")
    print("-" * 83)
    for data in samples:
        print(f"{data.get('name','?'):<45}"
              f"{str(data.get('n_files_ok','?'))+'/'+str(data.get('n_files_total','?')):>12}"
              f"{data.get('weight_per_event', float('nan')):>12.3e}"
              f"{data['weight_sum_reco']:>12.3e}")
    print("=" * 70)

    plt.tight_layout()
    suffix = group_name.lower()
    plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}_{suffix}.pdf", bbox_inches="tight")
    print(f"Saved per-sample contributions to {output_prefix}_{suffix}.png/.pdf")
    plt.close()


def _plot_fatjet_pt_group(samples, group_name, group_info, output_prefix):
    """Plot FatJet pT variants (uncorr / all-corr / gen) for one sample group."""
    bins = np.linspace(0, 2000, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    gen_color = "#2ca02c"
    color_uncorr = "#d62728"
    color_allcorr = "#1f77b4"
    legend_title = group_info["legend_title"]

    total_corr = total_uncorr = total_gen = None
    corr_w2 = uncorr_w2 = gen_w2 = None

    for data in samples:
        h_c,  _ = np.histogram(data["pt_corr"],   bins=bins, weights=data["weights_reco"])
        h_u,  _ = np.histogram(data["pt_uncorr"], bins=bins, weights=data["weights_reco"])
        h_g,  _ = np.histogram(data["pt_gen"],    bins=bins, weights=data["weights_gen"])
        h_c2, _ = np.histogram(data["pt_corr"],   bins=bins, weights=data["weights_reco"] ** 2)
        h_u2, _ = np.histogram(data["pt_uncorr"], bins=bins, weights=data["weights_reco"] ** 2)
        h_g2, _ = np.histogram(data["pt_gen"],    bins=bins, weights=data["weights_gen"]  ** 2)
        if total_corr is None:
            total_corr, total_uncorr, total_gen = h_c, h_u, h_g
            corr_w2, uncorr_w2, gen_w2 = h_c2, h_u2, h_g2
        else:
            total_corr += h_c; total_uncorr += h_u; total_gen += h_g
            corr_w2 += h_c2; uncorr_w2 += h_u2; gen_w2 += h_g2

    corr_err = np.sqrt(corr_w2)
    uncorr_err = np.sqrt(uncorr_w2)
    gen_err = np.sqrt(gen_w2)

    ratio_c = np.divide(total_corr,   total_gen, where=total_gen > 0, out=np.ones_like(total_corr))
    ratio_u = np.divide(total_uncorr, total_gen, where=total_gen > 0, out=np.ones_like(total_uncorr))
    ratio_c_err = ratio_c * np.sqrt(
        np.divide(corr_err**2,  total_corr**2,   where=total_corr > 0,   out=np.zeros_like(total_corr)) +
        np.divide(gen_err**2,   total_gen**2,    where=total_gen > 0,    out=np.zeros_like(total_gen)))
    ratio_u_err = ratio_u * np.sqrt(
        np.divide(uncorr_err**2, total_uncorr**2, where=total_uncorr > 0, out=np.zeros_like(total_uncorr)) +
        np.divide(gen_err**2,    total_gen**2,    where=total_gen > 0,    out=np.zeros_like(total_gen)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})
    ax_main, ax_ratio = axes

    ax_main.hist(bins[:-1], bins=bins, weights=total_corr,   histtype="step", lw=2, color=color_allcorr, ls="-",  label="All corrections", alpha=0.8)
    ax_main.hist(bins[:-1], bins=bins, weights=total_uncorr, histtype="step", lw=2, color=color_uncorr, ls="--", label="Uncorrected", alpha=0.8)
    ax_main.hist(bins[:-1], bins=bins, weights=total_gen,    histtype="step", lw=2, color=gen_color, ls="-",  label="Gen")
    ax_main.errorbar(bin_centers, total_corr,   yerr=corr_err,   fmt="none", ecolor=color_allcorr, elinewidth=2, alpha=0.6)
    ax_main.errorbar(bin_centers, total_uncorr, yerr=uncorr_err, fmt="none", ecolor=color_uncorr, elinewidth=2, alpha=0.6)
    ax_main.errorbar(bin_centers, total_gen,    yerr=gen_err,    fmt="none", ecolor=gen_color, elinewidth=2, alpha=0.6)

    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_c, histtype="step", lw=2, color=color_allcorr, ls="-",  label="All-corr/Gen")
    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_u, histtype="step", lw=2, color=color_uncorr, ls="--", label="Uncorr/Gen")
    ax_ratio.errorbar(bin_centers, ratio_c, yerr=ratio_c_err, fmt="none", ecolor=color_allcorr, elinewidth=2, alpha=0.6)
    ax_ratio.errorbar(bin_centers, ratio_u, yerr=ratio_u_err, fmt="none", ecolor=color_uncorr, elinewidth=2, alpha=0.6)

    ax_main.set_ylabel("Arbitrary Units", ha="right", y=1.0, fontsize=14)
    ax_main.set_yscale("log")
    ax_main.axvline(150, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax_main.set_title(r"$\mathit{Private\ Work}$ $\mathbf{(CMS\ Simulation)}$",
                      loc="left", fontsize=20, pad=0, x=0.0)
    leg = ax_main.legend(loc="upper right", frameon=True, fontsize=10, title=legend_title)
    leg.get_title().set_fontsize(11); leg.get_title().set_fontweight("bold")
    ax_main.grid(True, alpha=0.2, which="both")
    ax_main.tick_params(axis="y", which="both", labelbottom=False, labelsize=12)
    yticks = ax_main.get_yticks(); ax_main.set_yticks(yticks[1:])

    ax_ratio.set_xlabel(r"FatJet $p_T$ [GeV]", ha="right", x=1.0, fontsize=14)
    ax_ratio.set_ylabel("Reco / Gen", ha="right", y=1.0, fontsize=14)
    ax_ratio.set_ylim(0.3, 2.0)
    ax_ratio.axhline(1.0, color="gray", ls="--", lw=1)
    ax_ratio.axvline(150, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax_ratio.legend(loc="upper right", frameon=True, fontsize=9)
    ax_ratio.grid(True, alpha=0.2, which="both")
    ax_ratio.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    suffix = group_name.lower()
    plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}_{suffix}.pdf", bbox_inches="tight")
    print(f"Saved {group_name} FatJet pT plots to {output_prefix}_{suffix}.png/.pdf")
    plt.close()


def _plot_met_group(samples, group_name, group_info, output_prefix):
    """Plot MET variants (uncorr / official-only / all-corr / gen) for one sample group."""
    bins = np.linspace(0, 1000, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    gen_color = "#2ca02c"
    legend_title = group_info["legend_title"]

    color_uncorr = "#d62728"
    color_official = "#ff7f0e"
    color_allcorr = "#1f77b4"

    total_uncorr = total_official = total_allcorr = total_gen = None
    uncorr_w2 = official_w2 = allcorr_w2 = gen_w2 = None

    for data in samples:
        h_u,  _ = np.histogram(data["met_uncorr"], bins=bins, weights=data["weights_met"])
        h_o,  _ = np.histogram(data["met_official_only"], bins=bins, weights=data["weights_met"])
        h_a,  _ = np.histogram(data["met_all_corr"], bins=bins, weights=data["weights_met"])
        h_g,  _ = np.histogram(data["met_gen"],    bins=bins, weights=data["weights_met"])
        h_u2, _ = np.histogram(data["met_uncorr"], bins=bins, weights=data["weights_met"] ** 2)
        h_o2, _ = np.histogram(data["met_official_only"], bins=bins, weights=data["weights_met"] ** 2)
        h_a2, _ = np.histogram(data["met_all_corr"], bins=bins, weights=data["weights_met"] ** 2)
        h_g2, _ = np.histogram(data["met_gen"],    bins=bins, weights=data["weights_met"] ** 2)
        if total_allcorr is None:
            total_uncorr, total_official, total_allcorr, total_gen = h_u, h_o, h_a, h_g
            uncorr_w2, official_w2, allcorr_w2, gen_w2 = h_u2, h_o2, h_a2, h_g2
        else:
            total_uncorr += h_u; total_official += h_o; total_allcorr += h_a; total_gen += h_g
            uncorr_w2 += h_u2; official_w2 += h_o2; allcorr_w2 += h_a2; gen_w2 += h_g2

    uncorr_err = np.sqrt(uncorr_w2)
    official_err = np.sqrt(official_w2)
    allcorr_err = np.sqrt(allcorr_w2)
    gen_err = np.sqrt(gen_w2)

    ratio_o = np.divide(total_official, total_gen, where=total_gen > 0, out=np.ones_like(total_official))
    ratio_a = np.divide(total_allcorr, total_gen, where=total_gen > 0, out=np.ones_like(total_allcorr))
    ratio_u = np.divide(total_uncorr, total_gen, where=total_gen > 0, out=np.ones_like(total_uncorr))
    ratio_o_err = ratio_o * np.sqrt(
        np.divide(official_err**2, total_official**2, where=total_official > 0, out=np.zeros_like(total_official)) +
        np.divide(gen_err**2, total_gen**2, where=total_gen > 0, out=np.zeros_like(total_gen)))
    ratio_a_err = ratio_a * np.sqrt(
        np.divide(allcorr_err**2, total_allcorr**2, where=total_allcorr > 0, out=np.zeros_like(total_allcorr)) +
        np.divide(gen_err**2, total_gen**2, where=total_gen > 0, out=np.zeros_like(total_gen)))
    ratio_u_err = ratio_u * np.sqrt(
        np.divide(uncorr_err**2, total_uncorr**2, where=total_uncorr > 0, out=np.zeros_like(total_uncorr)) +
        np.divide(gen_err**2,    total_gen**2,    where=total_gen > 0,    out=np.zeros_like(total_gen)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})
    ax_main, ax_ratio = axes

    ax_main.hist(bins[:-1], bins=bins, weights=total_allcorr,  histtype="step", lw=2, color=color_allcorr, ls="-",  label="All corrections", alpha=0.8)
    ax_main.hist(bins[:-1], bins=bins, weights=total_official, histtype="step", lw=2, color=color_official, ls="-.", label="Official only", alpha=0.8)
    ax_main.hist(bins[:-1], bins=bins, weights=total_uncorr,   histtype="step", lw=2, color=color_uncorr, ls="--", label="Uncorrected", alpha=0.8)
    ax_main.hist(bins[:-1], bins=bins, weights=total_gen,    histtype="step", lw=2, color=gen_color, ls="-",  label="Gen MET")
    ax_main.errorbar(bin_centers, total_allcorr,  yerr=allcorr_err,  fmt="none", ecolor=color_allcorr, elinewidth=2, alpha=0.6)
    ax_main.errorbar(bin_centers, total_official, yerr=official_err, fmt="none", ecolor=color_official, elinewidth=2, alpha=0.6)
    ax_main.errorbar(bin_centers, total_uncorr,   yerr=uncorr_err,   fmt="none", ecolor=color_uncorr,   elinewidth=2, alpha=0.6)
    ax_main.errorbar(bin_centers, total_gen,    yerr=gen_err,    fmt="none", ecolor=gen_color, elinewidth=2, alpha=0.6)

    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_a, histtype="step", lw=2, color=color_allcorr, ls="-",  label="All-corr/Gen")
    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_o, histtype="step", lw=2, color=color_official, ls="-.", label="Official-only/Gen")
    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_u, histtype="step", lw=2, color=color_uncorr, ls="--", label="Uncorr/Gen")
    ax_ratio.errorbar(bin_centers, ratio_a, yerr=ratio_a_err, fmt="none", ecolor=color_allcorr, elinewidth=2, alpha=0.6)
    ax_ratio.errorbar(bin_centers, ratio_o, yerr=ratio_o_err, fmt="none", ecolor=color_official, elinewidth=2, alpha=0.6)
    ax_ratio.errorbar(bin_centers, ratio_u, yerr=ratio_u_err, fmt="none", ecolor=color_uncorr, elinewidth=2, alpha=0.6)

    ax_main.set_ylabel("Arbitrary Units", ha="right", y=1.0, fontsize=14)
    ax_main.set_yscale("log")
    ax_main.set_title(r"$\mathit{Private\ Work}$ $\mathbf{(CMS\ Simulation)}$",
                      loc="left", fontsize=20, pad=0, x=0.0)
    leg = ax_main.legend(loc="upper right", frameon=True, fontsize=10, title=legend_title)
    leg.get_title().set_fontsize(11); leg.get_title().set_fontweight("bold")
    ax_main.grid(True, alpha=0.2, which="both")
    ax_main.tick_params(axis="y", which="both", labelbottom=False, labelsize=12)
    yticks = ax_main.get_yticks(); ax_main.set_yticks(yticks[1:])

    ax_ratio.set_xlabel(r"$E_T^{\mathrm{miss}}$ [GeV]", ha="right", x=1.0, fontsize=14)
    ax_ratio.set_ylabel("Reco / Gen", ha="right", y=1.0, fontsize=14)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.axhline(1.0, color="gray", ls="--", lw=1)
    ax_ratio.legend(loc="upper right", frameon=True, fontsize=9)
    ax_ratio.grid(True, alpha=0.2, which="both")
    ax_ratio.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    suffix = group_name.lower()
    plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}_{suffix}.pdf", bbox_inches="tight")
    print(f"Saved {group_name} MET plots to {output_prefix}_{suffix}.png/.pdf")
    plt.close()


def _plot_met_closure_group(samples, group_name, group_info, output_prefix):
    """Plot MET closure: ScoutMET - GenMET for all MET variants in one group."""
    bins = np.linspace(-500, 500, 81)
    color_uncorr = "#d62728"
    color_official = "#ff7f0e"
    color_allcorr = "#1f77b4"

    all_uncorr = []
    all_official = []
    all_allcorr = []
    all_weights = []

    for data in samples:
        d_uncorr = data["met_uncorr"] - data["met_gen"]
        d_official = data["met_official_only"] - data["met_gen"]
        d_allcorr = data["met_all_corr"] - data["met_gen"]
        all_uncorr.append(d_uncorr)
        all_official.append(d_official)
        all_allcorr.append(d_allcorr)
        all_weights.append(data["weights_met"])

    d_uncorr = np.concatenate(all_uncorr)
    d_official = np.concatenate(all_official)
    d_allcorr = np.concatenate(all_allcorr)
    weights = np.concatenate(all_weights)

    # Weighted means and RMS are key MET-closure diagnostics.
    mean_uncorr = np.average(d_uncorr, weights=weights)
    mean_official = np.average(d_official, weights=weights)
    mean_allcorr = np.average(d_allcorr, weights=weights)
    rms_uncorr = np.sqrt(np.average((d_uncorr - mean_uncorr) ** 2, weights=weights))
    rms_official = np.sqrt(np.average((d_official - mean_official) ** 2, weights=weights))
    rms_allcorr = np.sqrt(np.average((d_allcorr - mean_allcorr) ** 2, weights=weights))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(d_allcorr, bins=bins, weights=weights, histtype="step", lw=2.2,
            color=color_allcorr,
            label=f"All corr: $\\mu$={mean_allcorr:.1f} GeV, $\\sigma$={rms_allcorr:.1f} GeV")
    ax.hist(d_official, bins=bins, weights=weights, histtype="step", lw=2.2,
            color=color_official, ls="-.",
            label=f"Official only: $\\mu$={mean_official:.1f} GeV, $\\sigma$={rms_official:.1f} GeV")
    ax.hist(d_uncorr, bins=bins, weights=weights, histtype="step", lw=2.2,
            color=color_uncorr, ls="--",
            label=f"Uncorrected: $\\mu$={mean_uncorr:.1f} GeV, $\\sigma$={rms_uncorr:.1f} GeV")

    ax.axvline(0.0, color="gray", ls="--", lw=1.5, alpha=0.8, label="Perfect closure")
    ax.set_xlabel(r"$E_T^{\mathrm{miss}}(\mathrm{Scout}) - E_T^{\mathrm{miss}}(\mathrm{Gen})$ [GeV]",
                  ha="right", x=1.0, fontsize=14)
    ax.set_ylabel("Weighted entries", ha="right", y=1.0, fontsize=14)
    ax.set_yscale("log")
    ax.set_title(r"$\mathit{Private\ Work}$ $\mathbf{(CMS\ Simulation)}$",
                 loc="left", fontsize=20, pad=0, x=0.0)
    ax.grid(True, alpha=0.25, which="both")
    leg = ax.legend(loc="upper right", frameon=True, fontsize=10, title=group_info["legend_title"])
    leg.get_title().set_fontsize(11)
    leg.get_title().set_fontweight("bold")

    plt.tight_layout()
    suffix = group_name.lower()
    plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}_{suffix}.pdf", bbox_inches="tight")
    print(f"Saved {group_name} MET closure plots to {output_prefix}_{suffix}.png/.pdf")
    plt.close()


def _plot_response_resolution_group(samples, group_name, output_prefix):
    """Plot mean response and resolution vs pT for one sample group."""
    pt_bins = np.array([100, 150, 200, 300, 400, 600, 800, 1200, 2000])
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    color_corr = "#1f77b4"; color_uncorr = "#d62728"

    mean_c = []; mean_u = []; rms_c = []; rms_u = []
    mean_c_err = []; mean_u_err = []; rms_c_err = []; rms_u_err = []

    for i in range(len(pt_bins) - 1):
        pt_min, pt_max = pt_bins[i], pt_bins[i + 1]
        rc_list = []; ru_list = []; w_list = []
        for data in samples:
            n = min(len(data["pt_gen"]), len(data["pt_corr"]))
            pt_gen   = data["pt_gen"][:n]
            pt_corr  = data["pt_corr"][:n]
            pt_uncorr = data["pt_uncorr"][:n]
            weights  = data["weights_reco"][:n]
            mask = (pt_gen >= pt_min) & (pt_gen < pt_max)
            if np.sum(mask) > 0:
                rc_list.extend(pt_corr[mask]  / pt_gen[mask])
                ru_list.extend(pt_uncorr[mask] / pt_gen[mask])
                w_list.extend(weights[mask])
        if rc_list:
            rc = np.array(rc_list); ru = np.array(ru_list); w = np.array(w_list)
            mc = np.average(rc, weights=w); mu = np.average(ru, weights=w)
            sc = np.sqrt(np.average((rc - mc) ** 2, weights=w))
            su = np.sqrt(np.average((ru - mu) ** 2, weights=w))
            n_eff = np.sum(w) ** 2 / np.sum(w ** 2)
            mean_c.append(mc); mean_u.append(mu)
            rms_c.append(sc); rms_u.append(su)
            mean_c_err.append(sc / np.sqrt(n_eff)); mean_u_err.append(su / np.sqrt(n_eff))
            rms_c_err.append(sc / np.sqrt(2 * n_eff)); rms_u_err.append(su / np.sqrt(2 * n_eff))
        else:
            for lst in [mean_c, mean_u, rms_c, rms_u]: lst.append(np.nan)
            for lst in [mean_c_err, mean_u_err, rms_c_err, rms_u_err]: lst.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1, ax2 = axes

    ax1.errorbar(pt_centers, mean_c, yerr=mean_c_err, marker="o", ms=8, lw=2, capsize=4,
                 label="Corrected (JEC+JER)", color=color_corr)
    ax1.errorbar(pt_centers, mean_u, yerr=mean_u_err, marker="s", ms=8, lw=2, capsize=4,
                 label="Uncorrected (raw)", color=color_uncorr)
    ax1.axhline(1.0, color="gray", ls="--", lw=2, alpha=0.5, label="Perfect response")
    ax1.fill_between(pt_centers, 0.98, 1.02, alpha=0.2, color="gray", label="±2% band")
    ax1.set_xlabel(r"Gen Jet $p_T$ [GeV]", fontsize=14)
    ax1.set_ylabel(r"Mean Response $\langle p_T^{\mathrm{reco}} / p_T^{\mathrm{gen}} \rangle$", fontsize=14)
    ax1.set_title(f"JEC Diagnostic: Mean Response vs $p_T$ ({group_name})", fontsize=16, pad=10)
    ax1.set_xscale("log"); ax1.set_ylim(0.85, 1.15)
    ax1.legend(loc="best", fontsize=11, frameon=True); ax1.grid(True, alpha=0.3, which="both")

    ax2.errorbar(pt_centers, rms_c, yerr=rms_c_err, marker="o", ms=8, lw=2, capsize=4,
                 label="Corrected (JEC+JER)", color=color_corr)
    ax2.errorbar(pt_centers, rms_u, yerr=rms_u_err, marker="s", ms=8, lw=2, capsize=4,
                 label="Uncorrected (raw)", color=color_uncorr)
    ax2.set_xlabel(r"Gen Jet $p_T$ [GeV]", fontsize=14)
    ax2.set_ylabel(r"Resolution (RMS of $p_T^{\mathrm{reco}} / p_T^{\mathrm{gen}}$)", fontsize=14)
    ax2.set_title(f"JER Diagnostic: Resolution vs $p_T$ ({group_name})", fontsize=16, pad=10)
    ax2.set_xscale("log")
    valid_rms = [v for v in rms_c + rms_u if not np.isnan(v)]
    ax2.set_ylim(0, max(valid_rms) * 1.3 if valid_rms else 1)
    ax2.legend(loc="best", fontsize=11, frameon=True); ax2.grid(True, alpha=0.3, which="both")
    ax2.text(0.05, 0.95, "JER should INCREASE resolution\n(make distribution wider)",
             transform=ax2.transAxes, fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.tight_layout()
    suffix = group_name.lower()
    plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}_{suffix}.pdf", bbox_inches="tight")
    print(f"Saved {group_name} response/resolution plots to {output_prefix}_{suffix}.png/.pdf")
    plt.close()

    print(f"\n{'='*80}")
    print(f"JEC/JER Performance Summary ({group_name}):")
    print(f"{'='*80}")
    print(f"{'pT Range [GeV]':<20} {'Mean Resp (Corr)':<18} {'Mean Resp (Uncorr)':<18} "
          f"{'RMS (Corr)':<15} {'RMS (Uncorr)':<15}")
    print("-" * 90)
    for i in range(len(pt_bins) - 1):
        if not np.isnan(mean_c[i]):
            print(f"{pt_bins[i]:.0f}-{pt_bins[i+1]:.0f}".ljust(20),
                  f"{mean_c[i]:.4f}±{mean_c_err[i]:.4f}".ljust(18),
                  f"{mean_u[i]:.4f}±{mean_u_err[i]:.4f}".ljust(18),
                  f"{rms_c[i]:.4f}".ljust(15), f"{rms_u[i]:.4f}".ljust(15))
    print("=" * 80 + "\n")


def _plot_response_distribution_group(samples, group_name, output_prefix):
    """Plot response distributions in pT bins for one sample group."""
    all_pt_gen = []; all_rc = []; all_ru = []
    for data in samples:
        n = min(len(data["pt_gen"]), len(data["pt_corr"]))
        pt_gen = data["pt_gen"][:n]
        all_pt_gen.extend(pt_gen)
        all_rc.extend(data["pt_corr"][:n]  / pt_gen)
        all_ru.extend(data["pt_uncorr"][:n] / pt_gen)

    all_pt_gen = np.array(all_pt_gen)
    all_rc = np.array(all_rc); all_ru = np.array(all_ru)

    pt_bins_to_plot = [(200, 300), (400, 600), (800, 1200)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (pt_min, pt_max) in zip(axes, pt_bins_to_plot):
        mask = (all_pt_gen >= pt_min) & (all_pt_gen < pt_max)
        if np.sum(mask) > 10:
            rc_bin = all_rc[mask]; ru_bin = all_ru[mask]
            mc, sc = np.mean(rc_bin), np.std(rc_bin)
            mu, su = np.mean(ru_bin), np.std(ru_bin)
            rbins = np.linspace(0.5, 1.5, 40)
            ax.hist(ru_bin, bins=rbins, histtype="step", lw=2.5, color="#d62728",
                    label=f"Uncorrected\nμ={mu:.3f}, σ={su:.3f}", density=True, alpha=0.8)
            ax.hist(rc_bin, bins=rbins, histtype="step", lw=2.5, color="#1f77b4",
                    label=f"Corrected (JEC+JER)\nμ={mc:.3f}, σ={sc:.3f}", density=True, alpha=0.8)
            ax.axvline(1.0, color="gray", ls="--", lw=2, alpha=0.5)
            ax.set_xlabel(r"Response ($p_T^{\mathrm{reco}} / p_T^{\mathrm{gen}}$)", fontsize=12)
            ax.set_ylabel("Normalized entries", fontsize=12)
            ax.set_title(f"{pt_min}-{pt_max} GeV", fontsize=14, pad=10)
            ax.legend(loc="upper right", fontsize=10, frameon=True)
            ax.grid(True, alpha=0.3); ax.set_ylim(0, None)
            if sc > su * 1.05:
                ax.text(0.05, 0.95, f"JER increases σ\nby {(sc/su-1)*100:.1f}%",
                        transform=ax.transAxes, fontsize=9, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))
            elif sc < su * 0.95:
                ax.text(0.05, 0.95, f"Warning: σ decreased\nby {(1-sc/su)*100:.1f}%",
                        transform=ax.transAxes, fontsize=9, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="orange", alpha=0.5))

    fig.suptitle(f"Response Distributions in $p_T$ Bins ({group_name})", fontsize=16, y=1.02)
    plt.tight_layout()
    suffix = group_name.lower()
    plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}_{suffix}.pdf", bbox_inches="tight")
    print(f"Saved {group_name} response distribution plots to {output_prefix}_{suffix}.png/.pdf")
    plt.close()


def _dphi(phi1, phi2):
    """Signed angular difference in [-pi, pi]."""
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def _plot_met_dphi_group(samples, group_name, group_info, output_prefix):
    """Plot Delta-phi(MET_reco, MET_gen) for all MET variants in one group."""
    bins = np.linspace(-np.pi, np.pi, 64)
    color_uncorr = "#d62728"
    color_official = "#ff7f0e"
    color_allcorr = "#1f77b4"

    all_dphi_uncorr = []; all_dphi_official = []; all_dphi_allcorr = []
    all_weights = []

    for data in samples:
        all_dphi_uncorr.append(_dphi(data["met_phi_uncorr"],       data["met_phi_gen"]))
        all_dphi_official.append(_dphi(data["met_phi_official_only"], data["met_phi_gen"]))
        all_dphi_allcorr.append(_dphi(data["met_phi_all_corr"],    data["met_phi_gen"]))
        all_weights.append(data["weights_met"])

    dphi_uncorr   = np.concatenate(all_dphi_uncorr)
    dphi_official = np.concatenate(all_dphi_official)
    dphi_allcorr  = np.concatenate(all_dphi_allcorr)
    weights       = np.concatenate(all_weights)

    rms_uncorr   = np.sqrt(np.average(dphi_uncorr   ** 2, weights=weights))
    rms_official = np.sqrt(np.average(dphi_official ** 2, weights=weights))
    rms_allcorr  = np.sqrt(np.average(dphi_allcorr  ** 2, weights=weights))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(dphi_allcorr,  bins=bins, weights=weights, histtype="step", lw=2.2,
            color=color_allcorr,
            label=f"All corr: RMS={rms_allcorr:.3f} rad")
    ax.hist(dphi_official, bins=bins, weights=weights, histtype="step", lw=2.2,
            color=color_official, ls="-.",
            label=f"Official only: RMS={rms_official:.3f} rad")
    ax.hist(dphi_uncorr,   bins=bins, weights=weights, histtype="step", lw=2.2,
            color=color_uncorr, ls="--",
            label=f"Uncorrected: RMS={rms_uncorr:.3f} rad")

    ax.axvline(0.0, color="gray", ls="--", lw=1.5, alpha=0.8, label="Perfect closure")
    ax.set_xlabel(r"$\Delta\phi(\vec{E}_T^{\mathrm{miss,Scout}},\, \vec{E}_T^{\mathrm{miss,Gen}})$ [rad]",
                  ha="right", x=1.0, fontsize=14)
    ax.set_ylabel("Weighted entries", ha="right", y=1.0, fontsize=14)
    ax.set_yscale("log")
    ax.set_title(r"$\mathit{Private\ Work}$ $\mathbf{(CMS\ Simulation)}$",
                 loc="left", fontsize=20, pad=0, x=0.0)
    ax.grid(True, alpha=0.25, which="both")
    leg = ax.legend(loc="upper right", frameon=True, fontsize=10,
                    title=group_info["legend_title"])
    leg.get_title().set_fontsize(11)
    leg.get_title().set_fontweight("bold")

    plt.tight_layout()
    suffix = group_name.lower()
    plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}_{suffix}.pdf", bbox_inches="tight")
    print(f"Saved {group_name} MET delta-phi plots to {output_prefix}_{suffix}.png/.pdf")
    plt.close()


def _plot_mt_group(samples, group_name, group_info, output_prefix):
    """Plot MT distributions for different MET variants in one sample group."""
    bins = np.linspace(0, 3000, 61)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    gen_color    = "#2ca02c"
    color_uncorr   = "#d62728"
    color_official = "#ff7f0e"
    color_allcorr  = "#1f77b4"
    legend_title = group_info["legend_title"]

    total = {k: None for k in ("uncorr", "official", "allcorr", "gen")}
    w2    = {k: None for k in total}

    for data in samples:
        w = data["weights_met"]
        for key, arr in [
            ("uncorr",   data["mt_uncorr"]),
            ("official", data["mt_official_only"]),
            ("allcorr",  data["mt_all_corr"]),
            ("gen",      data["mt_gen"]),
        ]:
            # guard for length mismatch (e.g. events with < 2 jets dropped)
            w_ = w[:len(arr)]
            h,  _ = np.histogram(arr, bins=bins, weights=w_)
            h2, _ = np.histogram(arr, bins=bins, weights=w_**2)
            if total[key] is None:
                total[key] = h; w2[key] = h2
            else:
                total[key] += h; w2[key] += h2

    err = {k: np.sqrt(v) for k, v in w2.items()}

    ratio_u   = np.divide(total["uncorr"],   total["gen"], where=total["gen"] > 0, out=np.ones_like(total["uncorr"]))
    ratio_o   = np.divide(total["official"], total["gen"], where=total["gen"] > 0, out=np.ones_like(total["official"]))
    ratio_a   = np.divide(total["allcorr"],  total["gen"], where=total["gen"] > 0, out=np.ones_like(total["allcorr"]))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})
    ax_main, ax_ratio = axes

    ax_main.hist(bins[:-1], bins=bins, weights=total["allcorr"],  histtype="step", lw=2, color=color_allcorr,  ls="-",  label="All corrections",  alpha=0.85)
    ax_main.hist(bins[:-1], bins=bins, weights=total["official"], histtype="step", lw=2, color=color_official, ls="-.", label="Official JEC only",  alpha=0.85)
    ax_main.hist(bins[:-1], bins=bins, weights=total["uncorr"],   histtype="step", lw=2, color=color_uncorr,   ls="--", label="Uncorrected MET",     alpha=0.85)
    ax_main.hist(bins[:-1], bins=bins, weights=total["gen"],      histtype="step", lw=2, color=gen_color,      ls="-",  label="Gen MET")
    for key, color in [("allcorr", color_allcorr),
                       ("official", color_official), ("uncorr", color_uncorr), ("gen", gen_color)]:
        ax_main.errorbar(bin_centers, total[key], yerr=err[key], fmt="none", ecolor=color, elinewidth=1.5, alpha=0.5)

    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_a,   histtype="step", lw=2, color=color_allcorr,  ls="-",  label="All-corr/Gen")
    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_o,   histtype="step", lw=2, color=color_official, ls="-.", label="Official/Gen")
    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_u,   histtype="step", lw=2, color=color_uncorr,   ls="--", label="Uncorr/Gen")

    # MT > 650 GeV cut line
    ax_main.axvline(650, color="gray", ls="--", lw=1.5, alpha=0.7, label="MT > 650 GeV cut")
    ax_ratio.axvline(650, color="gray", ls="--", lw=1.5, alpha=0.7)

    ax_main.set_ylabel("Arbitrary Units", ha="right", y=1.0, fontsize=14)
    ax_main.set_yscale("log")
    ax_main.set_title(r"$\mathit{Private\ Work}$ $\mathbf{(CMS\ Simulation)}$",
                      loc="left", fontsize=20, pad=0, x=0.0)
    leg = ax_main.legend(loc="upper right", frameon=True, fontsize=10, title=legend_title)
    leg.get_title().set_fontsize(11); leg.get_title().set_fontweight("bold")
    ax_main.grid(True, alpha=0.2, which="both")
    yticks = ax_main.get_yticks(); ax_main.set_yticks(yticks[1:])

    ax_ratio.set_xlabel(r"$M_T$ [GeV]", ha="right", x=1.0, fontsize=14)
    ax_ratio.set_ylabel("Reco / Gen", ha="right", y=1.0, fontsize=14)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.axhline(1.0, color="gray", ls="--", lw=1)
    ax_ratio.legend(loc="upper right", frameon=True, fontsize=9)
    ax_ratio.grid(True, alpha=0.2, which="both")
    ax_ratio.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    suffix = group_name.lower()
    plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}_{suffix}.pdf", bbox_inches="tight")
    print(f"Saved {group_name} MT plots to {output_prefix}_{suffix}.png/.pdf")
    plt.close()


def _plot_rt_group(samples, group_name, group_info, output_prefix):
    """Plot RT distributions for different MET variants in one sample group."""
    bins = np.linspace(0, 1.0, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    gen_color    = "#2ca02c"
    color_uncorr   = "#d62728"
    color_official = "#ff7f0e"
    color_allcorr  = "#1f77b4"
    legend_title = group_info["legend_title"]

    total = {k: None for k in ("uncorr", "official", "allcorr", "gen")}
    w2    = {k: None for k in total}

    for data in samples:
        w = data["weights_met"]
        for key, arr in [
            ("uncorr",   data["rt_uncorr"]),
            ("official", data["rt_official_only"]),
            ("allcorr",  data["rt_all_corr"]),
            ("gen",      data["rt_gen"]),
        ]:
            # guard for length mismatch (e.g. events with < 2 jets dropped)
            w_ = w[:len(arr)]
            h,  _ = np.histogram(arr, bins=bins, weights=w_)
            h2, _ = np.histogram(arr, bins=bins, weights=w_**2)
            if total[key] is None:
                total[key] = h; w2[key] = h2
            else:
                total[key] += h; w2[key] += h2

    err = {k: np.sqrt(v) for k, v in w2.items()}

    ratio_u   = np.divide(total["uncorr"],   total["gen"], where=total["gen"] > 0, out=np.ones_like(total["uncorr"]))
    ratio_o   = np.divide(total["official"], total["gen"], where=total["gen"] > 0, out=np.ones_like(total["official"]))
    ratio_a   = np.divide(total["allcorr"],  total["gen"], where=total["gen"] > 0, out=np.ones_like(total["allcorr"]))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})
    ax_main, ax_ratio = axes

    ax_main.hist(bins[:-1], bins=bins, weights=total["allcorr"],  histtype="step", lw=2, color=color_allcorr,  ls="-",  label="All corrections",  alpha=0.85)
    ax_main.hist(bins[:-1], bins=bins, weights=total["official"], histtype="step", lw=2, color=color_official, ls="-.", label="Official JEC only",  alpha=0.85)
    ax_main.hist(bins[:-1], bins=bins, weights=total["uncorr"],   histtype="step", lw=2, color=color_uncorr,   ls="--", label="Uncorrected MET",     alpha=0.85)
    ax_main.hist(bins[:-1], bins=bins, weights=total["gen"],      histtype="step", lw=2, color=gen_color,      ls="-",  label="Gen MET")
    for key, color in [("allcorr", color_allcorr),
                       ("official", color_official), ("uncorr", color_uncorr), ("gen", gen_color)]:
        ax_main.errorbar(bin_centers, total[key], yerr=err[key], fmt="none", ecolor=color, elinewidth=1.5, alpha=0.5)

    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_a,   histtype="step", lw=2, color=color_allcorr,  ls="-",  label="All-corr/Gen")
    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_o,   histtype="step", lw=2, color=color_official, ls="-.", label="Official/Gen")
    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_u,   histtype="step", lw=2, color=color_uncorr,   ls="--", label="Uncorr/Gen")

    # RT > 0.15 cut line
    ax_main.axvline(0.15, color="gray", ls="--", lw=1.5, alpha=0.7, label="RT > 0.15 cut")
    ax_ratio.axvline(0.15, color="gray", ls="--", lw=1.5, alpha=0.7)

    ax_main.set_ylabel("Arbitrary Units", ha="right", y=1.0, fontsize=14)
    ax_main.set_yscale("log")
    ax_main.set_title(r"$\mathit{Private\ Work}$ $\mathbf{(CMS\ Simulation)}$",
                      loc="left", fontsize=20, pad=0, x=0.0)
    leg = ax_main.legend(loc="upper right", frameon=True, fontsize=10, title=legend_title)
    leg.get_title().set_fontsize(11); leg.get_title().set_fontweight("bold")
    ax_main.grid(True, alpha=0.2, which="both")
    yticks = ax_main.get_yticks(); ax_main.set_yticks(yticks[1:])

    ax_ratio.set_xlabel(r"$R_T$ (MET / $M_T$)", ha="right", x=1.0, fontsize=14)
    ax_ratio.set_ylabel("Reco / Gen", ha="right", y=1.0, fontsize=14)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.axhline(1.0, color="gray", ls="--", lw=1)
    ax_ratio.legend(loc="upper right", frameon=True, fontsize=9)
    ax_ratio.grid(True, alpha=0.2, which="both")
    ax_ratio.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    suffix = group_name.lower()
    plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}_{suffix}.pdf", bbox_inches="tight")
    print(f"Saved {group_name} RT plots to {output_prefix}_{suffix}.png/.pdf")
    plt.close()


def _plot_deltaphimin_group(samples, group_name, group_info, output_prefix):
    """Plot deltaphimin distributions for different MET variants in one sample group."""
    bins = np.linspace(0, 1.5, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    gen_color    = "#2ca02c"
    color_uncorr   = "#d62728"
    color_official = "#ff7f0e"
    color_allcorr  = "#1f77b4"
    legend_title = group_info["legend_title"]

    total = {k: None for k in ("uncorr", "official", "allcorr", "gen")}
    w2    = {k: None for k in total}

    for data in samples:
        w = data["weights_met"]
        for key, arr in [
            ("uncorr",   data["deltaphimin_uncorr"]),
            ("official", data["deltaphimin_official_only"]),
            ("allcorr",  data["deltaphimin_all_corr"]),
            ("gen",      data["deltaphimin_gen"]),
        ]:
            # guard for length mismatch (e.g. events with < 2 jets dropped)
            w_ = w[:len(arr)]
            h,  _ = np.histogram(arr, bins=bins, weights=w_)
            h2, _ = np.histogram(arr, bins=bins, weights=w_**2)
            if total[key] is None:
                total[key] = h; w2[key] = h2
            else:
                total[key] += h; w2[key] += h2

    err = {k: np.sqrt(v) for k, v in w2.items()}

    ratio_u   = np.divide(total["uncorr"],   total["gen"], where=total["gen"] > 0, out=np.ones_like(total["uncorr"]))
    ratio_o   = np.divide(total["official"], total["gen"], where=total["gen"] > 0, out=np.ones_like(total["official"]))
    ratio_a   = np.divide(total["allcorr"],  total["gen"], where=total["gen"] > 0, out=np.ones_like(total["allcorr"]))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})
    ax_main, ax_ratio = axes

    ax_main.hist(bins[:-1], bins=bins, weights=total["allcorr"],  histtype="step", lw=2, color=color_allcorr,  ls="-",  label="All corrections",  alpha=0.85)
    ax_main.hist(bins[:-1], bins=bins, weights=total["official"], histtype="step", lw=2, color=color_official, ls="-.", label="Official JEC only",  alpha=0.85)
    ax_main.hist(bins[:-1], bins=bins, weights=total["uncorr"],   histtype="step", lw=2, color=color_uncorr,   ls="--", label="Uncorrected MET",     alpha=0.85)
    ax_main.hist(bins[:-1], bins=bins, weights=total["gen"],      histtype="step", lw=2, color=gen_color,      ls="-",  label="Gen MET")
    for key, color in [("allcorr", color_allcorr),
                       ("official", color_official), ("uncorr", color_uncorr), ("gen", gen_color)]:
        ax_main.errorbar(bin_centers, total[key], yerr=err[key], fmt="none", ecolor=color, elinewidth=1.5, alpha=0.5)

    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_a,   histtype="step", lw=2, color=color_allcorr,  ls="-",  label="All-corr/Gen")
    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_o,   histtype="step", lw=2, color=color_official, ls="-.", label="Official/Gen")
    ax_ratio.hist(bins[:-1], bins=bins, weights=ratio_u,   histtype="step", lw=2, color=color_uncorr,   ls="--", label="Uncorr/Gen")

    # DeltaPhiMin < 0.8 cut line (typical selection)
    ax_main.axvline(0.8, color="gray", ls="--", lw=1.5, alpha=0.7, label=r"$\Delta\phi_{min}$ < 0.8 cut")
    ax_ratio.axvline(0.8, color="gray", ls="--", lw=1.5, alpha=0.7)

    ax_main.set_ylabel("Arbitrary Units", ha="right", y=1.0, fontsize=14)
    ax_main.set_yscale("log")
    ax_main.set_title(r"$\mathit{Private\ Work}$ $\mathbf{(CMS\ Simulation)}$",
                      loc="left", fontsize=20, pad=0, x=0.0)
    leg = ax_main.legend(loc="upper right", frameon=True, fontsize=10, title=legend_title)
    leg.get_title().set_fontsize(11); leg.get_title().set_fontweight("bold")
    ax_main.grid(True, alpha=0.2, which="both")
    yticks = ax_main.get_yticks(); ax_main.set_yticks(yticks[1:])

    ax_ratio.set_xlabel(r"$\Delta\phi_{min}$ (jet, MET) [rad]", ha="right", x=1.0, fontsize=14)
    ax_ratio.set_ylabel("Reco / Gen", ha="right", y=1.0, fontsize=14)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.axhline(1.0, color="gray", ls="--", lw=1)
    ax_ratio.legend(loc="upper right", frameon=True, fontsize=9)
    ax_ratio.grid(True, alpha=0.2, which="both")
    ax_ratio.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    suffix = group_name.lower()
    plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}_{suffix}.pdf", bbox_inches="tight")
    print(f"Saved {group_name} DeltaPhiMin plots to {output_prefix}_{suffix}.png/.pdf")
    plt.close()


def _plot_met_resolution_vs_genmet_group(samples, group_name, group_info, output_prefix):
    """
    Plot MET relative resolution vs gen MET in bins.
    Resolution = RMS of (MET_reco - MET_gen) / MET_gen in each gen-MET bin.
    """
    met_bins = np.array([0, 50, 100, 150, 200, 300, 400, 600, 800, 1000])
    met_centers = (met_bins[:-1] + met_bins[1:]) / 2
    color_uncorr   = "#d62728"
    color_official = "#ff7f0e"
    color_allcorr  = "#1f77b4"

    res_uncorr   = []; res_official = []; res_allcorr = []
    res_uncorr_err = []; res_official_err = []; res_allcorr_err = []
    bias_uncorr  = []; bias_official = []; bias_allcorr = []

    for i in range(len(met_bins) - 1):
        lo, hi = met_bins[i], met_bins[i + 1]
        ru_list = []; ro_list = []; ra_list = []; w_list = []
        for data in samples:
            mask = (data["met_gen"] >= lo) & (data["met_gen"] < hi) & (data["met_gen"] > 0)
            if np.sum(mask) == 0:
                continue
            gen = data["met_gen"][mask]
            ru_list.append((data["met_uncorr"][mask]       - gen) / gen)
            ro_list.append((data["met_official_only"][mask] - gen) / gen)
            ra_list.append((data["met_all_corr"][mask]     - gen) / gen)
            w_list.append(data["weights_met"][mask])

        if not ru_list:
            for lst in [res_uncorr, res_official, res_allcorr,
                        res_uncorr_err, res_official_err, res_allcorr_err,
                        bias_uncorr, bias_official, bias_allcorr]:
                lst.append(np.nan)
            continue

        ru = np.concatenate(ru_list); ro = np.concatenate(ro_list)
        ra = np.concatenate(ra_list); w  = np.concatenate(w_list)
        n_eff = np.sum(w) ** 2 / np.sum(w ** 2)

        mu_u = np.average(ru, weights=w); mu_o = np.average(ro, weights=w)
        mu_a = np.average(ra, weights=w)
        su = np.sqrt(np.average((ru - mu_u) ** 2, weights=w))
        so = np.sqrt(np.average((ro - mu_o) ** 2, weights=w))
        sa = np.sqrt(np.average((ra - mu_a) ** 2, weights=w))

        res_uncorr.append(su);   res_official.append(so);   res_allcorr.append(sa)
        res_uncorr_err.append(su / np.sqrt(2 * n_eff))
        res_official_err.append(so / np.sqrt(2 * n_eff))
        res_allcorr_err.append(sa / np.sqrt(2 * n_eff))
        bias_uncorr.append(mu_u); bias_official.append(mu_o); bias_allcorr.append(mu_a)

    res_uncorr   = np.array(res_uncorr);   res_official = np.array(res_official)
    res_allcorr  = np.array(res_allcorr)
    bias_uncorr  = np.array(bias_uncorr);  bias_official = np.array(bias_official)
    bias_allcorr = np.array(bias_allcorr)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_res, ax_bias = axes

    for ax_res, ax_bias in [axes]:
        ax_res.errorbar(met_centers, res_allcorr,  yerr=res_allcorr_err,
                        marker="o", ms=7, lw=2, capsize=4, color=color_allcorr,
                        label="All corrections")
        ax_res.errorbar(met_centers, res_official, yerr=res_official_err,
                        marker="^", ms=7, lw=2, capsize=4, color=color_official,
                        label="Official only")
        ax_res.errorbar(met_centers, res_uncorr,   yerr=res_uncorr_err,
                        marker="s", ms=7, lw=2, capsize=4, color=color_uncorr,
                        label="Uncorrected")
        ax_res.set_xlabel(r"Gen $E_T^{\mathrm{miss}}$ [GeV]", fontsize=14)
        ax_res.set_ylabel(r"$\sigma[(E_T^{\mathrm{miss,reco}} - E_T^{\mathrm{miss,gen}}) / E_T^{\mathrm{miss,gen}}]$",
                          fontsize=13)
        ax_res.set_title(f"MET Relative Resolution vs Gen MET ({group_name})", fontsize=15, pad=10)
        ax_res.legend(loc="upper right", fontsize=11, frameon=True,
                      title=group_info["legend_title"])
        ax_res.grid(True, alpha=0.3); ax_res.set_ylim(bottom=0)

        ax_bias.errorbar(met_centers, bias_allcorr,  fmt="o-", ms=7, lw=2, capsize=4,
                         color=color_allcorr,  label="All corrections")
        ax_bias.errorbar(met_centers, bias_official, fmt="^-", ms=7, lw=2, capsize=4,
                         color=color_official, label="Official only")
        ax_bias.errorbar(met_centers, bias_uncorr,   fmt="s-", ms=7, lw=2, capsize=4,
                         color=color_uncorr,   label="Uncorrected")
        ax_bias.axhline(0.0, color="gray", ls="--", lw=1.5, alpha=0.8)
        ax_bias.fill_between(met_centers, -0.02, 0.02, alpha=0.15, color="gray",
                             label="±2% band")
        ax_bias.set_xlabel(r"Gen $E_T^{\mathrm{miss}}$ [GeV]", fontsize=14)
        ax_bias.set_ylabel(r"Mean bias $\langle (E_T^{\mathrm{miss,reco}} - E_T^{\mathrm{miss,gen}}) / E_T^{\mathrm{miss,gen}} \rangle$",
                           fontsize=11)
        ax_bias.set_title(f"MET Relative Bias vs Gen MET ({group_name})", fontsize=15, pad=10)
        ax_bias.legend(loc="best", fontsize=11, frameon=True)
        ax_bias.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = group_name.lower()
    plt.savefig(f"{output_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}_{suffix}.pdf", bbox_inches="tight")
    print(f"Saved {group_name} MET resolution vs gen MET to {output_prefix}_{suffix}.png/.pdf")
    plt.close()


# ---------------------------------------------------------------------------
# Public plotting functions (thin wrappers over helpers above)
# ---------------------------------------------------------------------------

def make_comparison_plots(data_dict, output_dir="jec_plots"):
    """Create FatJet pT comparison plots for all sample groups."""
    categories = _categorize_data(data_dict)
    output_prefix = os.path.join(output_dir, "fatjet_pt_comparison")
    for group_name, samples in categories.items():
        if samples:
            _plot_fatjet_pt_group(samples, group_name, SAMPLE_GROUPS[group_name], output_prefix)

def make_met_comparison_plots(data_dict, output_dir="jec_plots"):
    """Create MET comparison plots for all sample groups."""
    categories = _categorize_data(data_dict)
    output_prefix = os.path.join(output_dir, "met_comparison")
    for group_name, samples in categories.items():
        if samples:
            _plot_met_group(samples, group_name, SAMPLE_GROUPS[group_name], output_prefix)

def make_response_and_resolution_plots(data_dict, output_dir="jec_plots"):
    """Create mean response and resolution vs pT plots for all sample groups."""
    categories = _categorize_data(data_dict)
    output_prefix = os.path.join(output_dir, "jet_response_resolution")
    for group_name, samples in categories.items():
        if samples:
            _plot_response_resolution_group(samples, group_name, output_prefix)

def make_response_distribution_plot(data_dict, output_dir="jec_plots"):
    """Create response distribution plots for all sample groups."""
    categories = _categorize_data(data_dict)
    output_prefix = os.path.join(output_dir, "jet_response_distributions")
    for group_name, samples in categories.items():
        if samples:
            _plot_response_distribution_group(samples, group_name, output_prefix)

def main():
    """Main function to process all datasets and create plots."""

    print("="*80)
    print("FatJet pT Comparison: Corrected vs Uncorrected")
    print("="*80)

    # Create timestamped output directory up front
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("jec_plots", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    for group_name, datasets in DATASET_GROUPS.items():
        print(f"\n{'='*80}")
        print(f"Processing group: {group_name} ({len(datasets)} dataset(s))")
        print("="*80)

        # Read all datasets for this group in parallel
        samples = []
        with ThreadPoolExecutor(max_workers=DATASET_WORKERS) as pool:
            future_to_name = {pool.submit(read_dataset, name, path): name
                              for name, path in datasets.items()}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                result = future.result()
                if result is not None:
                    samples.append(result)

        if not samples:
            print(f"  No data for {group_name}, skipping.")
            continue

        # Make all plots for this group
        group_info = SAMPLE_GROUPS[group_name]
        #_plot_per_sample_contributions(samples, group_name, os.path.join(output_dir, "per_sample_contributions"))
        _plot_fatjet_pt_group(samples, group_name, group_info, os.path.join(output_dir, "fatjet_pt_comparison"))
        _plot_met_group(samples, group_name, group_info, os.path.join(output_dir, "met_comparison"))
        #_plot_met_closure_group(samples, group_name, group_info, os.path.join(output_dir, "met_closure"))
        _plot_met_dphi_group(samples, group_name, group_info, os.path.join(output_dir, "met_dphi"))
        #_plot_met_resolution_vs_genmet_group(samples, group_name, group_info, os.path.join(output_dir, "met_resolution_vs_genmet"))
        #_plot_response_resolution_group(samples, group_name, os.path.join(output_dir, "jet_response_resolution"))
        #_plot_response_distribution_group(samples, group_name, os.path.join(output_dir, "jet_response_distributions"))
        _plot_mt_group(samples, group_name, group_info, os.path.join(output_dir, "mt_comparison"))
        _plot_rt_group(samples, group_name, group_info, os.path.join(output_dir, "rt_comparison"))
        _plot_deltaphimin_group(samples, group_name, group_info, os.path.join(output_dir, "deltaphimin_comparison"))

        # Explicitly free the data before loading the next group
        del samples

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
