import os
import awkward
import numpy
import correctionlib as _correctionlib
from copy import copy

# Resolve data/ directory independent of cwd
_DATA_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../data"))

# correctionlib CorrectionSet cache: year_str -> CorrectionSet
_met_xy_cache = {}


def _get_year_str_for_met_xy(year):
    """Map a year value (str or int) to the key used in data/jme/met/ directory names."""
    y = str(year)
    if "APV" in y or "preVFP" in y:
        return "2016APV"
    # strip any trailing letters (e.g. "2016postVFP" -> "2016")
    for base in ("2016", "2017", "2018"):
        if base in y:
            return base
    raise ValueError(f"Unsupported year for MET XY correction: {year}")


def apply_met_xy_corrections(met_pt, met_phi, year, npv, run_arr, is_data):
    """Apply MET phi modulation (XY) corrections using correctionlib.

    Parameters
    ----------
    met_pt, met_phi : awkward arrays (per-event)
        T1-corrected MET before XY correction.
    year : str or int
        Data-taking year.
    npv : awkward array (per-event)
        Number of good primary vertices per event.
    run_arr : awkward array (per-event)
        Run number per event. For MC pass zeros (run argument ignored by MC key).
    is_data : bool
        True for data, False for MC.

    Returns
    -------
    (corrected_pt, corrected_phi) as numpy arrays.
    """
    year_str = _get_year_str_for_met_xy(year)

    if year_str not in _met_xy_cache:
        json_path = os.path.join(_DATA_DIR, "jme", "met", f"{year_str}_UL", "met.json.gz")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"MET XY correction file not found: {json_path}")
        _met_xy_cache[year_str] = _correctionlib.CorrectionSet.from_file(json_path)

    cset = _met_xy_cache[year_str]

    # Clip npv and pt to the valid range of the correction table.
    # These are physical upper bounds (npv>200 is unphysical; pt>1000 is
    # extremely rare and outside the fitted range), so clipping is safe.
    npv_clipped = awkward.to_numpy(awkward.where(npv > 200, 200, npv)).astype(float)
    pt_clipped  = awkward.to_numpy(awkward.where(met_pt > 1000.0, 1000.0, met_pt)).astype(float)
    phi_arr     = awkward.to_numpy(met_phi).astype(float)
    run_np      = awkward.to_numpy(run_arr).astype(float)

    if is_data:
        corrected_pt  = cset["pt_metphicorr_pfmet_data"].evaluate(pt_clipped, phi_arr, npv_clipped, run_np)
        corrected_phi = cset["phi_metphicorr_pfmet_data"].evaluate(pt_clipped, phi_arr, npv_clipped, run_np)
    else:
        corrected_pt  = cset["pt_metphicorr_pfmet_mc"].evaluate(pt_clipped, phi_arr, npv_clipped, run_np)
        corrected_phi = cset["phi_metphicorr_pfmet_mc"].evaluate(pt_clipped, phi_arr, npv_clipped, run_np)

    return corrected_pt, corrected_phi

#here function from coffea METCorrector corrector to be used to update T1 MET with current JECs applied
def update_met_t1_corr(met_pt, met_phi, jet_pt, jet_phi, jet_pt_orig):
    sj, cj = numpy.sin(jet_phi), numpy.cos(jet_phi)

    #invert signs
    x = met_pt * numpy.cos(met_phi) - awkward.sum(
        jet_pt * cj - jet_pt_orig * cj, axis=1
    )
    y = met_pt * numpy.sin(met_phi) - awkward.sum(
        jet_pt * sj - jet_pt_orig * sj, axis=1
    )    
    return awkward.zip({"pt": numpy.hypot(x, y), "phi": numpy.arctan2(y, x)})

def apply_uncl_variation_to_met_t1(met_pt, met_phi,positive=None,dx=None,dy=None):

    x = met_pt * numpy.cos(met_phi) 
    y = met_pt * numpy.sin(met_phi) 
    if positive is not None and dx is not None and dy is not None:
        x = x + dx if positive else x - dx
        y = y + dy if positive else y - dy

    return awkward.zip(
        {"pt": numpy.hypot(x, y), "phi": numpy.arctan2(y, x)}, depth_limit=1
    )