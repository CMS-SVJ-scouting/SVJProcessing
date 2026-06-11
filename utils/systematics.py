import awkward as ak
import numpy as np
import copy as cp
import utils.gen_matching_tools as gen_matching_tools
from utils.jet_energy_scale_svj_factory import SVJCustomJESCalculator 
from utils.met_jecs_factory import update_met_t1_corr, apply_uncl_variation_to_met_t1
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import CorrectedMETFactory
from utils.met_significance_factory_pfnano import MetSignificanceCalculator

# Maps (year_era, run) -> JEC key era for cases where a single JEC txt file
# covers multiple run eras (e.g. Fall17_17Nov2017DE covers both Run D and Run E).
# year_era uses the resolved prefix: "2016preVFP", "2016postVFP", "2017", "2018".
_RUN_ERA_TO_JEC_ERA = {
    # 2016 preVFP: B, C, D all share the BCD JEC file
    ("2016preVFP", "dataB"):   "dataBCD",
    ("2016preVFP", "dataC"):   "dataBCD",
    ("2016preVFP", "dataD"):   "dataBCD",
    # 2016 postVFP: E+F share EF; G+H share GH
    ("2016postVFP", "dataE"):  "dataEF",
    ("2016postVFP", "dataF"):  "dataEF",
    ("2016postVFP", "dataG"):  "dataGH",
    ("2016postVFP", "dataH"):  "dataGH",
    # 2017: D and E share the DE JEC file; B, C, F are individual
    ("2017", "dataD"):         "dataDE",
    ("2017", "dataE"):         "dataDE",
    # 2018: RunA/B/C/D are all individual — no remapping needed
}

def _resolve_year_era(year):
    """Convert raw year string to the year prefix used in factory keys."""
    if "APV" in year:
        return year.replace("APV", "preVFP")
    if "2016" in year:
        return "2016postVFP"
    return year

def _map_run_to_jec_era(run, year):
    """
    Translate a per-run era string (as set by -run_era) to the JEC key era
    for the given year.
    E.g. year='2017', run='dataD' -> 'dataDE'
         year='2016APV', run='dataB' -> 'dataBCD'
         year='2018', run='dataRunA' -> 'dataRunA' (unchanged)
    """
    year_era = _resolve_year_era(year)
    return _RUN_ERA_TO_JEC_ERA.get((year_era, run), run)


def calc_jec_variation(
        pt, eta, phi, energy,
        jer_factor, jec_unc, orig_idx,
        variation_orig_idx, variation_jer_factor
        ):
    """
    Applies a JEC variation (up or down) on a 4-vector.

    Note there are 3 'ordering levels':
    - "Final": the final ordering of jets after centrally applied corrections
    - "Original": the ordering of jets _before_ any corrections
    - "Variation": the final ordering of jets after the applying the correction of the
        _variation_

    The algorithm below first creates a map to reorder "Final" to "Variation", then
    applies the correction after ordering everything in "Variation" ordering.

    Args:
        pt (ak.Array): jet pt
        eta (ak.Array): jet eta
        phi (ak.Array): jet phi
        energy (ak.Array): jet energy
        jer_factor (ak.Array): the JER factor that was applied centrally to obtain the
            final jet
        jec_unc (ak.Array): the JEC uncertainty
        orig_idx (ak.Array): mapping of final corrected jet ordering back to 'original'
            ordering
        variation_orig_idx (ak.Array): mapping of variation ordering back to 'original'
            ordering
        variation_jer_factor (ak.Array): the variation's JER factor

    Returns:
        (ak.Array, ak.Array, ak.Array, ak.Array, ak.Array) : pt, eta, phi, energy after
            applying the variation and reordered by the pT after variation; permutation
            to propagate the reordering to the whole collection
    """

    # Create a map to reorder final corrected jets to the ordering of the variation
    map_orig_idx_to_var_idx = ak.argsort(variation_orig_idx, axis=-1)
    map_final_idx_to_orig_idx = orig_idx
    reorder_final_to_var = map_final_idx_to_orig_idx[map_orig_idx_to_var_idx]

    # Reorder everything that is in "Final" order to "Variation" order
    pt = pt[reorder_final_to_var]
    eta = eta[reorder_final_to_var]
    phi = phi[reorder_final_to_var]
    energy = energy[reorder_final_to_var]
    jer_factor = jer_factor[reorder_final_to_var]
    jec_unc = jec_unc[reorder_final_to_var]

    corr = 1. / jer_factor * (1.+jec_unc) * variation_jer_factor
    return pt*corr, eta, phi, energy*corr, reorder_final_to_var


def calc_jer_variation(
        pt, eta, phi, energy,
        jer_factor, orig_idx,
        variation_orig_idx, variation_jer_factor
        ):
    """
    Applies a JER variation (up or down) on a 4-vector.

    Note there are 3 'ordering levels':
    - "Final": the final ordering of jets after centrally applied corrections
    - "Original": the ordering of jets _before_ any corrections
    - "Variation": the final ordering of jets after the applying the correction of the
        _variation_

    The algorithm below first creates a map to reorder "Final" to "Variation", then
    applies the correction after ordering everything in "Variation" ordering.

    Args:
        pt (ak.Array): jet pt
        eta (ak.Array): jet eta
        phi (ak.Array): jet phi
        energy (ak.Array): jet energy
        jer_factor (ak.Array): the JER factor that was applied centrally to obtain the
            final jet
        orig_idx (ak.Array): mapping of final corrected jet ordering back to 'original'
            ordering
        variation_orig_idx (ak.Array): mapping of variation ordering back to 'original'
            ordering
        variation_jer_factor (ak.Array): the variation's JER factor

    Returns:
        (ak.Array, ak.Array, ak.Array, ak.Array) : pt, eta, phi, and energy after
            applying the variation and reordered by the pT after variation.
    """

    # Create a map to reorder final corrected jets to the ordering of the variation
    map_orig_idx_to_var_idx = ak.argsort(variation_orig_idx, axis=-1)
    map_final_idx_to_orig_idx = orig_idx
    reorder_final_to_var = map_final_idx_to_orig_idx[map_orig_idx_to_var_idx]

    # Reorder everything that is in "Final" order to "Variation" order
    pt = pt[reorder_final_to_var]
    eta = eta[reorder_final_to_var]
    phi = phi[reorder_final_to_var]
    energy = energy[reorder_final_to_var]
    jer_factor = jer_factor[reorder_final_to_var]

    corr = 1. / jer_factor * variation_jer_factor
    return pt*corr, eta, phi, energy*corr, reorder_final_to_var



###############################
####### PFNano section ########
###############################

def build_genjet_idx_manual(jet_eta_reco, jet_phi_reco, jet_eta_gen, jet_phi_gen, max_dr=0.4):
    """
    Manual gen-matching via deltaR.
    TODO: This should be implemented in the CMSSW ntuplizer for efficiency!
    Returns array of gen jet indices for each reco jet (-1 if no match).
    
    This is a PLACEHOLDER for testing - proper implementation should be upstream.
    """
    # Calculate deltaR between all reco-gen jet pairs
    def calc_dr(eta1, phi1, eta2, phi2):
        deta = eta1 - eta2
        dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
        return np.sqrt(deta**2 + dphi**2)
    
    indices = []
    for evt_reco_eta, evt_reco_phi, evt_gen_eta, evt_gen_phi in zip(
        jet_eta_reco, jet_phi_reco, jet_eta_gen, jet_phi_gen
    ):
        evt_indices = []
        for reco_eta, reco_phi in zip(evt_reco_eta, evt_reco_phi):
            best_idx = -1
            min_dr = max_dr
            for gen_idx, (gen_eta, gen_phi) in enumerate(zip(evt_gen_eta, evt_gen_phi)):
                dr = calc_dr(reco_eta, reco_phi, gen_eta, gen_phi)
                if dr < min_dr:
                    min_dr = dr
                    best_idx = gen_idx
            evt_indices.append(best_idx)
        indices.append(evt_indices)
    
    return ak.Array(indices)


def make_jets_for_jerc(events,jet_coll, event_rho, correction_key):
        
        jets = {}
        # Check if rawFactor exists (standard NanoAOD) or use jets as-is (scouting)
        raw_factor_field = f"{jet_coll}_rawFactor"
        if raw_factor_field in events.fields:
            jets["pt_raw"] = (1 - events[raw_factor_field])*events[f"{jet_coll}_pt"]
            jets["mass_raw"] = (1 - events[raw_factor_field])*events[f"{jet_coll}_mass"]
        else:
            # Scouting data: pt is already scouting-JEC-corrected; use it as pt_raw so
            # the official JEC is applied on top of the scouting correction.
            jets["pt_raw"] = events[f"{jet_coll}_pt"]
            jets["mass_raw"] = events[f"{jet_coll}_mass"]
        jets["event_rho"] = ak.broadcast_arrays(event_rho, events[f"{jet_coll}_pt"])[0]
        if "NOJER" not in correction_key:
            if "FatJet" in jet_coll:
                m_genMatch_dR2max = 0.4*0.4
                
                # TODO: FatJet_genJetAK8Idx should be implemented in CMSSW ntuplizer!
                # For now, build indices manually as placeholder for testing
                if f"FatJet_genJetAK8Idx" in events.fields:
                    genjet_idx = events[f"FatJet_genJetAK8Idx"]
                else:
                    # Fallback: manual matching (slow, for testing only)
                    genjet_idx = build_genjet_idx_manual(
                        events[f"{jet_coll}_eta"],
                        events[f"{jet_coll}_phi"],
                        events[f"Gen{jet_coll}_eta"],
                        events[f"Gen{jet_coll}_phi"],
                        max_dr=0.4
                    )
                
                jets["pt_gen"] = gen_matching_tools.get_matched_gen_jets(events[f"{jet_coll}_pt"], 
                                                                         events[f"{jet_coll}_eta"], 
                                                                         events[f"{jet_coll}_phi"], 
                                                                         events[f"Gen{jet_coll}_pt"], 
                                                                         events[f"Gen{jet_coll}_eta"], 
                                                                         events[f"Gen{jet_coll}_phi"], 
                                                                         genjet_idx,
                                                                         m_genMatch_dR2max,
                                                                         correction_key.replace("NOJEC",""),
                                                                         jet_coll,
                                                                         jets["event_rho"]
                                                                         )
            else:
                m_genMatch_dR2max = 0.2*0.2
                
                # TODO: Jet_genJetIdx should be implemented in CMSSW ntuplizer!
                # For now, build indices manually as placeholder for testing
                if f"{jet_coll}_genJetIdx" in events.fields:
                    genjet_idx = events[f"{jet_coll}_genJetIdx"]
                else:
                    # Fallback: manual matching (slow, for testing only)
                    genjet_idx = build_genjet_idx_manual(
                        events[f"{jet_coll}_eta"],
                        events[f"{jet_coll}_phi"],
                        events[f"Gen{jet_coll}_eta"],
                        events[f"Gen{jet_coll}_phi"],
                        max_dr=0.2
                    )
                
                jets["pt_gen"] = gen_matching_tools.get_matched_gen_jets(events[f"{jet_coll}_pt"], 
                                                                         events[f"{jet_coll}_eta"], 
                                                                         events[f"{jet_coll}_phi"], 
                                                                         events[f"Gen{jet_coll}_pt"], 
                                                                         events[f"Gen{jet_coll}_eta"], 
                                                                         events[f"Gen{jet_coll}_phi"], 
                                                                         genjet_idx,
                                                                         m_genMatch_dR2max,
                                                                         correction_key.replace("NOJEC",""),
                                                                         jet_coll,
                                                                         jets["event_rho"]
                                                                         )


        #make jets an ak array
        if "NOJER" not in correction_key:
            jets = ak.zip({
                "pt": events[f"{jet_coll}_pt"],
                "eta": events[f"{jet_coll}_eta"],
                "phi": events[f"{jet_coll}_phi"],
                "mass": events[f"{jet_coll}_mass"],
                "area": events[f"{jet_coll}_area"],
                "pt_raw": jets["pt_raw"],
                "mass_raw": jets["mass_raw"],
                "event_rho": jets["event_rho"],
                "pt_gen": ak.values_astype(ak.fill_none(jets["pt_gen"], 0), np.float32),
            })
        else:
            jets = ak.zip({
                "pt": events[f"{jet_coll}_pt"],
                "eta": events[f"{jet_coll}_eta"],
                "phi": events[f"{jet_coll}_phi"],
                "mass": events[f"{jet_coll}_mass"],
                "area": events[f"{jet_coll}_area"],
                "pt_raw": jets["pt_raw"],
                "mass_raw": jets["mass_raw"],
                "event_rho": jets["event_rho"],
            })
             
        return jets

#CZZ: hardcoded PFMET for now
def make_met_for_jerc(events):
        
        met = {}
        
        # Check if RawMET exists (standard NanoAOD) or use ScoutMET/MET (scouting)
        if "RawMET_pt" in events.fields:
            met["pt"] = events["RawMET_pt"]
            met["phi"] = events["RawMET_phi"]
        elif "ScoutMET_pt" in events.fields:
            # Scouting data uses ScoutMET
            met["pt"] = events["ScoutMET_pt"]
            met["phi"] = events["ScoutMET_phi"]
        else:
            # Fallback to regular MET
            met["pt"] = events["MET_pt"]
            met["phi"] = events["MET_phi"]
            
        met["MetUnclustEnUpDeltaX"] = events["MET_MetUnclustEnUpDeltaX"]
        met["MetUnclustEnUpDeltaY"] = events["MET_MetUnclustEnUpDeltaY"]

        #make met an ak array
        met = ak.zip({
            "pt": met["pt"],
            "phi": met["phi"],
            "MetUnclustEnUpDeltaX": met["MetUnclustEnUpDeltaX"],
            "MetUnclustEnUpDeltaY": met["MetUnclustEnUpDeltaY"],
        })
        


        return met




def apply_jers_PFNano(
    events: ak.Array,
    year: str,
    run: str,
    jet_coll: str,
    jerc_variations: dict,
    jerc_cache: dict,
    ) -> ak.Array:
    
    jet_factory = jerc_variations[f'{jet_coll.lower()}_factory']


    # calculate all variables needed as inputs
    jerc_key_label = ""

    # jerc_key_label = "NOJEC"  # COMMENTED TO TEST JER

    #build jet corrections
    correction_key = None
    if "2016" in year:
        if "APV" in year:
            correction_key = f"{year.replace('APV','preVFP')}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 
        else:
            correction_key = f"2016postVFP{_map_run_to_jec_era(run, year)}{jerc_key_label}"
    else:
        correction_key = f"{year}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 

    # Check if using scouting data (rho) or standard NanoAOD (fixedGridRhoFastjetAll)
    rho = events.rho if "rho" in events.fields else events.fixedGridRhoFastjetAll
    jets_corrected = jet_factory[correction_key].build(make_jets_for_jerc(events,jet_coll, rho, correction_key), jerc_cache)

    #extract corrections to jets
    pt_corr = jets_corrected.pt_jer
    eta_corr = events[f"{jet_coll}_eta"]
    phi_corr = events[f"{jet_coll}_phi"]
    mass_corr = jets_corrected.mass_jer


    return pt_corr, eta_corr, phi_corr, mass_corr


def apply_jecs_PFNano(
    events: ak.Array,
    year: str,
    run: str,
    jet_coll: str,
    jerc_variations: dict,
    jerc_cache: dict,
    ) -> ak.Array:
    
    jet_factory = jerc_variations[f'{jet_coll.lower()}_factory']


    # calculate all variables needed as inputs
    # JER smearing only applies to MC; for data use NOJER key (JEC residuals only)
    jerc_key_label = "NOJER" if "data" in run.lower() else ""


    #build jet corrections
    correction_key = None
    if "2016" in year:
        if "APV" in year:
            correction_key = f"{year.replace('APV','preVFP')}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 
        else:
            correction_key = f"2016postVFP{_map_run_to_jec_era(run, year)}{jerc_key_label}"
    else:
        correction_key = f"{year}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 

    # Check if using scouting data (rho) or standard NanoAOD (fixedGridRhoFastjetAll)
    rho = events.rho if "rho" in events.fields else events.fixedGridRhoFastjetAll
    jets_corrected = jet_factory[correction_key].build(make_jets_for_jerc(events,jet_coll, rho, correction_key), jerc_cache)


    #extract corrections to jets
    pt_corr = jets_corrected.pt_jec
    eta_corr = events[f"{jet_coll}_eta"]
    phi_corr = events[f"{jet_coll}_phi"]
    mass_corr = jets_corrected.mass_jec

    #define new raw factor
    pt_raw = (1. - events[f"{jet_coll}_rawFactor"])*events[f"{jet_coll}_pt"]
    raw_factor = 1. - pt_raw/pt_corr


    return pt_corr, eta_corr, phi_corr, mass_corr,raw_factor

#Apply JECs and JERs to nominal jets 
def apply_jercs_PFNano(
    events: ak.Array,
    year: str,
    run: str,
    jet_coll: str,
    jerc_variations: dict,
    jerc_cache: dict,
    ) -> ak.Array:
    
    jet_factory = jerc_variations[f'{jet_coll.lower()}_factory']

    # calculate all variables needed as inputs, apply both JEC and JER
    # JER smearing only applies to MC; for data use NOJER key (JEC residuals only)
    jerc_key_label = "NOJER" if "data" in run.lower() else ""
    # jerc_key_label = "NOJER" 

    #build jet corrections
    correction_key = None
    if "2016" in year:
        if "APV" in year:
            correction_key = f"{year.replace('APV','preVFP')}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 
        else:
            correction_key = f"2016postVFP{_map_run_to_jec_era(run, year)}{jerc_key_label}"
    else:
        correction_key = f"{year}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 

    # Check if using scouting data (rho) or standard NanoAOD (fixedGridRhoFastjetAll)
    rho = events.rho if "rho" in events.fields else events.fixedGridRhoFastjetAll
    jets_input = make_jets_for_jerc(events,jet_coll, rho, correction_key)
    jets_corrected = jet_factory[correction_key].build(jets_input, jerc_cache)

    #extract corrections to jets
    pt_corr = jets_corrected.pt
    eta_corr = jets_corrected.eta
    phi_corr = jets_corrected.phi
    mass_corr = jets_corrected.mass
    
    # Extract raw (uncorrected) values to save
    pt_uncorr = jets_input.pt_raw
    mass_uncorr = jets_input.mass_raw

    return pt_corr, eta_corr, phi_corr, mass_corr, pt_uncorr, mass_uncorr


#Propage JECs to MET
def propagate_jecs_to_MET_PFNano(
                events: ak.Array,
                year: str,
                run: str,
                jet_coll: str,
                jerc_variations: dict,
                jerc_cache: dict,
                ) -> ak.Array:
    
    jet_factory = jerc_variations[f'{jet_coll.lower()}_factory']


    # calculate all variables needed as inputs
    jerc_key_label = "NOJER"


    #build jet corrections
    correction_key = None
    if "2016" in year:
        if "APV" in year:
            correction_key = f"{year.replace('APV','preVFP')}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 
        else:
            correction_key = f"2016postVFP{_map_run_to_jec_era(run, year)}{jerc_key_label}"
    else:
        correction_key = f"{year}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 

    # Check if using scouting data (rho) or standard NanoAOD (fixedGridRhoFastjetAll)
    rho = events.rho if "rho" in events.fields else events.fixedGridRhoFastjetAll
    
    # Use CorrT1METJet for T1 MET jets if available, otherwise fall back to jet_coll
    t1met_coll = "CorrT1METJet" if "CorrT1METJet_pt" in events.fields else jet_coll

    jets_corrected_nom = jet_factory[correction_key].build(make_jets_for_jerc(events,t1met_coll, rho, correction_key), jerc_cache)

    #extract corrected jets
    jet_pt_corr_nom  = jets_corrected_nom.pt
    jet_phi_corr_nom  = events[f"{t1met_coll}_phi"]

    # The factory raw input is scouting-corrected pt for scouting events.
    # Keep both baselines so we can save two corrected MET flavors:
    # - official_only : official JEC delta on top of scouting-corrected jets
    # - all_corr      : full delta from truly raw scouting pt to final corrected pt
    jet_pt_raw_official_only = jets_corrected_nom.pt_raw
    pt_raw_scouting_field = f"{t1met_coll}_pt_raw_scouting"
    if pt_raw_scouting_field in events.fields:
        jet_pt_raw_all_corr = events[pt_raw_scouting_field]
    else:
        jet_pt_raw_all_corr = jet_pt_raw_official_only

    # Get raw MET - check for RawMET (standard) or ScoutMET (scouting)
    if "RawMET_pt" in events.fields:
        met_pt_raw = events["RawMET_pt"]
        met_phi_raw = events["RawMET_phi"]
    elif "ScoutMET_pt" in events.fields:
        met_pt_raw = events["ScoutMET_pt"]
        met_phi_raw = events["ScoutMET_phi"]
    else:
        met_pt_raw = events["MET_pt"]
        met_phi_raw = events["MET_phi"]

    corr_t1_met_official_only = update_met_t1_corr(
        met_pt_raw,
        met_phi_raw,
        jet_pt_corr_nom,
        jet_phi_corr_nom,
        jet_pt_raw_official_only,
    )
    corr_t1_met_all_corr = update_met_t1_corr(
        met_pt_raw,
        met_phi_raw,
        jet_pt_corr_nom,
        jet_phi_corr_nom,
        jet_pt_raw_all_corr,
    )
    
    # Return corrected MET and uncorrected (raw) MET for saving
    return (
        corr_t1_met_all_corr.pt,
        corr_t1_met_all_corr.phi,
        met_pt_raw,
        met_phi_raw,
        corr_t1_met_official_only.pt,
        corr_t1_met_official_only.phi,
    )



#Propage JECs to MET
def propagate_jecs_to_METSig_PFNano(
                events: ak.Array,
                year: str,
                run: str,
                jet_coll: str,
                jerc_variations: dict,
                jerc_cache: dict,
                make_unclustered_En_var: bool = False,
                variation: str = None,
                ) -> ak.Array:
    
    jet_factory = jerc_variations[f'{jet_coll.lower()}_factory']

    # calculate all variables needed as inputs
    # jerc_key_label = "NOJER"  # COMMENTED TO TEST JER
    jerc_key_label = ""

    #build jet corrections
    correction_key = None
    if "2016" in year:
        if "APV" in year:
            correction_key = f"{year.replace('APV','preVFP')}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 
        else:
            correction_key = f"2016postVFP{_map_run_to_jec_era(run, year)}{jerc_key_label}"
    else:
        correction_key = f"{year}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 

    # Check if using scouting data (rho) or standard NanoAOD (fixedGridRhoFastjetAll)
    rho = events.rho if "rho" in events.fields else events.fixedGridRhoFastjetAll
    jets_corrected_nom = jet_factory[correction_key].build(make_jets_for_jerc(events,jet_coll, rho, correction_key), jerc_cache)

    #extract corrected jets
    jet_pt_corr_nom  = jets_corrected_nom.pt

    #copy events to avoid modifying the original array
    events_copy = cp.deepcopy(events)

    #correct the copy of events with corrected jets
    #adding the JEC varied jets to the events
    jes_permutation = ak.argsort(jet_pt_corr_nom, ascending=False)
    jes_corr_pt = jet_pt_corr_nom[jes_permutation]
    jes_corr_eta = events_copy[f"{jet_coll}_eta"][jes_permutation]
    jes_corr_phi = events_copy[f"{jet_coll}_phi"][jes_permutation]
    jes_corr_mass = events_copy[f"{jet_coll}_mass"][jes_permutation]
    events_copy[f"{jet_coll}_pt"] = jes_corr_pt
    events_copy[f"{jet_coll}_eta"] = jes_corr_eta
    events_copy[f"{jet_coll}_phi"] = jes_corr_phi
    events_copy[f"{jet_coll}_mass"] = jes_corr_mass
    jes_pf_cand_jet_idx = events_copy[f"{jet_coll}PFCands_jetIdx"]
    jes_sorted_pf_cand_jet_idx = ak.Array([p[idx] for idx, p in zip(jes_pf_cand_jet_idx, jes_permutation)])
    events_copy[f"{jet_coll}PFCands_jetIdx"] = jes_sorted_pf_cand_jet_idx

    variation_direction = ""
    #extract variation direction
    if make_unclustered_En_var:
        #extract from variation
        variation_direction = "up" if "up" in variation else "down"

    met_sig_recalculator = MetSignificanceCalculator(events_copy,
                            year,
                            run,
                            make_unclustered_En_var,
                            variation_direction,
                        ) 
 
     
    met_sig_corr = met_sig_recalculator.getSignificance() 

    #delete the copy of events
    del events_copy

    return met_sig_corr




def calc_jerc_variations_PFNano(
    events: ak.Array,
    year: str,
    run: str,
    jet_coll: str,
    jerc_variations: dict,
    variation: str,
    jerc_cache: dict,
    ) -> ak.Array:
    

    jet_factory = jerc_variations[f'{jet_coll.lower()}_factory']

    # calculate all variables needed as inputs
    # TODO for now only JEC, no JER, empty string before 
    # For JEC variations, use NOJER to avoid gen-matching requirement
    # For JER variations, use empty string to include both JEC and JER
    jerc_key_label = "NOJER" if "jec" in variation else ""
    access_jerc_corr_jets =  ""

    if "jec" in variation:
        access_jerc_corr_jets = "JES_jes"
    if "jer" in variation:
        access_jerc_corr_jets = "JER"

    #build jet corrections
    correction_key = None
    if "2016" in year:
        if "APV" in year:
            correction_key = f"{year.replace('APV','preVFP')}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 
        else:
            correction_key = f"2016postVFP{_map_run_to_jec_era(run, year)}{jerc_key_label}"
    else:
        correction_key = f"{year}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 

    
    # Check if using scouting data (rho) or standard NanoAOD (fixedGridRhoFastjetAll)
    rho = events.rho if "rho" in events.fields else events.fixedGridRhoFastjetAll

    # Apply JERCs to the main jet collection (Jet or FatJet) — output saved to file
    jets_input = make_jets_for_jerc(events, jet_coll, rho, correction_key)
    jets_corrected = jet_factory[correction_key].build(jets_input, jerc_cache)

    #extract the direction of the variation
    direction = "up" if "up" in variation else "down"

    #make dummy x_ratio
    x_ratio = ak.ones_like(jets_corrected.pt)

    #extract corrections to jets (from jet_coll)
    pt_corr = eval(f"jets_corrected.{access_jerc_corr_jets}.{direction}.pt")
    eta_corr = eval(f"jets_corrected.{access_jerc_corr_jets}.{direction}.eta")
    phi_corr = eval(f"jets_corrected.{access_jerc_corr_jets}.{direction}.phi")
    mass_corr = eval(f"jets_corrected.{access_jerc_corr_jets}.{direction}.mass")

    # Extract uncorrected values to save (from jet_coll)
    pt_uncorr = jets_input.pt_raw
    mass_uncorr = jets_input.mass_raw

    met = None
    met_pt_raw = None
    met_phi_raw = None
    if jet_coll == "Jet":
        # For T1 MET propagation, use CorrT1METJet if available (AK4 only), else Jet
        t1met_coll = "CorrT1METJet" if "CorrT1METJet_pt" in events.fields else jet_coll
        if (t1met_coll != jet_coll) and ("jer" not in variation):
            # Compute JERCs on CorrT1METJet separately for MET propagation only
            t1met_jets_input = make_jets_for_jerc(events, t1met_coll, rho, correction_key)
            t1met_jets_corrected = jet_factory[correction_key].build(t1met_jets_input, {})
            t1met_pt_corr = eval(f"t1met_jets_corrected.{access_jerc_corr_jets}.{direction}.pt")
            t1met_phi_corr = eval(f"t1met_jets_corrected.{access_jerc_corr_jets}.{direction}.phi")
            _t1met_pt_raw_var = eval(f"t1met_jets_corrected.{access_jerc_corr_jets}.{direction}.pt_raw")
            pt_raw_scouting_field = f"{t1met_coll}_pt_raw_scouting"
            t1met_pt_raw_var = events[pt_raw_scouting_field] if pt_raw_scouting_field in events.fields else _t1met_pt_raw_var
        else:
            # No CorrT1METJet: use Jet corrections for MET propagation too
            t1met_pt_corr = pt_corr
            t1met_phi_corr = phi_corr
            _t1met_pt_raw_var = eval(f"jets_corrected.{access_jerc_corr_jets}.{direction}.pt_raw")
            pt_raw_scouting_field = f"{jet_coll}_pt_raw_scouting"
            t1met_pt_raw_var = events[pt_raw_scouting_field] if pt_raw_scouting_field in events.fields else _t1met_pt_raw_var

        # Get raw MET - check for RawMET (standard) or ScoutMET (scouting)
        if "RawMET_pt" in events.fields:
            met_pt_raw = events["RawMET_pt"]
            met_phi_raw = events["RawMET_phi"]
        elif "ScoutMET_pt" in events.fields:
            met_pt_raw = events["ScoutMET_pt"]
            met_phi_raw = events["ScoutMET_phi"]
        else:
            met_pt_raw = events["MET_pt"]
            met_phi_raw = events["MET_phi"]
        met = update_met_t1_corr(met_pt_raw, met_phi_raw, t1met_pt_corr, t1met_phi_corr, t1met_pt_raw_var)

    #extract corrections to met
    met_ptcorr = None
    met_phicorr = None
    if jet_coll == "Jet":
        met_ptcorr = met.pt
        met_phicorr = met.phi


    return pt_corr, eta_corr, phi_corr, mass_corr, met_ptcorr, met_phicorr, x_ratio, pt_uncorr, mass_uncorr, met_pt_raw, met_phi_raw



def calc_unclustered_met_variations_PFNano(
    events: ak.Array,
    year: str,
    run: str,
    jet_coll: str,
    jerc_variations: dict,
    variation: str,
    jerc_cache: dict,
    ) -> ak.Array:
    
    jet_factory = jerc_variations[f'{jet_coll.lower()}_factory']

    # calculate all variables needed as inputs
    # jerc_key_label = "NOJER"  # COMMENTED TO TEST JER
    jerc_key_label = ""


    #build jet corrections
    correction_key = None
    if "2016" in year:
        if "APV" in year:
            correction_key = f"{year.replace('APV','preVFP')}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 
        else:
            correction_key = f"2016postVFP{_map_run_to_jec_era(run, year)}{jerc_key_label}"
    else:
        correction_key = f"{year}{_map_run_to_jec_era(run, year)}{jerc_key_label}" 

    # Check if using scouting data (rho) or standard NanoAOD (fixedGridRhoFastjetAll)
    rho = events.rho if "rho" in events.fields else events.fixedGridRhoFastjetAll
    # Use CorrT1METJet for T1 MET jets if available, otherwise fall back to jet_coll
    t1met_coll = "CorrT1METJet" if "CorrT1METJet_pt" in events.fields else jet_coll
    jets_corrected_nom = jet_factory[correction_key].build(make_jets_for_jerc(events, t1met_coll, rho, correction_key), jerc_cache)

    #extract corrected jets
    jet_pt_corr_nom  = jets_corrected_nom.pt
    jet_phi_corr_nom  = events[f"{t1met_coll}_phi"]

    # Use truly raw pt (before scouting JEC) as baseline for T1 formula when available
    pt_raw_scouting_field = f"{t1met_coll}_pt_raw_scouting"
    if pt_raw_scouting_field in events.fields:
        jet_pt_raw = events[pt_raw_scouting_field]
    else:
        jet_pt_raw = jets_corrected_nom.pt_raw

    # Get raw MET - check for RawMET (standard) or ScoutMET (scouting)
    if "RawMET_pt" in events.fields:
        met_pt_raw = events["RawMET_pt"]
        met_phi_raw = events["RawMET_phi"]
    elif "ScoutMET_pt" in events.fields:
        met_pt_raw = events["ScoutMET_pt"]
        met_phi_raw = events["ScoutMET_phi"]
    else:
        met_pt_raw = events["MET_pt"]
        met_phi_raw = events["MET_phi"]

    #get T1 corrected MET (nominal)
    corr_t1_met = update_met_t1_corr(met_pt_raw, met_phi_raw, jet_pt_corr_nom, jet_phi_corr_nom, jet_pt_raw)

    #extract the direction of the variation
    direction = "up" if "up" in variation else "down"

    positive = None
    if (direction == "up"):
        positive = True
    if (direction == "down"):
        positive = False

    #unpack corr_t1_met
    met_pt_t1_nom = corr_t1_met.pt
    met_phi_t1_nom = corr_t1_met.phi

    #extracting correction due to unclustered energy variation
    corr_t1_met_uncl_var = apply_uncl_variation_to_met_t1(met_pt_t1_nom,met_phi_t1_nom, positive=positive, dx=events["MET_MetUnclustEnUpDeltaX"], dy=events["MET_MetUnclustEnUpDeltaY"])

    return corr_t1_met_uncl_var.pt, corr_t1_met_uncl_var.phi



def calc_custom_svj_jes_variations_PFNano(
                events: ak.Array,
                year: str,
                run: str,
                jet_coll: str,
                variation: str,
                ) -> ak.Array:
    

    #build jet corrections
    correction_key = None
    if "2016" in year:
        if "APV" in year:
            correction_key = f"2016preVFP{_map_run_to_jec_era(run, year)}"
        else:
            correction_key = f"2016postVFP{_map_run_to_jec_era(run, year)}"
    else:
        correction_key = f"{year}{_map_run_to_jec_era(run, year)}"

    #extract the direction of the variation
    direction = "up" if "up" in variation else "down"

    #build custom jec calculator
    svjJESCalc = SVJCustomJESCalculator(
        events,
        jet_coll,
        correction_key
    )

    #fetch correction
    #need to retrieve pT raw pT to propagate to MET 
    svj_jecs, x_ratio = svjJESCalc.getVariation(direction)
    jets_corrected = ak.zip({
        "pt": events[f"{jet_coll}_pt"]*svj_jecs,
        "phi": events[f"{jet_coll}_phi"],
        "pt_raw": events[f"{jet_coll}_pt"],
    })

    # For SVJ JES variations, the uncorrected values are the pre-existing corrected values
    # (since SVJ JES is applied on top of standard JEC)
    # These should already exist in events as *_uncorr from the nominal correction
    # But we need to return them here for consistency
    pt_uncorr = events[f"{jet_coll}_pt_uncorr"] if f"{jet_coll}_pt_uncorr" in events.fields else events[f"{jet_coll}_pt"]
    mass_uncorr = events[f"{jet_coll}_mass_uncorr"] if f"{jet_coll}_mass_uncorr" in events.fields else events[f"{jet_coll}_mass"]

    #now propagate custom jes to met (T1-like correction) if jet_coll is Jet
    met_ptcorr = None
    met_phicorr = None
    met_pt_uncorr = None
    met_phi_uncorr = None
    if jet_coll == "Jet":
        # Check if this is scouting data (has ScoutMET) or regular NanoAOD (has MET)
        met_field = "ScoutMET_pt" if "ScoutMET_pt" in events.fields else "MET_pt"
        met_phi_field = "ScoutMET_phi" if "ScoutMET_phi" in events.fields else "MET_phi"
        met_uncorr_field = f"{met_field.split('_')[0]}_pt_uncorr"
        met_phi_uncorr_field = f"{met_phi_field.split('_')[0]}_phi_uncorr"
        
        corr_met_custom_jecs = update_met_t1_corr(events[met_field], events[met_phi_field], jets_corrected.pt, jets_corrected.phi, jets_corrected.pt_raw)
        met_ptcorr = corr_met_custom_jecs.pt
        met_phicorr = corr_met_custom_jecs.phi
        
        # Get uncorrected MET if it exists
        met_pt_uncorr = events[met_uncorr_field] if met_uncorr_field in events.fields else events[met_field]
        met_phi_uncorr = events[met_phi_uncorr_field] if met_phi_uncorr_field in events.fields else events[met_phi_field]

    return jets_corrected.pt, events[f"{jet_coll}_eta"], jets_corrected.phi, events[f"{jet_coll}_mass"]*svj_jecs, met_ptcorr, met_phicorr, x_ratio, pt_uncorr, mass_uncorr, met_pt_uncorr, met_phi_uncorr
    



    