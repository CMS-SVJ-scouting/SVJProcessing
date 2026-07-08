import awkward as ak

from skimmer import skimmer_utils
from utils.awkward_array_utilities import as_type
import analysis_configs.triggers as trg
import utils.variables_computation.event_variables as event_vars
from analysis_configs.met_filters import met_filters_nanoaod as met_filters
from analysis_configs import sequences_s_channel_scouting as sequences


GOLDEN_JSON_PATHS = {
    "2016APV": "/work/mgais/SVJProcessing/data/golden_json/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
    "2016": "/work/mgais/SVJProcessing/data/golden_json/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
    "2017": "/work/mgais/SVJProcessing/data/golden_json/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", 
    "2018": "/work/mgais/SVJProcessing/data/golden_json/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt", 
}


def process(events, cut_flow, year, primary_dataset="", dataset_name="", pn_tagger=False, **kwargs):
    """SVJ s-channel scouting pre-selection."""

    # Golden JSON lumi mask (data only)
    if skimmer_utils.is_data(events) and len(events) != 0:
        from coffea.lumi_tools import LumiMask
        lumi_mask = LumiMask(GOLDEN_JSON_PATHS[year])
        mask = lumi_mask(events.run, events.lumSec)
        events = events[mask]
        skimmer_utils.update_cut_flow(cut_flow, "GoldenJSON", events)

    # TT stitching: avoid double-counting between inclusive and HT-binned samples.
    # Inclusive keeps lheHT < 600; HT-binned keeps lheHT >= 600 
    _ttjets_ht_binned = ["TTJets_HT-600to800", "TTJets_HT-800to1200", "TTJets_HT-1200to2500", "TTJets_HT-2500toInf"]
    if "TTJets_TuneCP5" in dataset_name and len(events) != 0:
        events = events[events.lheHT < 600]
        skimmer_utils.update_cut_flow(cut_flow, "TTStitching", events)
    elif any(b in dataset_name for b in _ttjets_ht_binned) and len(events) != 0:
        events = events[events.lheHT >= 600]
        skimmer_utils.update_cut_flow(cut_flow, "TTStitching", events)

    # Trigger event selection
    triggers = getattr(trg, f"s_channel_scouting")
    events = skimmer_utils.apply_trigger_cut(events, triggers)
    skimmer_utils.update_cut_flow(cut_flow, "Trigger", events)

    # Good jet filters
    if ak.count(events.FatJet_pt) != 0:
        events = sequences.apply_good_ak8_jet_filter(events)
    skimmer_utils.update_cut_flow(cut_flow, "GoodJetsAK8", events)

    # Removing events with no jets to avoid crashes
    filter_njets = ak.count(events.FatJet_pt, axis=1) > 0
    events = events[filter_njets]
    skimmer_utils.update_cut_flow(cut_flow, "nJetsAK8Gt0", events)
    
    # Adding JetsAK8_isGood branch already so that it can be used
    # in the rest of the pre-selection
    if len(events) != 0:
        events = sequences.add_good_ak8_jet_branch(events)
        events = sequences.add_good_ak4_jet_branch(events)
        events = sequences.add_veto_leptons_branches(events)

    # HEM issue filter: (apply to random events with same fraction as the lumi affected by HEM issue in data)
    # needs to have veto leptons and good ak4 jets already
    if len(events) != 0:
        good_ak4_jets = ak.zip({
            "eta": events.Jet_eta[events.Jet_isGood],
            "phi": events.Jet_phi[events.Jet_isGood],
        })
        veto_electrons = ak.zip({
            "eta": events.Electron_eta[events.Electron_isVeto],
            "phi": events.Electron_phi[events.Electron_isVeto],
        })
        veto_muons = ak.zip({
            "eta": events.Muon_eta[events.Muon_isVeto],
            "phi": events.Muon_phi[events.Muon_isVeto],
        })
        if year == "2018" and skimmer_utils.is_data(events):
            events = skimmer_utils.apply_hem_veto(events, good_ak4_jets, veto_electrons, veto_muons)
        if year == "2018" and skimmer_utils.is_mc(events):
            filter = skimmer_utils.get_hem_veto_filter(good_ak4_jets, veto_electrons, veto_muons)
            events["HEMVeto"] = filter
    skimmer_utils.update_cut_flow(cut_flow, "HEMVeto", events)

    # Requiring at least 2 good FatJets
    if len(events) != 0:
        filter = ak.count(events.FatJet_pt[events.FatJet_isGood], axis=1) >= 2
        events = events[filter]
    skimmer_utils.update_cut_flow(cut_flow, "nJetsAK8Gt2", events)

    # veto events with mini-isolated leptons (muons and electrons)
    if len(events) != 0:
        events = sequences.add_n_lepton_veto_branch(events)
        events = sequences.apply_isolated_lepton_veto(events)

    skimmer_utils.update_cut_flow(cut_flow, "IsolatedLeptonVeto", events)

    # MET filters — good primary vertex filter
    # Require PV_isValidVtx == 1, |PV_z| <= 24, sqrt(PV_x²+PV_y²) < 2
    if len(events) != 0:
        events = sequences.add_good_pv_branch(events)
        filter_good_pv = ak.sum(events.PV_isGood, axis=1) >= 1
        events = events[filter_good_pv]
    skimmer_utils.update_cut_flow(cut_flow, "goodVerticesFilter", events)
    
    # could implement experimental bad muon filters (official ones are not reproducible in scouting)

    # Phi spike filter: check subleading good AK8 jet vs dead cells
    if len(events) != 0:
        events = sequences.apply_scouting_phi_spike_filter(events, year)
    skimmer_utils.update_cut_flow(cut_flow, "PhiSpikeFilter", events)

    # Gap jet veto: veto events with high pt leading AK4 jets with high photon energy fraction (mostly important for data/MC in the tails)
    if len(events) != 0:
        events = sequences.apply_gap_jet_veto(events)
    skimmer_utils.update_cut_flow(cut_flow, "GapJetVeto", events)

    #apply RT filter (RT = MET over MT)
    if len(events) != 0:
        jets = skimmer_utils.make_pt_eta_phi_mass_lorentz_vector(
            pt=events.FatJet_pt[events.FatJet_isGood],
            eta=events.FatJet_eta[events.FatJet_isGood],
            phi=events.FatJet_phi[events.FatJet_isGood],
            mass=events.FatJet_mass[events.FatJet_isGood],
        )
        met = skimmer_utils.make_pt_eta_phi_mass_lorentz_vector(
            pt=events.ScoutMET_pt,
            phi=events.ScoutMET_phi,
        )
        mt = event_vars.calculate_transverse_mass(jets, met)
        rt = events.ScoutMET_pt / mt
        filter_rt = rt > 0.15
        filter_rt = as_type(filter_rt, bool)   #not needed
        events = events[filter_rt]
    
    skimmer_utils.update_cut_flow(cut_flow, "RT_selection", events)


    #apply DeltaEta filter (not applied because it is used as a control region)
    # if len(events) != 0:
    #     jets = skimmer_utils.make_pt_eta_phi_mass_lorentz_vector(
    #         pt=events.FatJet_pt[events.FatJet_isGood],
    #         eta=events.FatJet_eta[events.FatJet_isGood],
    #         phi=events.FatJet_phi[events.FatJet_isGood],
    #         mass=events.FatJet_mass[events.FatJet_isGood],
    #     )
    #     delta_eta = abs(event_vars.calculate_delta_eta(jets))
    #     filter_deltaeta = delta_eta < 1.5
    #     filter_deltaeta = as_type(filter_deltaeta, bool)
    #     events = events[filter_deltaeta]
    
    # skimmer_utils.update_cut_flow(cut_flow, "DeltaEtaj0j1 selection", events)

    #apply MT selection
    if len(events) != 0:
        jets = skimmer_utils.make_pt_eta_phi_mass_lorentz_vector(
            pt=events.FatJet_pt[events.FatJet_isGood],
            eta=events.FatJet_eta[events.FatJet_isGood],
            phi=events.FatJet_phi[events.FatJet_isGood],
            mass=events.FatJet_mass[events.FatJet_isGood],
        )
        met = skimmer_utils.make_pt_eta_phi_mass_lorentz_vector(
            pt=events.ScoutMET_pt,
            phi=events.ScoutMET_phi,
        )
        mt = event_vars.calculate_transverse_mass(jets, met)
        filter_mt = mt > 650
        filter_mt = as_type(filter_mt, bool)
        events = events[filter_mt]
    
    skimmer_utils.update_cut_flow(cut_flow, "MT_selection", events)

    # Delta phi min cut
    if len(events) != 0:
        # If needed because the selection crashes due to the special ak type
        met = skimmer_utils.make_pt_eta_phi_mass_lorentz_vector(
            pt=events.ScoutMET_pt,
            phi=events.ScoutMET_phi,
        )
        jets = skimmer_utils.make_pt_eta_phi_mass_lorentz_vector(
            pt=events.FatJet_pt[events.FatJet_isGood],
            eta=events.FatJet_eta[events.FatJet_isGood],
            phi=events.FatJet_phi[events.FatJet_isGood],
            mass=events.FatJet_mass[events.FatJet_isGood],
        )

        met = ak.broadcast_arrays(met, jets)[0]
        delta_phi_min = ak.min(abs(jets.delta_phi(met)), axis=1)
        filter_deltaphi = delta_phi_min < 0.8
        # Needed otherwise type is not defined and skim cannot be written
        filter_deltaphi = as_type(filter_deltaphi, bool)
        events = events[filter_deltaphi]

    skimmer_utils.update_cut_flow(cut_flow, "DeltaPhiMin_selection", events)
    
    if len(events) != 0:
        events = sequences.add_analysis_branches(events)
    if sequences.has_dark_quark_info(events):
        events = sequences.add_dark_quark_matching(events)
    events = sequences.remove_collections(events)

    return events, cut_flow

