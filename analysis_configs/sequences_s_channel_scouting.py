import numpy as np
import awkward as ak

from skimmer import skimmer_utils
from utils.variables_computation import event_variables as event_vars
from utils.data.triggers import primary_dataset_triggers
from utils.tree_maker.triggers import trigger_table
from utils.awkward_array_utilities import as_type
# from utils.variables_computation import jet_variables as jet_vars
# import analysis_configs.physics_objects_nano_aod as physicsObj
from analysis_configs import objects_definition_s_channel_scouting as obj

from utils.Logger import *


def apply_good_ak8_jet_filter(events):
    # events = add_branches_for_ak8_jet_id(events) # reads variables from ntuplizer now, no need to recompute them
    events = add_ak8_jet_id_branch(events)
    ak8_jets = ak.zip(
        {
            "pt": events.FatJet_pt,
            "mass": events.FatJet_mass,
            "eta": events.FatJet_eta,
            "phi": events.FatJet_phi,
            "id": events.FatJet_jetId,
        },
        with_name="PtEtaPhiMLorentzVector",
    )

    
    analysis_jets = ak8_jets[obj.is_analysis_ak8_jet(ak8_jets)]
    good_ak8_jets_filter = ak.all(obj.is_good_ak8_jet(analysis_jets), axis=1)
    events = events[good_ak8_jets_filter]
    return events


def add_ak8_jet_id_branch(events):
    CHM = events.FatJet_chargedMultiplicity
    NumConst = events.FatJet_nConstituents

    passID = ((abs(events.FatJet_eta) <= 2.6)
                   & (CHM > 0)
                   & (events.FatJet_chargedHadronEnergyFraction > 0)
                   & (events.FatJet_neutralHadronEnergyFraction < 0.9)
                   & (NumConst > 1)
                   & (events.FatJet_photonEnergyFraction < 0.9)
                   & (events.FatJet_muonEnergyFraction < 0.8)
                   & (events.FatJet_electronEnergyFraction < 0.8))

    passID = as_type(passID, int)

    events["FatJet_jetId"] = passID

    return events


# def __unflatten_to_event_level(jet_variable, n_jets):
#     arrays = [ak.unflatten(jet_variable, n_jets, axis=0)]
#     if len(arrays) == 1:
#         return arrays[0]
#     else:
#         return arrays
#
# def add_branches_for_ak8_jet_id(events):
#     jet_collection_name = "FatJet"
#     constituent_collection_name = "PFCands"
#     jets = physicsObj.get_jets(events, jet_collection_name)
#     jet_pf_cands = physicsObj.get_jet_pf_cands(events, jet_collection_name, constituent_collection_name)
#     jet_pf_cands_per_jet, _ = jet_vars.make_constituents_per_jet(jet_pf_cands, n_jets=ak.count(events[jet_collection_name+"_pt"], axis=-1))
#     flat_jets = ak.flatten(jets)
#
#     variables_fractions = ["chHEF", "neHEF", "electron_energy_fraction", "muon_energy_fraction", "photon_energy_fraction"]
#     for variable_name in variables_fractions:
#         function = getattr(jet_vars, f"calculate_{variable_name}")
#         variable = function(jet_pf_cands_per_jet, flat_jets)
#         branch_name = f"FatJet_{variable_name.replace('_','')}"
#         events[branch_name] = __unflatten_to_event_level(variable, n_jets=ak.count(events[jet_collection_name+"_pt"], axis=-1))
#
#     variables_multiplicities = ["electron_multiplicity", "muon_multiplicity", "chargedhadron_multiplicity"]
#     for variable_name in variables_multiplicities:
#         function = getattr(jet_vars, f"calculate_{variable_name}")
#         variable = function(jet_pf_cands_per_jet)
#         branch_name = f"{jet_collection_name}_{variable_name.replace('_','')}"
#         events[branch_name] = __unflatten_to_event_level(variable, n_jets=ak.count(events[jet_collection_name+"_pt"], axis=-1))
#
#     function = getattr(jet_vars, "calculate_multiplicity")
#     variable = function(jet_pf_cands_per_jet)
#     branch_name = f"{jet_collection_name}_multiplicity"
#     events[branch_name] = __unflatten_to_event_level(variable, n_jets=ak.count(events[jet_collection_name+"_pt"], axis=-1))
#     return events
#
# def add_ak8_jet_id_branch_from_pf_cands(events):
#     # Old approach: recompute ID variables from PF candidates
#     events = add_branches_for_ak8_jet_id(events)
#     CHM = events.FatJet_chargedhadronmultiplicity + events.FatJet_electronmultiplicity + events.FatJet_muonmultiplicity
#     passID = ((abs(events.FatJet_eta) <= 2.6)
#                    & (CHM > 0)
#                    & (events.FatJet_chHEF > 0)
#                    & (events.FatJet_neHEF < 0.9)
#                    & (events.FatJet_multiplicity > 1)
#                    & (events.FatJet_photonenergyfraction < 0.9)
#                    & (events.FatJet_muonenergyfraction < 0.8)
#                    & (events.FatJet_electronenergyfraction < 0.8))
#     passID = as_type(passID, int)
#     events["FatJet_jetId"] = passID
#     return events


def add_ak4_jet_id_branch(events):
    CHM = events.Jet_chargedMultiplicity
    NumConst = events.Jet_nConstituents

    passID = ((abs(events.Jet_eta) <= 2.6)
                   & (CHM > 0)
                   & (events.Jet_chargedHadronEnergyFraction > 0)
                   & (events.Jet_neutralHadronEnergyFraction < 0.9)
                   & (NumConst > 1)
                   & (events.Jet_photonEnergyFraction < 0.9)
                   & (events.Jet_muonEnergyFraction < 0.8)
                   & (events.Jet_electronEnergyFraction < 0.8))

    passID = as_type(passID, int)

    events["Jet_jetId"] = passID

    return events

def _compute_electron_id(events):
    # Veto electron ID based on offline, had to remove relIso and pass conversion ratio requirements
    # Barrel: |eta| < 1.479, Endcap: 1.479 < |eta| < 2.5
    is_barrel = abs(events["Electron_eta"]) < 1.479

    sieie_cut = ak.where(is_barrel, 0.015, 0.045)
    hoe_cut = 0.2
    detain_cut = ak.where(is_barrel, 0.008, 0.012)
    dphiin_cut = 0.06
    ecaliso_cut = ak.where(is_barrel, 0.25, 0.1)
    trkiso_cut = 0.001
    hcaliso_cut = ak.where(is_barrel, 0.4, 0.6)
    #ooEMOop_cut = ak.where(is_barrel, 0.209, 0.132)
    #mhits_cut = ak.where(is_barrel, 2, 3)

    electron_energy = events["Electron_pt"] * np.cosh(events["Electron_eta"])

    passID = (
        (events["Electron_sieie"] < sieie_cut)
        & (abs(events["Electron_detain"]) < detain_cut)
        & (abs(events["Electron_dphiin"]) < dphiin_cut)
        & (events["Electron_hoe"] < hoe_cut)
        & (events["Electron_ecaliso"] / electron_energy < ecaliso_cut)
        & (events["Electron_hcaliso"] / electron_energy < hcaliso_cut)
        & (events["Electron_trkiso"] / electron_energy < trkiso_cut)
        #& (abs(events["Electron_ooEMOop"]) < ooEMOop_cut)
        #& (events["Electron_mHits"] <= mhits_cut)
    )
    return as_type(passID, int)


def _compute_muon_id(events):
    # Loose muon ID: global muon or tracker muon
    passID = (
        events["Muon_isGlobal"]
        | events["Muon_isTracker"]
    )
    return as_type(passID, int)


def add_good_ak8_jet_branch(events):
    ak8_jets = ak.zip(
        {
            "pt": events.FatJet_pt,
            "mass": events.FatJet_mass,
            "eta": events.FatJet_eta,
            "phi": events.FatJet_phi,
            "id": events.FatJet_jetId,
        },
        with_name="PtEtaPhiMLorentzVector",
    )
    
    is_good_analysis_ak8_jet = (
        obj.is_analysis_ak8_jet(ak8_jets)
        & obj.is_good_ak8_jet(ak8_jets)
    )

    #add new branch to the events
    events["FatJet_isGood"] = is_good_analysis_ak8_jet

    return events

def add_good_ak4_jet_branch(events):
    events = add_ak4_jet_id_branch(events)
    ak4_jets = ak.zip(
        {
            "pt": events.Jet_pt,
            "mass": events.Jet_mass,
            "eta": events.Jet_eta,
            "phi": events.Jet_phi,
            "id": events.Jet_jetId,
        },
        with_name="PtEtaPhiMLorentzVector",
    )
    
    is_good_analysis_ak4_jet = (
        obj.is_good_ak4_jet(ak4_jets)
        & obj.is_analysis_ak4_jet(ak4_jets)
    )

    #add new branch to the events
    events["Jet_isGood"] = is_good_analysis_ak4_jet

    return events


def add_good_pv_branch(events):
    is_good_pv = (
        (events.PV_isValidVtx == 1)
        & (abs(events.PV_z) <= 24)
        & ((events.PV_x**2 + events.PV_y**2) < 4)
    )
    events["PV_isGood"] = as_type(is_good_pv, int)
    return events


def add_veto_leptons_branches(events):
    electrons, muons = _build_scouting_lepton_collections(events)
    if electrons is None or muons is None:
        print("Warning: missing electron or muon collections, cannot compute veto lepton branches")
        n_events = ak.num(events.FatJet_pt, axis=0)
        events["Electron_isVeto"] = as_type(ak.Array([[]] * n_events), int)
        events["Muon_isVeto"] = as_type(ak.Array([[]] * n_events), int)
    else:
        events["Electron_isVeto"] = as_type(obj.is_veto_electron(electrons), int)
        events["Muon_isVeto"] = as_type(obj.is_veto_muon(muons), int)
    return events


def _build_scouting_lepton_collections(events):
    required_fields = [
        "Electron_pt",
        "Electron_eta",
        "Electron_combinedMiniIso",
        "Electron_sieie",
        "Electron_detain",
        "Electron_dphiin",
        "Electron_hoe",
        "Electron_ecaliso",
        "Electron_hcaliso",
        "Electron_trkiso",
        #"Electron_ooEMOop",
        #"Electron_mHits",
        "rho",
        "Muon_pt",
        "Muon_eta",
        "Muon_combinedMiniIso",
        "Muon_isGlobal",
        "Muon_isTracker",
    ]

    if any(field not in events.fields for field in required_fields):
        return None, None

    electrons = ak.zip(
        {
            "pt": events["Electron_pt"],
            "eta": events["Electron_eta"],
            "iso": events["Electron_combinedMiniIso"],
            "id": _compute_electron_id(events),
        }
    )

    muons = ak.zip(
        {
            "pt": events["Muon_pt"],
            "eta": events["Muon_eta"],
            "iso": events["Muon_combinedMiniIso"],
            "id": _compute_muon_id(events),
        }
    )

    return electrons, muons

def __get_number_of_veto_leptons(
        events,
        electron_extra_condition=None,
        muon_extra_condition=None,
    ):

    electrons, muons = _build_scouting_lepton_collections(events)
    if electrons is None or muons is None:
        return ak.zeros_like(ak.num(events.FatJet_pt, axis=1), dtype=np.int64)

    electron_condition = obj.is_veto_electron(electrons)
    if electron_extra_condition is not None:
        electron_condition = electron_condition & electron_extra_condition

    muon_condition = obj.is_veto_muon(muons)
    if muon_extra_condition is not None:
        muon_condition = muon_condition & muon_extra_condition

    veto_electrons = electrons[electron_condition]
    veto_muons = muons[muon_condition]
    n_veto_electrons = ak.count(veto_electrons.pt, axis=1)
    n_veto_muons = ak.count(veto_muons.pt, axis=1)
    n_veto_leptons = n_veto_electrons + n_veto_muons

    return n_veto_leptons


def add_n_lepton_veto_branch(events):
    n_veto_leptons = __get_number_of_veto_leptons(
        events,
        electron_extra_condition=None,
        muon_extra_condition=None,
    )

    events = ak.with_field(
        events,
        n_veto_leptons,
        "nVetoLeptons",
    )

    return events
 

def apply_isolated_lepton_veto(
        events,
        electron_extra_condition=None,
        muon_extra_condition=None,
        revert=False,
    ):
    n_veto_leptons = __get_number_of_veto_leptons(
        events,
        electron_extra_condition=electron_extra_condition,
        muon_extra_condition=muon_extra_condition,
    )

    if revert:
        filter = (n_veto_leptons != 0)
    else:
        filter = (n_veto_leptons == 0)
    events = events[filter]

    return events

def add_analysis_branches(events):

    # Event variables
    good_jets_ak8_lv = skimmer_utils.make_pt_eta_phi_mass_lorentz_vector(
        pt=events.FatJet_pt[events.FatJet_isGood],
        eta=events.FatJet_eta[events.FatJet_isGood],
        phi=events.FatJet_phi[events.FatJet_isGood],
        mass=events.FatJet_mass[events.FatJet_isGood],
    )
    met = skimmer_utils.make_pt_eta_phi_mass_lorentz_vector(
            pt=events.ScoutMET_pt,
            phi=events.ScoutMET_phi,
    )

    #add rt variable
    mt = event_vars.calculate_transverse_mass(good_jets_ak8_lv, met)
    rt = met.pt / mt
    events["RTFatJet"] = rt

    #add mt variable
    events["MT01FatJetMET"] = mt

    #add deltaeta the two leading jets
    events["DeltaEtaJ0J1FatJet"] = event_vars.calculate_delta_eta(good_jets_ak8_lv)
    
    #add minimum delta phi between the MET and the two leading jets
    events["DeltaPhiMinFatJetMET"] = event_vars.calculate_delta_phi_min(good_jets_ak8_lv, met)
    
    return events


def has_dark_quark_info(events):
    return "MatrixElementGenParticle_pt" in events.fields


def add_dark_quark_matching(events):
    """
    performs delta R matching between AK8 jets and dark quarks and saves the indices of the matched jets in a new branch
    """

    jets = skimmer_utils.make_pt_eta_phi_mass_lorentz_vector(
        pt=events.FatJet_pt,
        eta=events.FatJet_eta,
        phi=events.FatJet_phi,
        mass=events.FatJet_mass,
    )

    dark_quarks =  skimmer_utils.make_pt_eta_phi_mass_lorentz_vector(
        pt=events.MatrixElementGenParticle_pt,
        eta=events.MatrixElementGenParticle_eta,
        phi=events.MatrixElementGenParticle_phi,
        mass=events.MatrixElementGenParticle_mass,
    )

    # delta R matching between jets and dark quarks
    dR1 = jets.delta_r(dark_quarks[:,0]) <= 0.8
    dR2 = jets.delta_r(dark_quarks[:,1]) <= 0.8
    matched_mask = dR1 | dR2

    #add new branch to the event tree
    events["FatJet_DarkQuarkMatched"] = ak.values_astype(matched_mask, "uint32")
    
    return events


def apply_scouting_phi_spike_filter(events, year):
    """Phi spike filter using only the subleading good AK4 jet.
    Bin centers and radius are read from data/dead_cells/outlier_centers_{year}.txt.
    The distance check follows the convention dist² < rad (consistent with apply_phi_spike_filter).
    """
    import os

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "dead_cells"
    )
    filepath = os.path.join(data_dir, f"outlier_centers_{year}.txt")

    rad = 0.028816 * 0.35  # half-diagonal of eta-phi cell, factor 0.35 optimized for s/b sensitivity

    eta_centers = []
    phi_centers = []
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            parts = stripped.split()
            eta_centers.append(float(parts[0]))
            phi_centers.append(float(parts[1]))

    eta_arr = np.array(eta_centers)
    phi_arr = np.array(phi_centers)

    # Subleading good AK4 jet (index 1)
    good_jet_eta = events.Jet_eta[events.Jet_isGood]
    good_jet_phi = events.Jet_phi[events.Jet_isGood]
    sub_eta = ak.to_numpy(ak.fill_none(ak.pad_none(good_jet_eta, 2)[:, 1], np.inf))
    sub_phi = ak.to_numpy(ak.fill_none(ak.pad_none(good_jet_phi, 2)[:, 1], np.inf))

    # Reject events where subleading jet falls within rad of any dead cell center
    # Convention: dist² < rad (same as apply_phi_spike_filter)
    deta = sub_eta[:, np.newaxis] - eta_arr[np.newaxis, :]
    dphi = sub_phi[:, np.newaxis] - phi_arr[np.newaxis, :]
    dphi = np.where(dphi >  np.pi, dphi - 2.0 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2.0 * np.pi, dphi)
    in_spike = np.any(deta**2 + dphi**2 < rad, axis=1)

    events = events[~in_spike]
    return events

def apply_gap_jet_veto(events):
    # Veto events with high pt AK4 jets with high photon energy fraction, mostly occuring in the gap between barrel and endcap

    ak4_jets = ak.zip(
        {
            "pt": events.Jet_pt,
            "eta": events.Jet_eta,
            "phi": events.Jet_phi,
            "mass": events.Jet_mass,
            "photonEnergyFraction": events.Jet_photonEnergyFraction,
        },
        with_name="PtEtaPhiMLorentzVector",
    )

    lead_jet = ak.pad_none(ak4_jets, 1)[:, 0]
    veto_mask = (
        (ak.fill_none(lead_jet.pt, 0) > 1000)
        & (ak.fill_none(lead_jet.photonEnergyFraction, 0) > 0.7)
    )
    return events[~veto_mask]

def remove_collections(events):
    #remove branch called genModel, hltResultName
    list_branches_to_remove = ["genModel", "hltResultName"]
    list_collections_to_remove = ["Off", "nOff", "FatJetDarkHadronsubJets", "GenFatJetDarkHadrons"] 
    for collection in list_collections_to_remove:
        branches_to_remove = [field for field in events.fields if field.startswith(collection)]
        list_branches_to_remove.extend(branches_to_remove)
    #print(f"Removing the following branches from the output file: {list_branches_to_remove}")
    events = events[[key for key in events.fields if key not in list_branches_to_remove]]
    
    return events
