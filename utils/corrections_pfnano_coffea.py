#import relevant libraries
import correctionlib
import os
import awkward as ak
import argparse

from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory
from coffea.util import save

from datetime import datetime
import requests
from bs4 import BeautifulSoup

now = datetime.now()


#define path for data directory
path_current_file = os.path.dirname(os.path.abspath(__file__))
out_path_corrections = path_current_file.replace("utils","data")



def __get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-jme', '--include_jme_corrections',
        help='Set True if input files are (PF)NanoAOD files', 
        default=False, 
        action='store_true',
    )

    parser.add_argument(
        '-pu', '--include_pu_corrections',
        help='Set True if input files are (PF)NanoAOD files', 
        default=False, 
        action='store_true',
    )

    parser.add_argument(
        '-all', '--include_all_corrections',
        help='Set True if input files are (PF)NanoAOD files', 
        default=False, 
        action='store_true',
    )
    
    parser.add_argument(
        '-met', '--include_met_corrections',
        help='Include MET phi (XY) corrections (separate from JME)',
        default=False,
        action='store_true',
    )

    return parser.parse_args()


def download_raw_files(url,output_dir):
    
    print(f"Downloading raw files from {url}...")

    # Create a directory to store the downloaded files BEFORE downloading
    os.makedirs(output_dir, exist_ok=True)

    # Convert GitHub web URL to API URL
    # Example: https://github.com/mcremone/decaf/tree/UL/analysis/data/jerc
    # becomes: https://api.github.com/repos/mcremone/decaf/contents/analysis/data/jerc?ref=UL
    
    if 'github.com' in url and '/tree/' in url:
        parts = url.replace('https://github.com/', '').split('/')
        owner = parts[0]
        repo = parts[1]
        branch = parts[3]
        path = '/'.join(parts[4:])
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
        
        try:
            # Get directory listing from GitHub API
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            files = response.json()
            
            # Download each file that ends with .txt or .gz
            # Skip non-correction files like Source.txt, README.txt
            for file_info in files:
                file_name = file_info['name']
                
                # Skip non-correction files
                if file_name in ['Source.txt', 'README.txt', 'README.md']:
                    continue
                    
                if file_info['name'].endswith('.txt') or file_info['name'].endswith('.gz'):
                    file_path = os.path.join(output_dir, file_name)
                    
                    # Determine the final filename with proper extension
                    # Match old version conventions: .jec.txt, .junc.txt, .jr.txt, .jersf.txt
                    final_name = file_name
                    if file_name.endswith('.txt'):
                        if '_Uncertainty_' in file_name and '_UncertaintySources_' not in file_name:
                            # Uncertainty files -> .junc.txt
                            final_name = file_name.replace('.txt', '.junc.txt')
                        elif '_UncertaintySources_' in file_name:
                            # UncertaintySources files -> .junc.txt
                            final_name = file_name.replace('.txt', '.junc.txt')
                        elif '_PtResolution_' in file_name or '_EtaResolution_' in file_name or '_PhiResolution_' in file_name:
                            # Resolution files -> .jr.txt
                            final_name = file_name.replace('.txt', '.jr.txt')
                        elif '_SF_' in file_name:
                            # Scale factor files -> .jersf.txt
                            final_name = file_name.replace('.txt', '.jersf.txt')
                        elif any(x in file_name for x in ['_L1FastJet_', '_L2Relative_', '_L3Absolute_', '_L2L3Residual_', '_L1RC_', '_L2Residual_']):
                            # JEC correction files -> .jec.txt
                            final_name = file_name.replace('.txt', '.jec.txt')
                    
                    final_path = os.path.join(output_dir, final_name)
                    
                    # Check if final file already exists (with renamed extension)
                    if os.path.exists(final_path):
                        print(f"{final_name} already exists in {output_dir}")
                        continue
                    
                    # Also check if original name exists
                    if os.path.exists(file_path) and final_name != file_name:
                        print(f"{file_name} already exists in {output_dir}")
                        continue
                    
                    # Use the download_url from GitHub API
                    download_url = file_info['download_url']
                    
                    print(f"Downloading {file_name}...")
                    
                    try:
                        # Download the file
                        file_response = requests.get(download_url, timeout=30)
                        file_response.raise_for_status()
                        
                        # Write the content to file with final name
                        with open(final_path, 'wb') as f:
                            f.write(file_response.content)
                        
                        if final_name != file_name:
                            print(f"{file_name} downloaded and renamed to {final_name}")
                        else:
                            print(f"{file_name} downloaded successfully.")
                        
                    except requests.exceptions.RequestException as e:
                        print(f"Error downloading {file_name}: {e}")
                        continue
                        
        except requests.exceptions.RequestException as e:
            print(f"Error accessing GitHub API: {e}")
            print("Falling back to direct download method...")
            # Fallback: try direct URLs if we know the branch
            return
    else:
        print(f"URL format not recognized: {url}")
        return



def fetch_and_save_files_corrections(pog,observable,year=None,version=None):

    path_saved_corrections = ""
    
    if pog == 'jme':
        
        #path where to save the corrections
        path_saved_corrections = f"{out_path_corrections}/jme/"
        
        if observable == 'met':
            
            #check if file met.json.gz is already in the directory
            if os.path.exists(f"{path_saved_corrections}/met/{year}_UL/met.json.gz"):
                print(f"File met.json.gz already exists in {path_saved_corrections}/met/{year}_UL")
                return path_saved_corrections
                
            #get the file met.json.gz from https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME, save it in out_path_corrections/jme/year_UL
            download_raw_files(url=f"https://github.com/mcremone/decaf/tree/UL/analysis/data/JetMETCorr/{year}_UL", output_dir=path_saved_corrections + f"/met/{year}_UL")

        if observable == 'jec':
            # JEC corrections from JECDatabase - download specific version only
            if version:
                url = f"https://github.com/cms-jet/JECDatabase/tree/master/textFiles/{version}"
            else:
                url = "https://github.com/cms-jet/JECDatabase/tree/master/textFiles"
            download_raw_files(url=url, output_dir=path_saved_corrections + "/jerc")

        if observable == 'jer':
            # JER corrections from JRDatabase - download specific version only
            if version:
                url = f"https://github.com/cms-jet/JRDatabase/tree/master/textFiles/{version}"
            else:
                url = "https://github.com/cms-jet/JRDatabase/tree/master/textFiles"
            download_raw_files(url=url, output_dir=path_saved_corrections + "/jerc")


    if pog == 'lum':

        #path where to save the corrections
        path_saved_corrections = f"{out_path_corrections}/lum/"

        #if f"{path_saved_corrections}/{year}_UL/ is not already created, create it
        os.makedirs(f"{path_saved_corrections}/{year}_UL", exist_ok=True)

        if observable == 'pu':

            #check if file puWeights.json.gz is already in the directory
            if os.path.exists(f"{path_saved_corrections}/{year}_UL/puWeights.json.gz"):
                print(f"File puWeights.json.gz already exists in {path_saved_corrections}/{year}_UL")
                return path_saved_corrections


            #get the file puWeights.json.gz from https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM, save it in out_path_corrections/lum/year_UL
            download_raw_files(url=f"https://github.com/mcremone/decaf/tree/UL/analysis/data/PUweight/{year}_UL", output_dir=path_saved_corrections + f"/{year}_UL/")

    return path_saved_corrections



####
# PU weight
# https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM
####
def get_pu_weight(year, trueint):
    correction = {'2018': 'Collisions18_UltraLegacy_goldenJSON',
                  '2017': 'Collisions17_UltraLegacy_goldenJSON',
                  '2016APV': 'Collisions16_UltraLegacy_goldenJSON',
                  '2016':'Collisions16_UltraLegacy_goldenJSON'}
    evaluator = correctionlib.CorrectionSet.from_file(f'{out_path_corrections}/lum/'+year+'_UL/puWeights.json.gz')
    weight_nom = evaluator[correction[year]].evaluate(trueint, 'nominal')
    weight_up = evaluator[correction[year]].evaluate(trueint, 'up')
    weight_down = evaluator[correction[year]].evaluate(trueint, 'down')

    return weight_nom, weight_up, weight_down



def XY_MET_Correction(year, npv, run, pt, phi, isData):
    
    npv = ak.where((npv>200),ak.full_like(npv,200),npv)
    pt  = ak.where((pt>1000.),ak.full_like(pt,1000.),pt)

    if '2016preVFP' in year:
        #run = ak.where((run<271036),ak.full_like(run,271036),run)
        run = ak.where((run>278771),ak.full_like(run,278771),run)
    if '2016postVFP' in year:
        #run = ak.where((run<271036),ak.full_like(run,271036),run)
        run = ak.where((run>284045),ak.full_like(run,284045),run)
    if '2017' in year:
        #run = ak.where((run<294927),ak.full_like(run,294927),run)
        run = ak.where((run>306463),ak.full_like(run,306463),run)
    if '2018' in year:
        #run = ak.where((run<314472),ak.full_like(run,314472),run)
        run = ak.where((run>325274),ak.full_like(run,325274),run)
        
    
    evaluator = correctionlib.CorrectionSet.from_file(f'{out_path_corrections}/jme/met/{year}_UL/met.json.gz')

    if isData:
        corrected_pt = evaluator['pt_metphicorr_pfmet_data'].evaluate(pt,phi,npv,run)
        corrected_phi = evaluator['phi_metphicorr_pfmet_data'].evaluate(pt,phi,npv,run)

    if not isData:
        corrected_pt = evaluator['pt_metphicorr_pfmet_mc'].evaluate(pt,phi,npv,run)
        corrected_phi = evaluator['phi_metphicorr_pfmet_mc'].evaluate(pt,phi,npv,run)

    return corrected_pt, corrected_phi



def jet_factory_factory(files,jec_name_map):
    ext = extractor()
    directory=f'{out_path_corrections}/jme/jerc'
    for filename in files:
        ext.add_weight_sets([f"* * {directory+'/'+filename}"])
    ext.finalize()
    jec_stack = JECStack(ext.make_evaluator())
    return CorrectedJetsFactory(jec_name_map, jec_stack)


def build_jet_and_corrections():

    # Define which JEC/JER versions we need (only these will be downloaded)
    jec_versions = [
        "Summer16_07Aug2017_V11_MC",  # For 2016
        #"Summer16_07Aug2017_V11_L1fix_MC", # for some early production campaigns
        "Fall17_17Nov2017_V32_MC",  # For 2017
        "Autumn18_V19_MC",  # For 2018
        # Data versions (needed for L2L3Residuals)
        "Summer16_07Aug2017BCD_V11_DATA",  # 2016 preVFP data
        "Summer16_07Aug2017EF_V11_DATA",   # 2016 postVFP data
        "Summer16_07Aug2017GH_V11_DATA",      # 2016 postVFP data
        "Fall17_17Nov2017B_V32_DATA",  # 2017 data
        "Fall17_17Nov2017C_V32_DATA",  # 2017 data
        "Fall17_17Nov2017DE_V32_DATA",  # 2017 data
        "Fall17_17Nov2017F_V32_DATA",  # 2017 data
        "Autumn18_RunA_V19_DATA",  # 2018 data
        "Autumn18_RunB_V19_DATA",  # 2018 data
        "Autumn18_RunC_V19_DATA",  # 2018 data
        "Autumn18_RunD_V19_DATA",  # 2018 data
    ]
    
    jer_versions = [
        "Summer16_25nsV1_MC",  # 2016
        "Fall17_V3_MC",  # For 2017
        "Autumn18_V7_MC",  # For 2018
    ]
    
    # Fetch only the specific JEC versions we need
    for version in jec_versions:
        fetch_and_save_files_corrections(pog='jme', observable='jec', version=version)
    
    # Fetch only the specific JER versions we need
    for version in jer_versions:
        fetch_and_save_files_corrections(pog='jme', observable='jer', version=version)

    jec_name_map = {
    'JetPt': 'pt',
    'JetMass': 'mass',
    'JetEta': 'eta',
    'JetA': 'area',
    'ptGenJet': 'pt_gen',
    'ptRaw': 'pt_raw',
    'massRaw': 'mass_raw',
    'Rho': 'event_rho',
    'METpt': 'pt',
    'METphi': 'phi',
    'JetPhi': 'phi',
    'UnClusteredEnergyDeltaX': 'MetUnclustEnUpDeltaX',
    'UnClusteredEnergyDeltaY': 'MetUnclustEnUpDeltaY',
    }


    jet_factory = {
        "2016preVFPmc": jet_factory_factory(
            files=[
                "Summer16_07Aug2017_V11_MC_L1FastJet_AK4PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L2Relative_AK4PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L3Absolute_AK4PF.jec.txt",
                # NOTE: UncertaintySources files have incompatible format (start with # comment)
                "Summer16_07Aug2017_V11_MC_UncertaintySources_AK4PF.junc.txt",
                "Summer16_07Aug2017_V11_MC_Uncertainty_AK4PF.junc.txt",
                "Summer16_25nsV1_MC_PtResolution_AK4PF.jr.txt",
                "Summer16_25nsV1_MC_SF_AK4PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2016preVFPmcNOJER": jet_factory_factory(
            files=[
                "Summer16_07Aug2017_V11_MC_L1FastJet_AK4PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L2Relative_AK4PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L3Absolute_AK4PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_Uncertainty_AK4PF.junc.txt",  # Incompatible format
            ],
            jec_name_map=jec_name_map,
        ),
        "2016preVFPmcNOJEC": jet_factory_factory(
            files=[
                "Summer16_25nsV1_MC_PtResolution_AK4PF.jr.txt",
                "Summer16_25nsV1_MC_SF_AK4PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2016postVFPmc": jet_factory_factory(
            files=[
                "Summer16_07Aug2017_V11_MC_L1FastJet_AK4PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L2Relative_AK4PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L3Absolute_AK4PF.jec.txt",
                # NOTE: Uncertainty files have incompatible format (start with # comment)
                "Summer16_07Aug2017_V11_MC_UncertaintySources_AK4PF.junc.txt",
                "Summer16_07Aug2017_V11_MC_Uncertainty_AK4PF.junc.txt",
                "Summer16_25nsV1_MC_PtResolution_AK4PF.jr.txt",
                "Summer16_25nsV1_MC_SF_AK4PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2016postVFPmcNOJER": jet_factory_factory(
            files=[
                "Summer16_07Aug2017_V11_MC_L1FastJet_AK4PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L2Relative_AK4PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L3Absolute_AK4PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_Uncertainty_AK4PF.junc.txt",  # Incompatible format
            ],
            jec_name_map=jec_name_map,
        ),
        "2016postVFPmcNOJEC": jet_factory_factory(
            files=[
                "Summer16_25nsV1_MC_PtResolution_AK4PF.jr.txt",
                "Summer16_25nsV1_MC_SF_AK4PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017mc": jet_factory_factory(
            files=[
                "Fall17_17Nov2017_V32_MC_L1FastJet_AK4PF.jec.txt",
                "Fall17_17Nov2017_V32_MC_L2Relative_AK4PF.jec.txt",
                "Fall17_17Nov2017_V32_MC_L3Absolute_AK4PF.jec.txt",
                # NOTE: Uncertainty files have incompatible format (start with # comment)
                "Fall17_17Nov2017_V32_MC_UncertaintySources_AK4PF.junc.txt",
                "Fall17_17Nov2017_V32_MC_Uncertainty_AK4PF.junc.txt",
                "Fall17_V3_MC_PtResolution_AK4PF.jr.txt",
                "Fall17_V3_MC_SF_AK4PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017mcNOJER": jet_factory_factory(
            files=[
                "Fall17_17Nov2017_V32_MC_L1FastJet_AK4PF.jec.txt",
                "Fall17_17Nov2017_V32_MC_L2Relative_AK4PF.jec.txt",
                "Fall17_17Nov2017_V32_MC_L3Absolute_AK4PF.jec.txt",
                "Fall17_17Nov2017_V32_MC_Uncertainty_AK4PF.junc.txt",  # Incompatible format
            ],
            jec_name_map=jec_name_map,
        ),
        "2017mcNOJEC": jet_factory_factory(
            files=[
                "Fall17_V3_MC_PtResolution_AK4PF.jr.txt",
                "Fall17_V3_MC_SF_AK4PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018mc": jet_factory_factory(
            files=[
                "Autumn18_V19_MC_L1FastJet_AK4PF.jec.txt",
                "Autumn18_V19_MC_L2Relative_AK4PF.jec.txt",
                "Autumn18_V19_MC_L3Absolute_AK4PF.jec.txt",
                # NOTE: Uncertainty files have incompatible format (parsed as jme_standard_function, not jec_uncertainty_lookup)
                "Autumn18_V19_MC_UncertaintySources_AK4PF.junc.txt",
                "Autumn18_V19_MC_Uncertainty_AK4PF.junc.txt",
                "Autumn18_V7_MC_PtResolution_AK4PF.jr.txt",
                "Autumn18_V7_MC_SF_AK4PF.jersf.txt",
                #"Summer19UL18_V5_MC_L1FastJet_AK4PFchs.jec.txt",
                #"Summer19UL18_V5_MC_L2Relative_AK4PFchs.jec.txt",
                #"Summer19UL18_V5_MC_UncertaintySources_AK4PFchs.junc.txt",
                #"Summer19UL18_V5_MC_Uncertainty_AK4PFchs.junc.txt",
                #"Summer19UL18_JRV2_MC_PtResolution_AK4PFchs.jr.txt",
                #"Summer19UL18_JRV2_MC_SF_AK4PFchs.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018mcNOJER": jet_factory_factory(
            files=[
                "Autumn18_V19_MC_L1FastJet_AK4PF.jec.txt",
                "Autumn18_V19_MC_L2Relative_AK4PF.jec.txt",
                "Autumn18_V19_MC_L3Absolute_AK4PF.jec.txt",
                # NOTE: Uncertainty files have incompatible format (parsed as jme_standard_function, not jec_uncertainty_lookup)
                "Autumn18_V19_MC_Uncertainty_AK4PF.junc.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018mcNOJEC": jet_factory_factory(
            files=[
                "Autumn18_V7_MC_PtResolution_AK4PF.jr.txt",
                "Autumn18_V7_MC_SF_AK4PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        # Data entries (L1FastJet + L2Relative + L3Absolute + L2L3Residual, no JER)
        "2016preVFPdataBCDNOJER": jet_factory_factory(
            files=[
                "Summer16_07Aug2017BCD_V11_DATA_L1FastJet_AK4PF.jec.txt",
                "Summer16_07Aug2017BCD_V11_DATA_L2Relative_AK4PF.jec.txt",
                "Summer16_07Aug2017BCD_V11_DATA_L3Absolute_AK4PF.jec.txt",
                "Summer16_07Aug2017BCD_V11_DATA_L2L3Residual_AK4PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2016postVFPdataEFNOJER": jet_factory_factory(
            files=[
                "Summer16_07Aug2017EF_V11_DATA_L1FastJet_AK4PF.jec.txt",
                "Summer16_07Aug2017EF_V11_DATA_L2Relative_AK4PF.jec.txt",
                "Summer16_07Aug2017EF_V11_DATA_L3Absolute_AK4PF.jec.txt",
                "Summer16_07Aug2017EF_V11_DATA_L2L3Residual_AK4PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2016postVFPdataGHNOJER": jet_factory_factory(
            files=[
                "Summer16_07Aug2017GH_V11_DATA_L1FastJet_AK4PF.jec.txt",
                "Summer16_07Aug2017GH_V11_DATA_L2Relative_AK4PF.jec.txt",
                "Summer16_07Aug2017GH_V11_DATA_L3Absolute_AK4PF.jec.txt",
                "Summer16_07Aug2017GH_V11_DATA_L2L3Residual_AK4PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017dataBNOJER": jet_factory_factory(
            files=[
                "Fall17_17Nov2017B_V32_DATA_L1FastJet_AK4PF.jec.txt",
                "Fall17_17Nov2017B_V32_DATA_L2Relative_AK4PF.jec.txt",
                "Fall17_17Nov2017B_V32_DATA_L3Absolute_AK4PF.jec.txt",
                "Fall17_17Nov2017B_V32_DATA_L2L3Residual_AK4PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017dataCNOJER": jet_factory_factory(
            files=[
                "Fall17_17Nov2017C_V32_DATA_L1FastJet_AK4PF.jec.txt",
                "Fall17_17Nov2017C_V32_DATA_L2Relative_AK4PF.jec.txt",
                "Fall17_17Nov2017C_V32_DATA_L3Absolute_AK4PF.jec.txt",
                "Fall17_17Nov2017C_V32_DATA_L2L3Residual_AK4PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017dataDENOJER": jet_factory_factory(
            files=[
                "Fall17_17Nov2017DE_V32_DATA_L1FastJet_AK4PF.jec.txt",
                "Fall17_17Nov2017DE_V32_DATA_L2Relative_AK4PF.jec.txt",
                "Fall17_17Nov2017DE_V32_DATA_L3Absolute_AK4PF.jec.txt",
                "Fall17_17Nov2017DE_V32_DATA_L2L3Residual_AK4PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017dataFNOJER": jet_factory_factory(
            files=[
                "Fall17_17Nov2017F_V32_DATA_L1FastJet_AK4PF.jec.txt",
                "Fall17_17Nov2017F_V32_DATA_L2Relative_AK4PF.jec.txt",
                "Fall17_17Nov2017F_V32_DATA_L3Absolute_AK4PF.jec.txt",
                "Fall17_17Nov2017F_V32_DATA_L2L3Residual_AK4PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018dataRunANOJER": jet_factory_factory(
            files=[
                "Autumn18_RunA_V19_DATA_L1FastJet_AK4PF.jec.txt",
                "Autumn18_RunA_V19_DATA_L2Relative_AK4PF.jec.txt",
                "Autumn18_RunA_V19_DATA_L3Absolute_AK4PF.jec.txt",
                "Autumn18_RunA_V19_DATA_L2L3Residual_AK4PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018dataRunBNOJER": jet_factory_factory(
            files=[
                "Autumn18_RunB_V19_DATA_L1FastJet_AK4PF.jec.txt",
                "Autumn18_RunB_V19_DATA_L2Relative_AK4PF.jec.txt",
                "Autumn18_RunB_V19_DATA_L3Absolute_AK4PF.jec.txt",
                "Autumn18_RunB_V19_DATA_L2L3Residual_AK4PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018dataRunCNOJER": jet_factory_factory(
            files=[
                "Autumn18_RunC_V19_DATA_L1FastJet_AK4PF.jec.txt",
                "Autumn18_RunC_V19_DATA_L2Relative_AK4PF.jec.txt",
                "Autumn18_RunC_V19_DATA_L3Absolute_AK4PF.jec.txt",
                "Autumn18_RunC_V19_DATA_L2L3Residual_AK4PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018dataRunDNOJER": jet_factory_factory(
            files=[
                "Autumn18_RunD_V19_DATA_L1FastJet_AK4PF.jec.txt",
                "Autumn18_RunD_V19_DATA_L2Relative_AK4PF.jec.txt",
                "Autumn18_RunD_V19_DATA_L3Absolute_AK4PF.jec.txt",
                "Autumn18_RunD_V19_DATA_L2L3Residual_AK4PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
    }

    
    fatjet_factory = {
        "2016preVFPmc": jet_factory_factory(
            files=[
                "Summer16_07Aug2017_V11_MC_L1FastJet_AK8PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L2Relative_AK8PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L3Absolute_AK8PF.jec.txt",
                # NOTE: Uncertainty files have incompatible format (start with # comment)
                "Summer16_07Aug2017_V11_MC_UncertaintySources_AK8PF.junc.txt",
                "Summer16_07Aug2017_V11_MC_Uncertainty_AK8PF.junc.txt",
                "Summer16_25nsV1_MC_PtResolution_AK8PF.jr.txt",
                "Summer16_25nsV1_MC_SF_AK8PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2016preVFPmcNOJER": jet_factory_factory(
            files=[
                "Summer16_07Aug2017_V11_MC_L1FastJet_AK8PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L2Relative_AK8PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L3Absolute_AK8PF.jec.txt",
                # NOTE: Uncertainty files have incompatible format (start with # comment)
                "Summer16_07Aug2017_V11_MC_Uncertainty_AK8PF.junc.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2016preVFPmcNOJEC": jet_factory_factory(
            files=[
                "Summer16_25nsV1_MC_PtResolution_AK8PF.jr.txt",
                "Summer16_25nsV1_MC_SF_AK8PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2016postVFPmc": jet_factory_factory(
            files=[
                "Summer16_07Aug2017_V11_MC_L1FastJet_AK8PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L2Relative_AK8PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L3Absolute_AK8PF.jec.txt",
                # NOTE: Uncertainty files have incompatible format (start with # comment)
                "Summer16_07Aug2017_V11_MC_UncertaintySources_AK8PF.junc.txt",
                "Summer16_07Aug2017_V11_MC_Uncertainty_AK8PF.junc.txt",
                "Summer16_25nsV1_MC_PtResolution_AK8PF.jr.txt",
                "Summer16_25nsV1_MC_SF_AK8PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2016postVFPmcNOJER": jet_factory_factory(
            files=[
                "Summer16_07Aug2017_V11_MC_L1FastJet_AK8PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L2Relative_AK8PF.jec.txt",
                "Summer16_07Aug2017_V11_MC_L3Absolute_AK8PF.jec.txt",
                # NOTE: Uncertainty files have incompatible format (start with # comment)
                "Summer16_07Aug2017_V11_MC_Uncertainty_AK8PF.junc.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2016postVFPmcNOJEC": jet_factory_factory(
            files=[
                "Summer16_25nsV1_MC_PtResolution_AK8PF.jr.txt",
                "Summer16_25nsV1_MC_SF_AK8PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017mc": jet_factory_factory(
            files=[
                "Fall17_17Nov2017_V32_MC_L1FastJet_AK8PF.jec.txt",
                "Fall17_17Nov2017_V32_MC_L2Relative_AK8PF.jec.txt",
                "Fall17_17Nov2017_V32_MC_L3Absolute_AK8PF.jec.txt",
                # NOTE: Uncertainty files have incompatible format (start with # comment)
                "Fall17_17Nov2017_V32_MC_UncertaintySources_AK8PF.junc.txt",
                "Fall17_17Nov2017_V32_MC_Uncertainty_AK8PF.junc.txt",
                "Fall17_V3_MC_PtResolution_AK8PF.jr.txt",
                "Fall17_V3_MC_SF_AK8PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017mcNOJER": jet_factory_factory(
            files=[
                "Fall17_17Nov2017_V32_MC_L1FastJet_AK8PF.jec.txt",
                "Fall17_17Nov2017_V32_MC_L2Relative_AK8PF.jec.txt",
                "Fall17_17Nov2017_V32_MC_L3Absolute_AK8PF.jec.txt",
                # NOTE: Uncertainty files have incompatible format (start with # comment)
                "Fall17_17Nov2017_V32_MC_Uncertainty_AK8PF.junc.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017mcNOJEC": jet_factory_factory(
            files=[
                "Fall17_V3_MC_PtResolution_AK8PF.jr.txt",
                "Fall17_V3_MC_SF_AK8PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018mc": jet_factory_factory(
            files=[
                "Autumn18_V19_MC_L1FastJet_AK8PF.jec.txt",
                "Autumn18_V19_MC_L2Relative_AK8PF.jec.txt",
                "Autumn18_V19_MC_L3Absolute_AK8PF.jec.txt",
                # NOTE: Uncertainty files have incompatible format (parsed as jme_standard_function, not jec_uncertainty_lookup)
                "Autumn18_V19_MC_UncertaintySources_AK8PF.junc.txt",
                "Autumn18_V19_MC_Uncertainty_AK8PF.junc.txt",
                "Autumn18_V7_MC_PtResolution_AK8PF.jr.txt",
                "Autumn18_V7_MC_SF_AK8PF.jersf.txt",
                # "Summer19UL18_V5_MC_L1FastJet_AK8PFPuppi.jec.txt",
                # "Summer19UL18_V5_MC_L2Relative_AK8PFPuppi.jec.txt",
                # "Summer19UL18_V5_MC_UncertaintySources_AK8PFPuppi.junc.txt",
                # "Summer19UL18_V5_MC_Uncertainty_AK8PFPuppi.junc.txt",
                # "Summer19UL18_JRV2_MC_PtResolution_AK8PFPuppi.jr.txt",
                # "Summer19UL18_JRV2_MC_SF_AK8PFPuppi.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018mcNOJER": jet_factory_factory(
            files=[
                "Autumn18_V19_MC_L1FastJet_AK8PF.jec.txt",
                "Autumn18_V19_MC_L2Relative_AK8PF.jec.txt",
                "Autumn18_V19_MC_L3Absolute_AK8PF.jec.txt",
                # NOTE: Uncertainty files have incompatible format (parsed as jme_standard_function, not jec_uncertainty_lookup)
                "Autumn18_V19_MC_Uncertainty_AK8PF.junc.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018mcNOJEC": jet_factory_factory(
            files=[
                "Autumn18_V7_MC_PtResolution_AK8PF.jr.txt",
                "Autumn18_V7_MC_SF_AK8PF.jersf.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        # Data entries (L1FastJet + L2Relative + L3Absolute + L2L3Residual, no JER)
        "2016preVFPdataBCDNOJER": jet_factory_factory(
            files=[
                "Summer16_07Aug2017BCD_V11_DATA_L1FastJet_AK8PF.jec.txt",
                "Summer16_07Aug2017BCD_V11_DATA_L2Relative_AK8PF.jec.txt",
                "Summer16_07Aug2017BCD_V11_DATA_L3Absolute_AK8PF.jec.txt",
                "Summer16_07Aug2017BCD_V11_DATA_L2L3Residual_AK8PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2016postVFPdataEFNOJER": jet_factory_factory(
            files=[
                "Summer16_07Aug2017EF_V11_DATA_L1FastJet_AK8PF.jec.txt",
                "Summer16_07Aug2017EF_V11_DATA_L2Relative_AK8PF.jec.txt",
                "Summer16_07Aug2017EF_V11_DATA_L3Absolute_AK8PF.jec.txt",
                "Summer16_07Aug2017EF_V11_DATA_L2L3Residual_AK8PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2016postVFPdataGHNOJER": jet_factory_factory(
            files=[
                "Summer16_07Aug2017GH_V11_DATA_L1FastJet_AK8PF.jec.txt",
                "Summer16_07Aug2017GH_V11_DATA_L2Relative_AK8PF.jec.txt",
                "Summer16_07Aug2017GH_V11_DATA_L3Absolute_AK8PF.jec.txt",
                "Summer16_07Aug2017GH_V11_DATA_L2L3Residual_AK8PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017dataBNOJER": jet_factory_factory(
            files=[
                "Fall17_17Nov2017B_V32_DATA_L1FastJet_AK8PF.jec.txt",
                "Fall17_17Nov2017B_V32_DATA_L2Relative_AK8PF.jec.txt",
                "Fall17_17Nov2017B_V32_DATA_L3Absolute_AK8PF.jec.txt",
                "Fall17_17Nov2017B_V32_DATA_L2L3Residual_AK8PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017dataCNOJER": jet_factory_factory(
            files=[
                "Fall17_17Nov2017C_V32_DATA_L1FastJet_AK8PF.jec.txt",
                "Fall17_17Nov2017C_V32_DATA_L2Relative_AK8PF.jec.txt",
                "Fall17_17Nov2017C_V32_DATA_L3Absolute_AK8PF.jec.txt",
                "Fall17_17Nov2017C_V32_DATA_L2L3Residual_AK8PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017dataDENOJER": jet_factory_factory(
            files=[
                "Fall17_17Nov2017DE_V32_DATA_L1FastJet_AK8PF.jec.txt",
                "Fall17_17Nov2017DE_V32_DATA_L2Relative_AK8PF.jec.txt",
                "Fall17_17Nov2017DE_V32_DATA_L3Absolute_AK8PF.jec.txt",
                "Fall17_17Nov2017DE_V32_DATA_L2L3Residual_AK8PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2017dataFNOJER": jet_factory_factory(
            files=[
                "Fall17_17Nov2017F_V32_DATA_L1FastJet_AK8PF.jec.txt",
                "Fall17_17Nov2017F_V32_DATA_L2Relative_AK8PF.jec.txt",
                "Fall17_17Nov2017F_V32_DATA_L3Absolute_AK8PF.jec.txt",
                "Fall17_17Nov2017F_V32_DATA_L2L3Residual_AK8PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018dataRunANOJER": jet_factory_factory(
            files=[
                "Autumn18_RunA_V19_DATA_L1FastJet_AK8PF.jec.txt",
                "Autumn18_RunA_V19_DATA_L2Relative_AK8PF.jec.txt",
                "Autumn18_RunA_V19_DATA_L3Absolute_AK8PF.jec.txt",
                "Autumn18_RunA_V19_DATA_L2L3Residual_AK8PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018dataRunBNOJER": jet_factory_factory(
            files=[
                "Autumn18_RunB_V19_DATA_L1FastJet_AK8PF.jec.txt",
                "Autumn18_RunB_V19_DATA_L2Relative_AK8PF.jec.txt",
                "Autumn18_RunB_V19_DATA_L3Absolute_AK8PF.jec.txt",
                "Autumn18_RunB_V19_DATA_L2L3Residual_AK8PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018dataRunCNOJER": jet_factory_factory(
            files=[
                "Autumn18_RunC_V19_DATA_L1FastJet_AK8PF.jec.txt",
                "Autumn18_RunC_V19_DATA_L2Relative_AK8PF.jec.txt",
                "Autumn18_RunC_V19_DATA_L3Absolute_AK8PF.jec.txt",
                "Autumn18_RunC_V19_DATA_L2L3Residual_AK8PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
        "2018dataRunDNOJER": jet_factory_factory(
            files=[
                "Autumn18_RunD_V19_DATA_L1FastJet_AK8PF.jec.txt",
                "Autumn18_RunD_V19_DATA_L2Relative_AK8PF.jec.txt",
                "Autumn18_RunD_V19_DATA_L3Absolute_AK8PF.jec.txt",
                "Autumn18_RunD_V19_DATA_L2L3Residual_AK8PF.jec.txt",
            ],
            jec_name_map=jec_name_map,
        ),
    }
    
    met_factory = CorrectedMETFactory(jec_name_map)

    return jet_factory, fatjet_factory, met_factory


def fetch_lum_corrections():

    for year in ["2016APV", "2016", "2017", "2018"]:
        _ = fetch_and_save_files_corrections(pog="lum",observable="pu",year=year)

    return 


def fetch_xy_met_corrections():
    
        #fetch the corrections for the met
        for year in ["2016APV", "2016", "2017", "2018"]:
          _ = fetch_and_save_files_corrections(pog='jme',observable='met',year=year)
    
        return

def fetch_leptons_corrections():

    #fetch the corrections for the electrons
    for year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
        fetch_and_save_files_corrections(pog='egamma',observable='ele',year=year)
        fetch_and_save_files_corrections(pog='muon',observable='mu',year=year)

    #fetch the corrections for the muons
    fetch_and_save_files_corrections(pog='lum',observable='mu',year='2016preVFP')
    fetch_and_save_files_corrections(pog='lum',observable='mu',year='2016postVFP')
    fetch_and_save_files_corrections(pog='lum',observable='mu',year='2017')
    fetch_and_save_files_corrections(pog='lum',observable='mu',year='2018')

    return



def build_corrections(corrections_to_include_dict):

    corrections = {}
    for key, value in corrections_to_include_dict.items():
        corrections[key] = value


    return corrections

def main():

    args = __get_arguments()

    #get actual date and time
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    #build corrections based on the arguments, if include_jme_corrections is True
    corrections_to_include_dict = {}

    print("Building corrections...")

    if args.include_jme_corrections or args.include_all_corrections:
        print("Including JME corrections...")

        jet_factory, fatjet_factory, met_factory = build_jet_and_corrections()

        corrections_to_include_dict["jet_factory"] = jet_factory
        corrections_to_include_dict["fatjet_factory"] = fatjet_factory
        corrections_to_include_dict["met_factory"] = met_factory

    # NOTE: do not automatically include MET XY corrections with JME.
    # MET phi (XY) corrections are controlled by the separate `-met` flag
    # to avoid applying them when only JME is requested.


    if args.include_pu_corrections or args.include_all_corrections:
        print("Including PU corrections...")

        fetch_lum_corrections()

        corrections_to_include_dict["get_pu_weight"] = get_pu_weight

    # MET XY corrections are optional and not included with -jme alone.
    if args.include_met_corrections or args.include_all_corrections:
        print("Including MET phi (XY) corrections...")
        fetch_xy_met_corrections()
        corrections_to_include_dict["get_met_xy_correction"] = XY_MET_Correction

    corrections = build_corrections(corrections_to_include_dict)

    print("Corrections built successfully !")

    #build a string with the arguments
    arguments = ""

    if args.include_all_corrections:
        arguments += "_all_corr"
        
    if args.include_jme_corrections and not args.include_all_corrections:
            arguments += "_jme"

    if args.include_pu_corrections and not args.include_all_corrections:
            arguments += "_pu"

    if not args.include_all_corrections:
            arguments += "_corr"

    print(f"Saving corrections to {out_path_corrections}/corrections_{date_time}{arguments}.coffea")
    save(corrections, f'{out_path_corrections}/corrections_{date_time}{arguments}.coffea')
    print(f"Corrections saved successfully.")

if __name__ == "__main__":

    main()