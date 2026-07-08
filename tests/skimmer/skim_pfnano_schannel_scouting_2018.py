import os

from tests import helper


_REDIRECTOR = "root://cmsdcache-kit-disk.gridka.de:1094/"
_BASE_PATH = "/store/user/mgaisdor/SVJScouting_ntuples/MC/2018_v4"
_CORRECTIONS_FILE = os.path.join(os.environ.get('SVJ_PROCESSING_ROOT', '.'), 'data', 'corrections_2026-04-28_18-13-31_all_corr.coffea')


def _files(dataset, *names):
    """Build full XRootD URLs for the given filenames in a dataset directory."""
    return [f"{_REDIRECTOR}/{_BASE_PATH}/{dataset}/{name}" for name in names]


def __run_skimmer(input_files, output_file, config, year, primary_dataset, run_particle_net,
                  corrections_file=None, variation=None, dataset_name=None):
    n_workers = 8
    chunk_size = 100000
    executor = "futures"

    bash_command = f"python {os.environ['SVJ_PROCESSING_ROOT']}/skimmer/skim.py -i {input_files} -o {output_file} -p {config} -y {year} -e {executor} -n {n_workers} -c {chunk_size} -pd {primary_dataset} -nano_scout -mc"
    if dataset_name is not None:
        bash_command += f" -ds {dataset_name}"
    if run_particle_net:
        bash_command += " -pn_tagger"
    if corrections_file is not None:
        bash_command += f" -corrfile {corrections_file}"
    if variation is not None:
        bash_command += f" -varnano {variation}"
    helper.test_command(bash_command)


def test_execution():
    _cfg = "analysis_configs.s_channel_scouting_pre_selection"
    _year = "2018"

    # Each entry: (year, config, files_list, primary_dataset, corrections_file, variation, dataset_name)
    # dataset_name is passed as -ds to the skimmer; needed for stitching cuts (e.g. TTJets).
    params_list = [
        # QCD HT1500to2000 — nominal
        (
            _year, _cfg,
            _files("QCD_HT1500to2000",
                "PFNano_ScoutingAOD_FF98E2A8-EFD4-DE4C-8C59-3CE5DA092AB2.root",
                "PFNano_ScoutingAOD_FFB1AF51-D3F9-7B43-A7BF-CDD985A2F648.root",
                "PFNano_ScoutingAOD_FFBEB68A-130A-4349-95F4-FF727999A2EF.root"
            ),
            "dummy", _CORRECTIONS_FILE, None, "QCD_HT1500to2000",
        ),
        # QCD HT1500to2000 — jec_up: exercises the full JERC code path with corrections file
        (
            _year, _cfg,
            _files("QCD_HT1500to2000",
                "PFNano_ScoutingAOD_FF98E2A8-EFD4-DE4C-8C59-3CE5DA092AB2.root",
                "PFNano_ScoutingAOD_FFB1AF51-D3F9-7B43-A7BF-CDD985A2F648.root",
                "PFNano_ScoutingAOD_FFBEB68A-130A-4349-95F4-FF727999A2EF.root"
            ),
            "dummy", _CORRECTIONS_FILE, "jec_up", "QCD_HT1500to2000",
        ),
        # TTJets_TuneCP5 inclusive — exercises the LHE HT stitching branch
        (
            _year, _cfg,
            _files("TTJets_TuneCP5",
                "PFNano_ScoutingAOD_FF09C4A3-5F54-F446-A6A9-6402871F88E0.root",
                "PFNano_ScoutingAOD_FF09E001-ADE9-7D4E-8202-7A47DD091BB5.root",
                "PFNano_ScoutingAOD_FFE7333C-3F11-8048-A485-E976675D7995.root"
            ),
            "dummy", _CORRECTIONS_FILE, None, "TTJets_TuneCP5",
        ),
        # WJets HT1200to2500 — higher-HT bin for better selection efficiency
        (
            _year, _cfg,
            _files("WJetsToLNu_HT-1200To2500",
                "PFNano_ScoutingAOD_FF9F32C6-FAC7-F543-A53E-2A71BA943145.root",
                "PFNano_ScoutingAOD_FFA17F50-82C3-1349-A233-D000D7B82D02.root",
                "PFNano_ScoutingAOD_FFED29BB-19A8-C84D-90E6-BD1C89DBBBFA.root"
            ),
            "dummy", _CORRECTIONS_FILE, None, "WJetsToLNu_HT-1200To2500",
        ),
        # ZJets HT1200to2500 — higher-HT bin for better selection efficiency
        (
            _year, _cfg,
            _files("ZJetsToNuNu_HT-1200To2500",
                "PFNano_ScoutingAOD_F4ADD3A2-B66D-B349-B288-699272AC8E9C.root",
                "PFNano_ScoutingAOD_F73484CE-B322-D64F-9B27-463FAC0A33F4.root",
                "PFNano_ScoutingAOD_F9285F3D-C54D-F94C-BE2C-40C680FD9D0E.root"
            ),
            "dummy", _CORRECTIONS_FILE, None, "ZJetsToNuNu_HT-1200To2500",
        ),
    ]


    run_particle_net = False

    output_path = helper.get_temporary_directory()
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for params in params_list:
        year, config, files_list, primary_dataset, corrections_file, variation, dataset_name = params
        input_files = ",".join(files_list)
        output_name = helper.run_bash_command(f'echo {input_files}_$(date +"%Y%m%d-%H%M%S") | shasum | cut -d " " -f1')
        output_file = f"{output_path}/{output_name}.root"

        __run_skimmer(input_files, output_file, config, year, primary_dataset, run_particle_net,
                      corrections_file=corrections_file, variation=variation, dataset_name=dataset_name)

        assert os.path.exists(output_file)


test_execution()

