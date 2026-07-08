import os

from tests import helper


_REDIRECTOR = "root://cmsdcache-kit-disk.gridka.de:1094/"
_BASE_PATH = "/store/user/mgaisdor/SVJScouting_ntuples/MC/2017_v4"
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
    _year = "2017"

    # Each entry: (year, config, files_list, primary_dataset, corrections_file, variation, dataset_name)
    # dataset_name is passed as -ds to the skimmer; needed for stitching cuts (e.g. TTJets).
    params_list = [
        # QCD HT1500to2000 — nominal
        (
            _year, _cfg,
            _files("QCD_HT1500to2000",
                "PFNano_ScoutingAOD_FF3E8635-E476-7144-AB63-19F71BA04832.root",
                "PFNano_ScoutingAOD_FF426860-7C9F-344B-B8E0-860469CDCD52.root",
                "PFNano_ScoutingAOD_FF8AF46D-23DB-0248-A740-F1DA68B92A78.root",
            ),
            "dummy", _CORRECTIONS_FILE, None, "QCD_HT1500to2000",
        ),
        # QCD HT1500to2000 — jec_up: exercises the full JERC code path with corrections file
        (
            _year, _cfg,
            _files("QCD_HT1500to2000",
                "PFNano_ScoutingAOD_FF3E8635-E476-7144-AB63-19F71BA04832.root",
                "PFNano_ScoutingAOD_FF426860-7C9F-344B-B8E0-860469CDCD52.root",
                "PFNano_ScoutingAOD_FF8AF46D-23DB-0248-A740-F1DA68B92A78.root",
            ),
            "dummy", _CORRECTIONS_FILE, "jec_up", "QCD_HT1500to2000",
        ),
        # TTJets_TuneCP5 inclusive — exercises the LHE HT stitching branch
        (
            _year, _cfg,
            _files("TTJets_TuneCP5",
                "PFNano_ScoutingAOD_FF8D614C-D1D5-3A40-A6D0-7FD6463FB548.root",
                "PFNano_ScoutingAOD_FFFF2C03-33D7-534C-94AF-B36F62C68395.root",
                "PFNano_ScoutingAOD_FFFF9923-EB75-1D4C-9FE4-8186A7E33529.root"
            ),
            "dummy", _CORRECTIONS_FILE, None, "TTJets_TuneCP5",
        ),
        # WJets HT1200to2500 — higher-HT bin for better selection efficiency
        (
            _year, _cfg,
            _files("WJetsToLNu_HT-1200To2500",
                "PFNano_ScoutingAOD_FFAA0084-3861-2745-B2B4-AA37BA46951C.root",
                "PFNano_ScoutingAOD_FFC63823-D047-7441-87C2-08015673F604.root",
                "PFNano_ScoutingAOD_FFF2EDB5-8738-6440-B5E9-AEE2C5F2B86B.root"
            ),
            "dummy", _CORRECTIONS_FILE, None, "WJetsToLNu_HT-1200To2500",
        ),
        # ZJets HT1200to2500 — higher-HT bin for better selection efficiency
        (
            _year, _cfg,
            _files("ZJetsToNuNu_HT-1200To2500",
                "PFNano_ScoutingAOD_E13CDB88-34F6-724C-8CF6-05B447F3EAE3.root",
                "PFNano_ScoutingAOD_E31D1C54-3E54-7642-95A1-F6A9FE2B2DDE.root",
                "PFNano_ScoutingAOD_F0F06B02-8BFB-E842-A277-73C88A921FB4.root"
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

