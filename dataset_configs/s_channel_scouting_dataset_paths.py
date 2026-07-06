###################################  README  ###################################
#
# Is called "dataset" a set of files corresponding to the same physics process.
# The object `datasets_info` describes the location of the different datasets.
# Its structure is the following:
#    * Keys are year
#    * Values are the "datasets_per_year_info"
# The structure of the `datasets_per_year_info` is the following:
#    * Keys are dataset names
#    * Values are the "dataset_info" defining which files belong to the dataset
# 
# The "dataset_info" has the following structure. It is a list of dict, which
# has 2 keys:
#    * "redirector": The XRootD redirector to the remote storage element
#    * "path": The path to the directory at which the files are located
#    * "regex": The regex to apply to select some files from that directory. 
#               The regex must be "" if no regex is applied.
#
################################################################################


years = ["2017","2018"]

datasets_info = {
    year: {} for year in years
}

# SVJ scouting signal models
signal_models = [

    "s-channel_mMed-500_mDark-20_rinv-0.3",
    "s-channel_mMed-600_mDark-20_rinv-0.3",
    "s-channel_mMed-700_mDark-20_rinv-0.3",
    "s-channel_mMed-800_mDark-20_rinv-0.3",
    "s-channel_mMed-900_mDark-20_rinv-0.3",
    "s-channel_mMed-1000_mDark-20_rinv-0.3",
    "s-channel_mMed-1100_mDark-20_rinv-0.3",
    "s-channel_mMed-1200_mDark-20_rinv-0.3",
    "s-channel_mMed-1300_mDark-20_rinv-0.3",
    "s-channel_mMed-1400_mDark-20_rinv-0.3",
    "s-channel_mMed-1500_mDark-20_rinv-0.3",

    "s-channel_mMed-500_mDark-20_rinv-0.5",
    "s-channel_mMed-600_mDark-20_rinv-0.5",
    "s-channel_mMed-700_mDark-20_rinv-0.5",
    "s-channel_mMed-800_mDark-20_rinv-0.5",
    "s-channel_mMed-900_mDark-20_rinv-0.5",
    "s-channel_mMed-1000_mDark-20_rinv-0.5",
    "s-channel_mMed-1100_mDark-20_rinv-0.5",
    "s-channel_mMed-1200_mDark-20_rinv-0.5",
    "s-channel_mMed-1300_mDark-20_rinv-0.5",
    "s-channel_mMed-1400_mDark-20_rinv-0.5",
    "s-channel_mMed-1500_mDark-20_rinv-0.5",

    "s-channel_mMed-500_mDark-20_rinv-0.7",
    "s-channel_mMed-600_mDark-20_rinv-0.7",
    "s-channel_mMed-700_mDark-20_rinv-0.7",
    "s-channel_mMed-800_mDark-20_rinv-0.7",
    "s-channel_mMed-900_mDark-20_rinv-0.7",
    "s-channel_mMed-1000_mDark-20_rinv-0.7",
    "s-channel_mMed-1100_mDark-20_rinv-0.7",
    "s-channel_mMed-1200_mDark-20_rinv-0.7",
    "s-channel_mMed-1300_mDark-20_rinv-0.7",
    "s-channel_mMed-1400_mDark-20_rinv-0.7",
    "s-channel_mMed-1500_mDark-20_rinv-0.7",

]

qcd_bins = [
    "QCD_HT300to500",
    "QCD_HT500to700",
    "QCD_HT700to1000",
    "QCD_HT1000to1500",
    "QCD_HT1500to2000",
    "QCD_HT2000toInf",
]

ttjets_bins = [
    "TTJets_TuneCP5",
    "TTJets_SingleLeptFromT",
    "TTJets_SingleLeptFromTbar",
    "TTJets_DiLept",
    "TTJets_HT-600to800",
    "TTJets_HT-800to1200",
    "TTJets_HT-1200to2500",
    "TTJets_HT-2500toInf",
]

wjets_bins = [
    "WJetsToLNu_HT-400To600",
    "WJetsToLNu_HT-600To800",
    "WJetsToLNu_HT-800To1200",
    "WJetsToLNu_HT-1200To2500",
    "WJetsToLNu_HT-2500ToInf",
]

zjets_bins = [
    "ZJetsToNuNu_HT-400To600",
    "ZJetsToNuNu_HT-600To800",
    "ZJetsToNuNu_HT-800To1200",
    "ZJetsToNuNu_HT-1200To2500",
    "ZJetsToNuNu_HT-2500ToInf",
]



for year in years:
    datasets_info[year].update({
        bin: [
            {
                #"redirector": "root://storage01.lcg.cscs.ch:1096//",
                #"path": f"/pnfs/lcg.cscs.ch/cms/trivcat/store/user/cazzanig/darkshowers/samples/scouting/PFNano/QCD_HT_binned_2018_v0/{bin}/",
                "redirector": "root://cmsdcache-kit-disk.gridka.de:1094/",
                #"path": f"/store/user/mgaisdor/SVJScouting_ntuples/MC/{year}/{bin}/",
                "path": f"/store/user/mgaisdor/SVJScouting_ntuples/MC/{year}_v5/merged_signals/{bin}/",
                "regex": f"",
                
            },
        ]
        for bin in signal_models
    })

for year in years:
    datasets_info[year].update({
        bin: [
            {
                #"redirector": "root://storage01.lcg.cscs.ch:1096//",
                #"path": f"/pnfs/lcg.cscs.ch/cms/trivcat/store/user/cazzanig/darkshowers/samples/scouting/PFNano/QCD_HT_binned_2018_v0/{bin}/",
                "redirector": "root://cmsdcache-kit-disk.gridka.de:1094/",
                "path": f"/store/user/mgaisdor/SVJScouting_ntuples/MC/{year}_v4/{bin}/",
                "regex": f"",
                
            },
        ]
        for bin in qcd_bins
    })

for year in years:
    datasets_info[year].update({
        bin: [
            {
                #"redirector": "root://storage01.lcg.cscs.ch:1096//",
                #"path": f"/pnfs/lcg.cscs.ch/cms/trivcat/store/user/cazzanig/darkshowers/samples/scouting/PFNano/QCD_HT_binned_2018_v0/{bin}/",
                "redirector": "root://cmsdcache-kit-disk.gridka.de:1094/",
                "path": f"/store/user/mgaisdor/SVJScouting_ntuples/MC/{year}_v4/{bin}/",
                "regex": f"",
                
            },
        ]
        for bin in ttjets_bins
    })

for year in years:
    datasets_info[year].update({
        bin: [
            {
                #"redirector": "root://storage01.lcg.cscs.ch:1096//",
                #"path": f"/pnfs/lcg.cscs.ch/cms/trivcat/store/user/cazzanig/darkshowers/samples/scouting/PFNano/QCD_HT_binned_2018_v0/{bin}/",
                "redirector": "root://cmsdcache-kit-disk.gridka.de:1094/",
                "path": f"/store/user/mgaisdor/SVJScouting_ntuples/MC/{year}_v4/{bin}/",
                "regex": f"",
                
            },
        ]
        for bin in wjets_bins
    })

for year in years:
    datasets_info[year].update({
        bin: [
            {
                #"redirector": "root://storage01.lcg.cscs.ch:1096//",
                #"path": f"/pnfs/lcg.cscs.ch/cms/trivcat/store/user/cazzanig/darkshowers/samples/scouting/PFNano/QCD_HT_binned_2018_v0/{bin}/",
                "redirector": "root://cmsdcache-kit-disk.gridka.de:1094/",
                "path": f"/store/user/mgaisdor/SVJScouting_ntuples/MC/{year}_v4/{bin}/",
                "regex": f"",
                
            },
        ]
        for bin in zjets_bins
    })

datasets_info["2017"].update({
    run: [
        {
            "redirector": "root://cmsdcache-kit-disk.gridka.de:1094/",
            "path": f"/store/user/mgaisdor/SVJScouting_ntuples/Data/2017_v2/{run}",
            "regex": f"",
        },
    ]
    for run in ["Run2017C", "Run2017D", "Run2017E", "Run2017F"]
})

datasets_info["2018"].update({
    run: [
        {
            "redirector": "root://cmsdcache-kit-disk.gridka.de:1094/",
            "path": f"/store/user/mgaisdor/SVJScouting_ntuples/Data/2018_v2/{run}",
            "regex": f"",
        },
    ]
    for run in ["Run2018A", "Run2018B", "Run2018C", "Run2018D"]
})