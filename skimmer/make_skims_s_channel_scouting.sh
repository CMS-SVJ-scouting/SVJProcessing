#!/bin/bash

MEMORY=4GB
CORES=1
CHUNK_SIZE=5000 #10000 or 1000
N_WORKERS=40
#EXECUTOR=dask/etpcondor   # HTCondor at KIT ETP
PORT=3719 # port for dask scheduler, needs to be opened by admins
#N_WORKERS=6
EXECUTOR=futures     # local job
FORCE_RECREATE=0 # 1 to recreate output file if it exists, 0 else
FIRST_FILE=0
LAST_FILE=-1 #150  # Use -1 to skim all input files

dataset_directory=/ceph/mgais/Run2ScoutingSkims_JEC

module=analysis_configs.s_channel_scouting_pre_selection
selection_name=s_channel_scouting_pre_selection

#module=analysis_configs.t_channel_wnae_qcd_training_region
#selection_name=t_channel_wnae_qcd_training_region

#module=analysis_configs.t_channel_wnae_top_training_region
#selection_name=t_channel_wnae_top_training_region

#module=analysis_configs.t_channel_lost_lepton_control_region
#selection_name=t_channel_lost_lepton_control_region

#do not pass to skim.py if you want to skip JERC application
pfnano_corrections_file=/work/mgais/SVJProcessing/data/corrections_2026-04-28_18-13-31_all_corr.coffea

year=2018

add_weights_variations=0  # 1 to add PDF/scale weight variations, 0 else
apply_scouting_jec=1      # 1 to apply custom scouting JECs, 0 to disable
apply_jer=1               # 1 to apply JER smearing for MC, 0 to disable

variations=(
    nominal
    # JEC/JER variations
    #jec_up
    #jec_down
    #jer_up
    #jer_down
    # Unclustered energy variations
    #unclEn_up
    #unclEn_down
)

# Output directory for nominal samples - no variation of the uncertainties
#output_directory=root://cmseos.fnal.gov//store/user/lpcdarkqcd/tchannel_UL/${year}/Full/PrivateSkims/${variation}
#output_directory=root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_v2
output_directory=root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_v2

dataset_names=(
    #
    # Signals
    #

    # s-channel_mMed-500_mDark-20_rinv-0.3
    # # s-channel_mMed-500_mDark-20_rinv-0.5
    # # s-channel_mMed-500_mDark-20_rinv-0.7

    # s-channel_mMed-600_mDark-20_rinv-0.3
    # # s-channel_mMed-600_mDark-20_rinv-0.5
    # # s-channel_mMed-600_mDark-20_rinv-0.7

    # s-channel_mMed-700_mDark-20_rinv-0.3
    # # s-channel_mMed-700_mDark-20_rinv-0.5
    # # s-channel_mMed-700_mDark-20_rinv-0.7

    # s-channel_mMed-800_mDark-20_rinv-0.3
    # # s-channel_mMed-800_mDark-20_rinv-0.5
    # # s-channel_mMed-800_mDark-20_rinv-0.7

    # s-channel_mMed-900_mDark-20_rinv-0.3
    # # s-channel_mMed-900_mDark-20_rinv-0.5
    # # s-channel_mMed-900_mDark-20_rinv-0.7

    # s-channel_mMed-1000_mDark-20_rinv-0.3
    # # s-channel_mMed-1000_mDark-20_rinv-0.5
    # # s-channel_mMed-1000_mDark-20_rinv-0.7

    # s-channel_mMed-1100_mDark-20_rinv-0.3
    # # s-channel_mMed-1100_mDark-20_rinv-0.5
    # # s-channel_mMed-1100_mDark-20_rinv-0.7

    # s-channel_mMed-1200_mDark-20_rinv-0.3
    # # s-channel_mMed-1200_mDark-20_rinv-0.5
    # # s-channel_mMed-1200_mDark-20_rinv-0.7

    # s-channel_mMed-1300_mDark-20_rinv-0.3
    # # s-channel_mMed-1300_mDark-20_rinv-0.5
    # # s-channel_mMed-1300_mDark-20_rinv-0.7

    # s-channel_mMed-1400_mDark-20_rinv-0.3
    # # s-channel_mMed-1400_mDark-20_rinv-0.5
    # # s-channel_mMed-1400_mDark-20_rinv-0.7

    # s-channel_mMed-1500_mDark-20_rinv-0.3
    # # s-channel_mMed-1500_mDark-20_rinv-0.5
    # # s-channel_mMed-1500_mDark-20_rinv-0.7

    # s-channel_mMed-3000_mDark-20_rinv-0.3
    # s-channel_mMed-3000_mDark-20_rinv-0.5
    # s-channel_mMed-3000_mDark-20_rinv-0.7
    
    #
    # Backgrounds
    #
    # QCD
    #
    # low HT bins < 300 don't survive the pre-selection, can be emitted
    #QCD_HT100to200
    #QCD_HT200to300 
    QCD_HT700to1000
    QCD_HT1000to1500
    QCD_HT1500to2000
    QCD_HT2000toInf
    QCD_HT300to500
    QCD_HT500to700

    #
    # TTJets
    #
    TTJets_TuneCP5
    # TTJets_SingleLeptFromT
    # TTJets_SingleLeptFromTbar
    # TTJets_DiLept
    TTJets_HT-600to800
    TTJets_HT-800to1200
    TTJets_HT-1200to2500
    TTJets_HT-2500toInf

    #
    # WJets
    #
    WJetsToLNu_HT-400To600
    WJetsToLNu_HT-600To800
    WJetsToLNu_HT-800To1200
    WJetsToLNu_HT-1200To2500
    WJetsToLNu_HT-2500ToInf

    #
    # ZJets
    #    
    ZJetsToNuNu_HT-400To600
    ZJetsToNuNu_HT-600To800
    ZJetsToNuNu_HT-800To1200
    ZJetsToNuNu_HT-1200To2500
    ZJetsToNuNu_HT-2500ToInf
)

cross_sections=(
    # # see https://github.com/CMS-SVJ-scouting/SVJScouting_ntuplizer/wiki/Cross-sections, for newest signal cross-sections see https://github.com/kpedro88/Stat/blob/width/Limits/test/dict_xsec_Zprime.txt
    #
    # Signals
    #
    # 79.21
    # # 79.21
    # # 79.21
    
    # 53.32
    # # 53.32
    # # 53.32

    # 35.89
    # # 35.89
    # # 35.89

    # 24.16
    # # 24.16
    # # 24.16

    # 16.26
    # # 16.26
    # # 16.26

    # 10.95
    # # 10.95
    # # 10.95

    # 7.368
    # # 7.368
    # # 7.368

    # 5.086
    # # 5.086
    # # 5.086

    # 3.586
    # # 3.586
    # # 3.586

    # 2.574
    # # 2.574
    # # 2.574

    # 1.875
    # # 1.875
    # # 1.875

    # 0.0412
    # 0.0412
    # 0.0412


    #
    # Backgrounds
    #
    # QCD
    6310
    1094
    99.38
    20.20
    323400
    30140

    # TTJets
    831.8
    # 182.2
    # 182.2
    # 87.3
    2.4
    0.98
    0.2
    0.002

    # WJets
    51.5
    12.5
    5.6
    1.3
    0.03

    # ZJets
    11.26
    2.73
    1.22
    0.28
    0.0064

)


make_skims() {

    local dataset_directory=$1
    local module=$2
    local selection_name=$3
    local year=$4
    local variation=$5
    local dataset_name=$6
    local output_directory=$7
    local xsec=$8

    # Path automatically built when preparing input files lists
    local files_list_directory=${dataset_directory}/skim_input_files_list/${year}/${selection_name}/${dataset_name}
    local output_directory=${output_directory}/${year}/${selection_name}/${variation}/${dataset_name}

    local output_redirector=$(echo ${output_directory} | cut -d/ -f 1-4)
    local output_dir=$(echo ${output_directory} | cut -d/ -f 4-)
    xrdfs ${output_redirector} ls ${output_dir} > /dev/null 2>&1
    if [ "$?" != "0" ]; then
        xrdfs ${output_redirector} mkdir -p ${output_dir}
    fi

    i_file=-1
    for files_list in $(ls ${files_list_directory} | sort -V); do
        ((i_file++))
        if [ ${i_file} -le ${LAST_FILE} ] || [ "${LAST_FILE}" == "-1" ]; then
            if [ ${i_file} -ge ${FIRST_FILE} ]; then

                local input_files=${files_list_directory}/${files_list}
                local output_file=${output_directory}/${files_list/.txt/.root}
                local output_file_name_tmp=$(echo ${ouput_file}_$(date +"%Y%m%d-%H%M%S") | shasum | cut -d " " -f1).root
                local output_file_tmp=/tmp/${USER}/${output_file_name_tmp}

                echo ""
                echo "Making skim file ${output_file}"

                local output_redirector=$(echo ${output_file} | cut -d/ -f 1-4)
                local output_file_path=$(echo ${output_file} | cut -d/ -f 4-)
                xrdfs ${output_redirector} ls ${output_file_path} > /dev/null 2>&1
                if [ "$?" != "0" ] || [ "${FORCE_RECREATE}" == "1" ]; then
                    if [ "${apply_scouting_jec}" == "1" ]; then
                        scouting_jec_flag=""
                    else
                        scouting_jec_flag="--disable_scouting_jec"
                    fi
                    if [ "${apply_jer}" == "1" ]; then
                        jer_flag=""
                    else
                        jer_flag="--disable_jer"
                    fi
                    if [ "${variation}" == "nominal" ]; then
                        variation_flag=""
                    else
                        variation_flag="-varnano ${variation}"
                    fi
                    if [ ${add_weights_variations} == 1 ]; then
                        weight_variation_flag="-wvarnano scale pdf pu"
                    else
                        weight_variation_flag=""
                    fi
                    python skim.py -i ${input_files} -o ${output_file_tmp} -p ${module} -pd ${dataset_name} -y ${year} -nano_scout -mc -xsec ${xsec} -corrfile ${pfnano_corrections_file} -e ${EXECUTOR} -port ${PORT} -n ${N_WORKERS} -c ${CHUNK_SIZE} --memory ${MEMORY} --cores ${CORES} -pn_tagger ${variation_flag} ${weight_variation_flag} ${scouting_jec_flag} ${jer_flag}
                    #python skim.py -i ${input_files} -o ${output_file_tmp} -p ${module} -pd ${dataset_name} -y ${year} -nano_scout -mc -xsec ${xsec} -corrfile ${pfnano_corrections_file} -e ${EXECUTOR} -port ${PORT} -n ${N_WORKERS} -c ${CHUNK_SIZE} --memory ${MEMORY} --cores ${CORES} -pn_tagger ${variation_flag} ${weight_variation_flag}
                    xrdcp -f ${output_file_tmp} ${output_file}
                    echo ${output_file} has been saved.
                    rm ${output_file_tmp}
                else
                    echo ${output_file} already exists and FORCE_RECREATE is 0. Skipping.
                fi
            fi
        fi
    done
}


#for dataset_name in ${dataset_names[@]}; do
#    make_skims ${dataset_directory} ${module} ${selection_name} ${year} ${dataset_name} ${output_directory}
#done

n_datasets=${#dataset_names[@]}

for ((i=0; i<$n_datasets; i++)); do
    dataset_name=${dataset_names[i]}
    cross_section=${cross_sections[i]}
    for variation in ${variations[@]}; do
        make_skims ${dataset_directory} ${module} ${selection_name} ${year} ${variation} ${dataset_name} ${output_directory} ${cross_section}
    done
done
