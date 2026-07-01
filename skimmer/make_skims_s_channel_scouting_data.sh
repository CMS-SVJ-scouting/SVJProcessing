#!/bin/bash

MEMORY=4GB
CORES=1
CHUNK_SIZE=10000 #10000 or 1000
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
#/work/mgais/JEC_SVJProcessing/data/corrections_2026-03-12_00-06-24_jme_corr.coffea
#/work/mgais/JEC_SVJProcessing/data/corrections_2025-11-21_15-01-21_jme_corr.coffea

year=2017

add_weights_variations=0  # 1 to add PDF/scale weight variations, 0 else
apply_scouting_jec=1      # 1 to apply custom scouting residual JECs, 0 to disable
apply_residuals=0         # 0 to skip L2L3Residual JEC corrections for data (test); 1 for standard

# Coffea corrections file: use the NORES-capable file when residuals are disabled
pfnano_corrections_file_nores=/work/mgais/SVJProcessing/data/corrections_nores_all_corr.coffea

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
output_directory=root://cmsdcache-kit-disk.gridka.de:1094//store/user/mgaisdor/SVJScouting_skims_v2_noResiduals


dataset_names=(
    Run2017C
    Run2017D
    Run2017E
    Run2017F

    # Run2018A
    # Run2018B
    # Run2018C
    # Run2018D
)



make_skims() {

    local dataset_directory=$1
    local module=$2
    local selection_name=$3
    local year=$4
    local variation=$5
    local dataset_name=$6
    local output_directory=$7
    #local xsec=$8

    # Path automatically built when preparing input files lists
    local files_list_directory=${dataset_directory}/skim_input_files_list/${year}/${selection_name}/${dataset_name}
    local output_directory=${output_directory}/${year}/${selection_name}/${variation}/${dataset_name}

    local output_redirector=$(echo ${output_directory} | cut -d/ -f 1-4)
    local output_dir=$(echo ${output_directory} | cut -d/ -f 4-)
    xrdfs ${output_redirector} ls ${output_dir} > /dev/null 2>&1
    if [ "$?" != "0" ]; then
        xrdfs ${output_redirector} mkdir -p ${output_dir}
    fi

    # Derive run era from dataset name:
    #   Run2017C/D/E/F -> C/D/E/F (no "Run" prefix needed, year=2017)
    #   Run2018A/B/C/D -> RunA/RunB/RunC/RunD (year=2018 uses "Run" prefix)
    local run_era_flag=""
    if [[ ${dataset_name} =~ Run20[0-9][0-9]([A-Z]+) ]]; then
        local era_letter="${BASH_REMATCH[1]}"
        if [[ ${year} == "2018" ]]; then
            run_era_flag="-run_era Run${era_letter}"
        else
            run_era_flag="-run_era ${era_letter}"
        fi
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
                    if [ "${apply_residuals}" == "1" ]; then
                        residuals_flag=""
                        corrfile_to_use=${pfnano_corrections_file}
                    else
                        residuals_flag="--disable_residuals"
                        corrfile_to_use=${pfnano_corrections_file_nores}
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
                    python skim.py -i ${input_files} -o ${output_file_tmp} -p ${module} -pd ${dataset_name} -y ${year} -nano_scout -corrfile ${corrfile_to_use} -e ${EXECUTOR} -port ${PORT} -n ${N_WORKERS} -c ${CHUNK_SIZE} --memory ${MEMORY} --cores ${CORES} -pn_tagger ${variation_flag} ${weight_variation_flag} ${scouting_jec_flag} ${residuals_flag} ${run_era_flag} #--skip_bad_files
                    #python skim.py -i ${input_files} -o ${output_file_tmp} -p ${module} -pd ${dataset_name} -y ${year} -nano_scout -e ${EXECUTOR} -port ${PORT} -n ${N_WORKERS} -c ${CHUNK_SIZE} --memory ${MEMORY} --cores ${CORES} -pn_tagger ${variation_flag[@]}
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
    #cross_section=${cross_sections[i]}
    for variation in ${variations[@]}; do
        make_skims ${dataset_directory} ${module} ${selection_name} ${year} ${variation} ${dataset_name} ${output_directory} #${cross_section}
    done
done
