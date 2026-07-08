#!/bin/bash

# Debugging copy of the Lund-reweighting skim script, adapted to run over the
# same dataset/files the production make_skims_s_channel_scouting.sh run crashed on
# (s-channel_mMed-500_mDark-20_rinv-0.3), with a single worker and a single chunk
# so a FastJet crash can be isolated to one file/chunk and inspected via the
# diagnostics added to LundReweighting/svjReweighter.py's prep_events().

MEMORY=4GB
CORES=1
CHUNK_SIZE=5000
N_WORKERS=1          # single worker: rule out any concurrency-related crash
MAX_CHUNKS=1         # -m 1: only process the first chunk, to isolate quickly
EXECUTOR=futures     # local job, matches make_skims_s_channel_scouting.sh
FORCE_RECREATE=1
FIRST_FILE=0
LAST_FILE=0          # only the first input file in the list

dataset_directory=/ceph/mgais/Run2ScoutingSkims_JEC

module=analysis_configs.s_channel_scouting_pre_selection
selection_name=s_channel_scouting_pre_selection

year=2018

apply_scouting_jec=1      # matches make_skims_s_channel_scouting.sh

# Local debug output directory - avoid touching the real production output
output_directory=/work/${USER}/lund_debug_skims

dataset_names=(
    s-channel_mMed-500_mDark-20_rinv-0.5
)

cross_sections=(
    79.21
)


make_skims() {

    local dataset_directory=$1
    local module=$2
    local selection_name=$3
    local year=$4
    local dataset_name=$5
    local output_directory=$6
    local xsec=$7

    # Path automatically built when preparing input files lists
    local files_list_directory=${dataset_directory}/skim_input_files_list/${year}/${selection_name}/${dataset_name}
    local output_directory=${output_directory}/${year}/${selection_name}/nominal/${dataset_name}

    mkdir -p ${output_directory}

    i_file=-1
    for files_list in $(ls ${files_list_directory} | sort -V); do
        ((i_file++))
        if [ ${i_file} -le ${LAST_FILE} ] || [ "${LAST_FILE}" == "-1" ]; then
            if [ ${i_file} -ge ${FIRST_FILE} ]; then

                local input_files=${files_list_directory}/${files_list}
                local output_file=${output_directory}/${files_list/.txt/.root}

                echo ""
                echo "Making debug skim file ${output_file}"

                if [ "${apply_scouting_jec}" == "1" ]; then
                    scouting_jec_flag=""
                else
                    scouting_jec_flag="--disable_scouting_jec"
                fi

                python skim.py -i ${input_files} -o ${output_file} -p ${module} -pd ${dataset_name} -y ${year} -nano_scout -mc -xsec ${xsec} -e ${EXECUTOR} -n ${N_WORKERS} -c ${CHUNK_SIZE} -m ${MAX_CHUNKS} --memory ${MEMORY} --cores ${CORES} -lund ${scouting_jec_flag}

                echo ${output_file} has been saved.
            fi
        fi
    done
}


n_datasets=${#dataset_names[@]}

for ((i=0; i<$n_datasets; i++)); do
    dataset_name=${dataset_names[i]}
    cross_section=${cross_sections[i]}
    make_skims ${dataset_directory} ${module} ${selection_name} ${year} ${dataset_name} ${output_directory} ${cross_section}
done
