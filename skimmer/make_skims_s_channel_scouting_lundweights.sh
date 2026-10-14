#!/bin/bash

#MEMORY=4GB
#CORES=1
#CHUNK_SIZE=100000 #10000 or 1000
#N_WORKERS=8
##EXECUTOR=dask/etpcondor   # HTCondor at KIT ETP
#PORT=3719 # port for dask scheduler, needs to be opened by admins
##N_WORKERS=6
#EXECUTOR=futures     # local job
#FORCE_RECREATE=1 # 1 to recreate output file if it exists, 0 else
#FIRST_FILE=0
#LAST_FILE=-1 #150  # Use -1 to skim all input files

MEMORY=10GB
TIME=12:00:00
PARTITION=standard
CORES=2
CHUNK_SIZE=10000000
N_WORKERS=300
#EXECUTOR=dask/lpccondor    # HTCondor at LPC
EXECUTOR=dask/slurm          #dask/slurm     # local job
FORCE_RECREATE=1   # 1 to recreate output file if it exists, 0 else
FIRST_FILE=0
LAST_FILE=-1  # Use -1 to skim all input files

dataset_directory=/work/cazzanig/datasets_lund_official/

module=analysis_configs.s_channel_scouting_pre_selection
selection_name=s_channel_scouting_pre_selection

pfnano_corrections_file=/work/cazzanig/lund_weights_scouting/merged_processing/SVJProcessing/data/corrections_2026-06-10_09-23-50_all_corr.coffea

year=2018

add_weights_variations=1  # 1 to add PDF/scale weight variations, 0 else
apply_scouting_jec=1     # 1 to apply custom scouting residual JECs, 0 to disable

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
    #SVJjec_up 
    #SVJjec_down
)

# Output directory for nominal samples - no variation of the uncertainties
#output_directory=root://cmseos.fnal.gov//store/user/lpcdarkqcd/tchannel_UL/${year}/Full/PrivateSkims/${variation}
output_directory=root://t3dcachedb03.psi.ch//pnfs/psi.ch/cms/trivcat/store/t3groups/ethz-susy/skims_lund/ #_small


dataset_names=(
    #
    # Signals
    #
    #s-channel_mMed-500_mDark-20_rinv-0.3
    #s-channel_mMed-500_mDark-20_rinv-0.5
    #s-channel_mMed-500_mDark-20_rinv-0.7

    #s-channel_mMed-600_mDark-20_rinv-0.3
    #s-channel_mMed-600_mDark-20_rinv-0.5
    #s-channel_mMed-600_mDark-20_rinv-0.7

    #s-channel_mMed-700_mDark-20_rinv-0.3
    #s-channel_mMed-700_mDark-20_rinv-0.5
    #s-channel_mMed-700_mDark-20_rinv-0.7

    #s-channel_mMed-800_mDark-20_rinv-0.3
    #s-channel_mMed-800_mDark-20_rinv-0.5
    #s-channel_mMed-800_mDark-20_rinv-0.7

    #s-channel_mMed-900_mDark-20_rinv-0.3
    #s-channel_mMed-900_mDark-20_rinv-0.5
    #s-channel_mMed-900_mDark-20_rinv-0.7

    #s-channel_mMed-1000_mDark-20_rinv-0.3
    #s-channel_mMed-1000_mDark-20_rinv-0.5
    #s-channel_mMed-1000_mDark-20_rinv-0.7

    s-channel_mMed-1100_mDark-20_rinv-0.3
    #s-channel_mMed-1100_mDark-20_rinv-0.5
    #s-channel_mMed-1100_mDark-20_rinv-0.7

    #s-channel_mMed-1200_mDark-20_rinv-0.3
    #s-channel_mMed-1200_mDark-20_rinv-0.5
    #s-channel_mMed-1200_mDark-20_rinv-0.7

    #s-channel_mMed-1300_mDark-20_rinv-0.3
    #s-channel_mMed-1300_mDark-20_rinv-0.5
    #s-channel_mMed-1300_mDark-20_rinv-0.7

    #s-channel_mMed-1400_mDark-20_rinv-0.3
    #s-channel_mMed-1400_mDark-20_rinv-0.5
    #s-channel_mMed-1400_mDark-20_rinv-0.7

    #s-channel_mMed-1500_mDark-20_rinv-0.3
    #s-channel_mMed-1500_mDark-20_rinv-0.5
    #s-channel_mMed-1500_mDark-20_rinv-0.7


)

cross_sections=(
    # # see https://github.com/CMS-SVJ-scouting/SVJScouting_ntuplizer/wiki/Cross-sections, for newest signal cross-sections see https://github.com/kpedro88/Stat/blob/width/Limits/test/dict_xsec_Zprime.txt
    #
    # Signals
    #
    #79.21
    #79.21
    #79.21
    
    #53.32
    #53.32
    #53.32

    #35.89
    #35.89
    #35.89

    #24.16
    #24.16
    #24.16

    #16.26
    #16.26
    #16.26

    10.95
    #10.95
    #10.95

    #7.368
    #7.368
    #7.368

    #5.086
    #5.086
    #5.086

    #3.586
    #3.586
    #3.586

    #2.574
    #2.574
    #2.574

    #1.875
    #1.875
    #1.875

    # 0.0412
    # 0.0412
    # 0.0412
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
                local output_file_tmp=/work/${USER}/tmp/${output_file_name_tmp}

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
                    #python skim.py -i ${input_files} -o ${output_file_tmp} -p ${module} -pd ${dataset_name} -y ${year} -nano_scout -mc -xsec ${xsec} -e ${EXECUTOR} -n ${N_WORKERS} -c ${CHUNK_SIZE} --memory ${MEMORY} --cores ${CORES} -pn_tagger ${variation_flag} ${weight_variation_flag} ${scouting_jec_flag} -lund -m 1
                    #python skim.py -i ${input_files} -o ${output_file_tmp} -p ${module} -pd ${dataset_name} -y ${year} -nano_scout -mc -xsec ${xsec} -corrfile ${pfnano_corrections_file} -e ${EXECUTOR} -port ${PORT} -n ${N_WORKERS} -c ${CHUNK_SIZE} --memory ${MEMORY} --cores ${CORES} -pn_tagger ${variation_flag} ${weight_variation_flag}
                    python skim.py -i ${input_files} -o ${output_file_tmp} -p ${module} -pd ${dataset_name} -y ${year} -e ${EXECUTOR} -n ${N_WORKERS} -c ${CHUNK_SIZE} --memory ${MEMORY} --queue ${PARTITION} --cores ${CORES} ${variation_flag} ${weight_variation_flag} ${scouting_jec_flag} -xsec ${xsec} -corrfile ${pfnano_corrections_file} -nano_scout -mc -lund -m 1

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
