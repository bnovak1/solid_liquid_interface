#!/bin/bash

template_file=${1}
restart_file=${2}
box_sizes_file=${3}
NPT_solid_infile=${4}
temp=${5}
n_temp_bins=${6}
timestep=${7}
run_num=${8}
orientation=${9}
outfile=${10}

boxx=`tail -1 ${box_sizes_file} | awk '{print $1}'`
boxy=`tail -1 ${box_sizes_file} | awk '{print $2}'`

cp ${template_file} ${outfile}

sed -i 's|\[RESTART_FILE\]|'${restart_file}'|g' ${outfile}
sed -i 's|\[BOXX\]|'${boxx}'|g' ${outfile}
sed -i 's|\[BOXY\]|'${boxy}'|g' ${outfile}
sed -i 's|\[TEMP\]|'${temp}'|g' ${outfile}
sed -i 's|\[N_TEMP_BINS\]|'${n_temp_bins}'|g' ${outfile}
sed -i 's|\[TIMESTEP\]|'${timestep}'|g' ${outfile}
sed -i 's|\[RUN_NUM\]|'${run_num}'|g' ${outfile}
sed -i 's|\[ORIENTATION\]|'${orientation}'|g' ${outfile}
