#!/bin/bash

lmp_infile=${1}
prefix=${2}

rm -f ${prefix}.restart.*
mpirun -np 20 lmp_vcsgc -in ${lmp_infile}

restart_files=(`ls -v ${prefix}.restart.*`)
nruns=${#restart_files[@]}

for irun in `seq 1 ${nruns}`; do
    jrun=`echo $irun - 1 | bc`
    restart_file=${restart_files[${jrun}]}
    restart_file_new=${prefix}.restart.${irun}
    mv -f ${restart_file} ${restart_file_new}
    lmp_vcsgc -restart2data ${restart_file_new} ${prefix}.data.${irun}
done
