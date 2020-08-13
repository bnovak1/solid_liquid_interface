#!/bin/bash

lattice_param_file=${1}
template=${2}
lmp_infile=${3}
outfile=${4}

lattice_param=`tail -1 ${lattice_param_file} | awk '{print $1}'`

cp ${template} ${lmp_infile}
sed -i 's|\[LATPARAM\]|'${lattice_param}'|g' ${lmp_infile}
sed -i 's|\[BOX_FILE\]|'${outfile}'|g' ${lmp_infile}

lmp_vcsgc -in ${lmp_infile}
