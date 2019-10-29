#!/bin/bash

template_file=${1}
lattice_type=${2}
lattice_param_file=${3}
temp=${4}
timestep=${5}
outfile=${6}

lattice_param=`tail -1 ${lattice_param_file}`

cp ${template_file} ${outfile}

sed -i 's|\[LATTICE_TYPE\]|'${lattice_type}'|g' ${outfile}
sed -i 's|\[LATPARAM\]|'${lattice_param}'|g' ${outfile}
sed -i 's|\[TEMP\]|'${temp}'|g' ${outfile}
sed -i 's|\[TIMESTEP\]|'${timestep}'|g' ${outfile}
