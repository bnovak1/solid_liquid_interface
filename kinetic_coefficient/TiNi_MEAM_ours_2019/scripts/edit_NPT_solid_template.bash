#!/bin/bash

template_file=${1}
build_template=${2}
build_outfile=${3}
lattice_type=${4}
lattice_param_file=${5}
orientation=${6}
temp=${7}
timestep=${8}
outfile=${9}

lattice_param=`tail -1 ${lattice_param_file}`

cp ${build_template} ${build_outfile}

sed -i 's|\[LATPARAM\]|'${lattice_param}'|g' ${build_outfile}
sed -i 's|\[LATTICE_TYPE\]|'${lattice_type}'|g' ${build_outfile}
sed -i 's|\[ORIENTATION\]|'${orientation}'|g' ${build_outfile}

cp ${template_file} ${outfile}

sed -i 's|\[LATTICE_TYPE\]|'${lattice_type}'|g' ${outfile}
sed -i 's|\[ORIENTATION\]|'${orientation}'|g' ${outfile}
sed -i 's|\[TEMP\]|'${temp}'|g' ${outfile}
sed -i 's|\[TIMESTEP\]|'${timestep}'|g' ${outfile}
