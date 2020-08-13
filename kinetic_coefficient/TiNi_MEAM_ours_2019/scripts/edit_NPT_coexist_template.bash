#!/bin/bash

template_file=${1}
restart_file=${2}
temp=${3}
timestep=${4}
nruns=${5}
orientation=${6}
outfile=${7}

cp ${template_file} ${outfile}

sed -i 's|\[RESTART_FILE\]|'${restart_file}'|g' ${outfile}
sed -i 's|\[TEMP\]|'${temp}'|g' ${outfile}
sed -i 's|\[TIMESTEP\]|'${timestep}'|g' ${outfile}
sed -i 's|\[NRUNS\]|'${nruns}'|g' ${outfile}
sed -i 's|\[ORIENTATION\]|'${orientation}'|g' ${outfile}
