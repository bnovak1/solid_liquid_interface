#!/bin/bash

template_file=${1}
restart_file=${2}
box_size_file=${3}
temp=${4}
timestep=${5}
orientation=${6}
outfile=${7}

box_x=`tail -1 ${box_size_file} | awk '{print $1}'`
box_y=`tail -1 ${box_size_file} | awk '{print $2}'`
box_z=`tail -1 ${box_size_file} | awk '{print $3}'`

cp ${template_file} ${outfile}

sed -i 's|\[RESTART_FILE\]|'${restart_file}'|g' ${outfile}
sed -i 's|\[BOXX\]|'${box_x}'|g' ${outfile}
sed -i 's|\[BOXY\]|'${box_y}'|g' ${outfile}
sed -i 's|\[BOXZ\]|'${box_z}'|g' ${outfile}
sed -i 's|\[ORIENTATION\]|'${orientation}'|g' ${outfile}
sed -i 's|\[TEMP\]|'${temp}'|g' ${outfile}
sed -i 's|\[TIMESTEP\]|'${timestep}'|g' ${outfile}
