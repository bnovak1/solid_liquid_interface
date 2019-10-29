#!/bin/bash

template_file=${1}
temp=${2}
orientation=${3}
run=${4}

outfile=pbs_files/submit_${temp}K_${orientation}_${run}.pbs

cp ${template_file} ${outfile}

sed -i 's|\[TEMP\]|'${temp}'|g' ${outfile}
sed -i 's|\[ORIENTATION\]|'${orientation}'|g' ${outfile}
sed -i 's|\[RUN\]|'${run}'|g' ${outfile}


