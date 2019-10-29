#!/bin/bash

# Input file name
infile=${1}

# Prefix for ouput file, extension is .dat
out_file_prefix=${2}

# Separate data sections to one file (1) or multiple files (0)
one_file_flag=${3}

# Column label for first column of data, default is "Time"
first_col=${4}
if [ ${#first_col} == 0 ]; then
    first_col=Time
fi

# Remove any numpy (.npy) files created for viscosity calculation
rm -f ${out_file_prefix}[0-9]*.npy

# Lines for beginnings and ends of data sections in infile
line_start=( `cat ${infile} | awk '{print NR, $1, $2}' | \
                              awk '$2 == "'${first_col}'" && $3 != "spent" && $3 != "step"  {print $1}'` )
line_end=( `cat ${infile} | awk '{print NR, $1}' | awk '$2 == "Loop" {print $1}'` )

# Number of lines found for beginnings and ends of data sections
nstart=${#line_start[@]}
nend=${#line_end[@]}

# It is possible that there are data sections where the thermo_style custom command
# with time as the first value was not specified for a run. Ignore those sections.
if [ ${nend} -gt ${nstart} ]; then

    ind_start_save=`echo "${nend} - ${nstart}" | bc`
    ind_end_save=`echo "${nend}-1" | bc`

    line_end2=( ${line_end[${ind_start_save}]} )

    ind_start_save=`echo "${ind_start_save} + 1" | bc`

    for i in `seq ${ind_start_save} ${ind_end_save}`; do
        line_end2=( ${line_end2[*]} ${line_end[${i}]} )
    done

    unset line_end
    line_end=${line_end2[*]}

fi

# Write to file(s)

nsections=${nstart}
nsections_m1=`echo $nsections - 1 | bc`

if [ ${one_file_flag} == 1 ]; then

    # Write each data section to 1 file: ${out_file_prefix}0.dat,

    header=`cat ${infile} | awk 'NR == '${line_start[0]}''`
    echo "# ${header}" > ${out_file_prefix}0.dat
    for i in `seq 0 $nsections_m1`; do
        cat ${infile} | awk 'NR > '${line_start[$i]}'+1 && NR < '${line_end[$i]}'' \
                      | awk '$1 ~ /[0-9]/' | cat >> ${out_file_prefix}0.dat
    done

elif [ ${one_file_flag} == 0 ]; then

    # Write each data section to separate files: ${out_file_prefix}0.dat,
    # ${out_file_prefix}1.dat, ...

    for i in `seq 0 $nsections_m1`; do
        header=`cat ${infile} | awk 'NR == '${line_start[$i]}''`
        echo "# ${header}" > ${out_file_prefix}${i}.dat
        cat ${infile} | awk 'NR > '${line_start[$i]}' && NR < '${line_end[$i]}'' \
                      | awk '$1 ~ /[0-9]/' | cat >> ${out_file_prefix}${i}.dat
    done

fi
