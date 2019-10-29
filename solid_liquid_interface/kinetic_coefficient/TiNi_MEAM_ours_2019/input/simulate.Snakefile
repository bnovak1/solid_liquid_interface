import glob
import subprocess
import sys
sys.path.append('../scripts')
import numpy as np


configfile: 'config.json'

run_location = config['RUN_LOCATION']
assert run_location=='qb' or run_location=='hera3', \
       'Unknown run location. Known are qb or hera3.'

if run_location == 'qb':
    LAMMPS_command = 'source activate LAMMPS; ' + \
                     'mpirun -bind-to core ' + \
                            '-machinefile $PBS_NODEFILE ' + \
                            '-x KMP_AFFINITY=compact ' + \
                            '-x OMP_NUM_THREADS=1 ' + \
                            '-x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/bnovak1/miniconda3/envs/LAMMPS/lib ' + \
                            'lmp -sf omp -pk omp 1 -screen none -in '
elif run_location == 'hera3':
    LAMMPS_command = 'mpirun -np ' + str(int(config['NTHREADS']/2)) + \
                            ' lmp_vcsgc -screen none -in '


nruns = int(np.loadtxt('nruns.dat'))

with open('temperatures.dat') as f:
    temps = f.readlines()
temps = [t.rstrip('\n') for t in temps]

mp = np.loadtxt('melting_point.dat').astype(str)
temps_extended = np.hstack((temps, mp))

lattice_type = config['LATTICE_TYPE']
orientations = config['ORIENTATIONS']


rule graph:
    input:
        'simulate.Snakefile'
    output:
        '../results/rulegraph_simulate.png'
    shell:
        'snakemake -s {input} --rulegraph results | dot -Tpng > {output}'


rule edit_NPT_solid_template_mp:
    input:
        script = '../scripts/edit_NPT_solid_template.bash',
        build = 'build_system_' + lattice_type + '_{orientation}_template.in',
        lattice_param = 'lattice_param_guess.txt',
        template = 'NPT_solid_template_mp.in',
        timestep = 'timestep.dat',
        mp = 'melting_point.dat'
    output:
        lmp_infile = 'lmp_infiles/{orientation}/NPT_solid_mp.in',
        build = 'lmp_infiles/{orientation}/build_system_' + lattice_type + '.in'
    params:
        lattice_type = lattice_type,
    shell:
        'bash {input.script} {input.template} {input.build} {output.build} '
        '{params.lattice_type} {input.lattice_param} {wildcards.orientation} '
        '`cat {input.mp}` `cat {input.timestep}` {output.lmp_infile}'


rule edit_NPT_solid_templates_mp:
    input:
        expand(rules.edit_NPT_solid_template_mp.output.lmp_infile, orientation=orientations)


rule run_NPT_solid_mp:
    input:
        lmp_infile = rules.edit_NPT_solid_template_mp.output.lmp_infile
    output:
        log = '../output/{orientation}/NPT_solid_mp.log',
        restart = '../output/{orientation}/NPT_solid_mp.restart',
        natoms = '../output/{orientation}/natoms_solid.dat'
    params:
        LAMMPS_command = LAMMPS_command
    threads: config['NTHREADS']
    shell:
        '{params.LAMMPS_command} {input.lmp_infile}'

rule run_NPT_solids_mp:
    input:
        expand(rules.run_NPT_solid_mp.output.log, orientation=orientations)


rule extract_NPT_solid_log_mp:
    input:
        script = '../scripts/data_from_log.lammps.bash',
        log = rules.run_NPT_solid_mp.output.log
    output:
        thermo = '../analysis/{orientation}/NPT_solid_mp_thermo_0.dat'
    params:
        thermo_prefix = '../analysis/{orientation}/NPT_solid_mp_thermo_'
    shell:
        'bash {input.script} {input.log} {params.thermo_prefix} 0'

rule extract_NPT_solid_logs_mp:
    input:
        expand(rules.extract_NPT_solid_log_mp.output.thermo, orientation=orientations)


rule solid_box_size_mp:
    input:
        script = '../scripts/box_size_mean.py',
        data = rules.extract_NPT_solid_log_mp.output.thermo
    output:
        box_size = '../analysis/{orientation}/solid_box_size_mp_0.dat',
    params:
        outprefix = '../analysis/{orientation}/solid_box_size_mp_',
        xcol = 1,
        ycol = 2,
        zcol = 3
    threads: config['NTHREADS']
    shell:
        'python {input.script} {input.data} {output.box_size} {params.xcol} '
        '{params.ycol} {params.zcol}'

rule solid_box_sizes_mp:
    input:
        expand(rules.solid_box_size_mp.output.box_size, orientation=orientations)


rule convert_NPT_solid_restart_mp:
    input:
        restart = rules.run_NPT_solid_mp.output.restart,
    output:
        data = str(rules.run_NPT_solid_mp.output.restart).rstrip('restart') + 'data'
    shell:
        '''
        lmp_vcsgc -restart2data {input.restart} {output.data}
        '''

rule convert_NPT_solid_restarts_mp:
    input:
        expand(rules.convert_NPT_solid_restart_mp.output.data, orientation=orientations)


rule edit_NVT_solid_template:
    input:
        restart = rules.run_NPT_solid_mp.output.restart,
        box_size = rules.solid_box_size_mp.output,
        script = '../scripts/edit_NVT_solid_template.bash',
        template = 'NVT_solid_template.in',
        timestep = 'timestep.dat',
        mp = 'melting_point.dat'
    output:
        lmp_infile = 'lmp_infiles/{orientation}/NVT_solid.in'
    shell:
        'bash {input.script} {input.template} {input.restart} {input.box_size} '
        '`cat {input.mp}` `cat {input.timestep}` {wildcards.orientation} '
        '{output.lmp_infile}'

rule edit_NVT_solid_templates:
    input:
        expand(rules.edit_NVT_solid_template.output.lmp_infile, orientation=orientations)


rule run_NVT_solid:
    input:
        lmp_infile = rules.edit_NVT_solid_template.output.lmp_infile
    output:
        log = '../output/{orientation}/NVT_solid.log',
        restart = '../output/{orientation}/NVT_solid.restart'
    params:
        LAMMPS_command = LAMMPS_command
    threads: config['NTHREADS']
    shell:
        '{params.LAMMPS_command} {input.lmp_infile}'

rule run_NVT_solids:
    input:
        expand(rules.run_NVT_solid.output.log, orientation=orientations)


rule convert_NVT_solid_restart:
    input:
        restart = rules.run_NVT_solid.output.restart,
    output:
        data = str(rules.run_NVT_solid.output.restart).rstrip('restart') + 'data'
    shell:
        '''
        lmp_vcsgc -restart2data {input.restart} {output.data}
        '''

rule convert_NVT_solid_restarts:
    input:
        expand(rules.convert_NVT_solid_restart.output.data, orientation=orientations)


rule edit_melt_template:
    input:
        restart = rules.run_NVT_solid.output.restart,
        script = '../scripts/edit_melt_template.bash',
        template = 'melt_template.in',
        timestep = 'timestep.dat',
        melt_temp = 'melt_temperature.dat'
    output:
        lmp_infile = 'lmp_infiles/{orientation}/melt.in'
    params:
        lattice_type = lattice_type
    shell:
        'bash {input.script} {input.template} {input.restart} `cat {input.melt_temp}` '
        '`cat {input.timestep}` {params.lattice_type} {wildcards.orientation} '
        '{output.lmp_infile}'

rule edit_melt_templates:
    input:
        expand(rules.edit_melt_template.output.lmp_infile, orientation=orientations)


rule run_melt:
    input:
        lmp_infile = rules.edit_melt_template.output.lmp_infile,
        build = rules.edit_NPT_solid_template_mp.output.build
    output:
        log = '../output/{orientation}/melt.log',
        restart = '../output/{orientation}/melt.restart',
    params:
        LAMMPS_command = LAMMPS_command
    threads: config['NTHREADS']
    shell:
        '{params.LAMMPS_command} {input.lmp_infile}'

rule run_melts:
    input:
        expand(rules.run_melt.output.log, orientation=orientations)


rule convert_melt_restart:
    input:
        restart = rules.run_melt.output.restart
    output:
        data = str(rules.run_melt.output.restart).rstrip('restart') + 'data'
    shell:
        '''
        lmp_vcsgc -restart2data {input.restart} {output.data}
        '''

rule convert_melt_restarts:
    input:
        expand(rules.convert_melt_restart.output.data, orientation=orientations)

#########################################################################################
# NPT with coexistence at melting temperature
#########################################################################################
rule edit_NPT_coexist_template_mp:
    input:
        restart = rules.run_melt.output.restart,
        script = '../scripts/edit_NPT_coexist_template.bash',
        template = 'NPT_coexist_template.in',
        timestep = 'timestep.dat',
        mp = 'melting_point.dat',
        nruns = 'nruns.dat'
    output:
        lmp_infile = 'lmp_infiles/{orientation}/NPT_coexist_mp.in'
    shell:
        'bash {input.script} {input.template} {input.restart} `cat {input.mp}` '
        '`cat {input.timestep}` `cat {input.nruns}` {wildcards.orientation} '
        '{output.lmp_infile}'

rule edit_NPT_coexist_templates_mp:
    input:
        expand(rules.edit_NPT_coexist_template_mp.output.lmp_infile, orientation=orientations)


rule run_NPT_coexist_mp:
    input:
        lmp_infile = rules.edit_NPT_coexist_template_mp.output.lmp_infile,
        script = '../scripts/run_NPT_coexist_mp.bash'
    output:
        log = '../output/{orientation}/NPT_coexist_mp.log',
        restart = ['../output/{{orientation}}/NPT_coexist_mp.restart.{}'.format(run) \
                   for run in range(1, nruns+1)],
        data = ['../output/{{orientation}}/NPT_coexist_mp.data.{}'.format(run) \
                for run in range(1, nruns+1)]
    params:
        prefix = '../output/{orientation}/NPT_coexist_mp'
    threads: config['NTHREADS']
    shell:
        'bash {input.script} {input.lmp_infile} {params.prefix}'
        #config['LAMMPS COMMAND'] + '-in {input.lmp_infile} -screen none

rule run_NPT_coexists_mp:
    input:
        expand(rules.run_NPT_coexist_mp.output.log, orientation=orientations)


#########################################################################################
# NPT of solid at solidification temperature to determine lattice parameter & enthalpy
#########################################################################################
rule edit_NPT_solid_template:
    input:
        script = '../scripts/edit_NPT_solid_small_template.bash',
        lattice_param = 'lattice_param_guess.txt',
        template = 'NPT_solid_template.in',
        timestep = 'timestep.dat'
    output:
        lmp_infile = 'lmp_infiles/NPT_small/{temp}K/solid.in'
    params:
        lattice_type = lattice_type
    shell:
        'bash {input.script} {input.template} {params.lattice_type} '
        '{input.lattice_param} {wildcards.temp} `cat {input.timestep}` '
        '{output.lmp_infile}'

rule edit_NPT_solid_templates:
    input:
        expand(rules.edit_NPT_solid_template.output.lmp_infile,
               temp=temps_extended)


rule run_NPT_solid:
    input:
        lmp_infile = rules.edit_NPT_solid_template.output.lmp_infile
    output:
        log = '../output/NPT_small/{temp}K/solid.log',
        restart = '../output/NPT_small/{temp}K/solid.restart'
    params:
        LAMMPS_command = LAMMPS_command
    threads: config['NTHREADS']
    shell:
        '{params.LAMMPS_command} {input.lmp_infile}'

rule run_NPT_solids:
    input:
        expand(rules.run_NPT_solid.output.log, temp=temps_extended)


rule extract_NPT_solid_log:
    input:
        script = '../scripts/data_from_log.lammps.bash',
        log = rules.run_NPT_solid.output.log
    output:
        thermo = '../analysis/{temp}K/NPT_solid_thermo_0.dat'
    params:
        thermo_prefix = '../analysis/{temp}K/NPT_solid_thermo_'
    shell:
        'bash {input.script} {input.log} {params.thermo_prefix} 0'

rule extract_NPT_solid_logs:
    input:
        expand(rules.extract_NPT_solid_log.output.thermo, temp=temps_extended)


rule lattice_param:
    input:
        script = '../scripts/confidence_interval.py',
        data = rules.extract_NPT_solid_log.output.thermo
    output:
        lat_param = '../analysis/{temp}K/lattice_param_block_error_extrapolation_0.dat'
    params:
        outprefix = '../analysis/{temp}K/lattice_param_'
    threads: config['NTHREADS']
    shell:
        'python {input.script} {input.data} 2 -op {params.outprefix} -eq 0.0 -nb 1000 '
        '-np ' + str(config['NTHREADS'])

rule lattice_params:
    input:
        expand(rules.lattice_param.output.lat_param, temp=temps_extended)


rule box_sizes:
    input:
        script = '../scripts/box_sizes_undercooled.bash',
        lat_param = rules.lattice_param.output.lat_param,
        template = 'box_sizes_' + lattice_type + '_{orientation}_template.in'
    output:
        lmp_infile = 'lmp_infiles/{orientation}/{temp}K/box_sizes_' + lattice_type + '.in',
        box_sizes = '../analysis/{orientation}/{temp}K/box_sizes.dat'
    shell:
        'bash {input.script} {input.lat_param} {input.template} {output.lmp_infile} {output.box_sizes}'

rule box_sizess:
    input:
        expand(rules.box_sizes.output.box_sizes, temp=temps_extended, orientation=orientations)


rule enthalpy_solid:
    input:
        script = '../scripts/confidence_interval.py',
        data = rules.extract_NPT_solid_log.output.thermo
    output:
        enthalpy = '../analysis/{temp}K/enthalpy_solid_block_error_extrapolation_0.dat',
    params:
        outprefix = '../analysis/{temp}K/enthalpy_solid_'
    threads: config['NTHREADS']
    shell:
        'python {input.script} {input.data} 4 -op {params.outprefix} -eq 0.0 -nb 1000 '
        '-np ' + str(config['NTHREADS'])

rule enthalpy_solids:
    input:
        expand(rules.enthalpy_solid.output.enthalpy, temp=temps_extended)


rule volume_solid:
    input:
        script = '../scripts/confidence_interval.py',
        data = rules.extract_NPT_solid_log.output.thermo
    output:
        volume = '../analysis/{temp}K/volume_solid_block_error_extrapolation_0.dat'
    params:
        outprefix = '../analysis/{temp}K/volume_solid_'
    threads: config['NTHREADS']
    shell:
        'python {input.script} {input.data} 5 -op {params.outprefix} -eq 0.0 -nb 1000 '
        '-np ' + str(config['NTHREADS'])

rule volume_solids:
    input:
        expand(rules.volume_solid.output.volume, temp=temps_extended)


#########################################################################################
# NPT of liquid at solidification temperature to determine enthalpy
#########################################################################################
rule edit_NPT_liquid_template:
    input:
        script = '../scripts/edit_NPT_liquid_template.bash',
        lattice_param = 'lattice_param_guess.txt',
        template = 'NPT_liquid_template.in',
        timestep = 'timestep.dat',
        tmelt = 'melt_temperature.dat',
    output:
        lmp_infile = 'lmp_infiles/NPT_small/{temp}K/liquid.in'
    shell:
        'bash {input.script} {input.template} {input.lattice_param} `cat {input.tmelt}` '
        '{wildcards.temp} `cat {input.timestep}` {output.lmp_infile}'

rule edit_NPT_liquid_templates:
    input:
        expand(rules.edit_NPT_liquid_template.output.lmp_infile, temp=temps_extended)


rule run_NPT_liquid:
    input:
        lmp_infile = rules.edit_NPT_liquid_template.output.lmp_infile
    output:
        log = '../output/NPT_small/{temp}K/liquid.log',
        restart = '../output/NPT_small/{temp}K/liquid.restart'
    params:
        LAMMPS_command = LAMMPS_command
    threads: config['NTHREADS']
    shell:
        '{params.LAMMPS_command} {input.lmp_infile}'

rule run_NPT_liquids:
    input:
        expand(rules.run_NPT_liquid.output.log, temp=temps_extended)


rule extract_NPT_liquid_log:
    input:
        script = '../scripts/data_from_log.lammps.bash',
        log = rules.run_NPT_liquid.output.log
    output:
        thermo = '../analysis/{temp}K/NPT_liquid_thermo_0.dat'
    params:
        thermo_prefix = '../analysis/{temp}K/NPT_liquid_thermo_'
    shell:
        'bash {input.script} {input.log} {params.thermo_prefix} 0'

rule extract_NPT_liquid_logs:
    input:
        expand(rules.extract_NPT_liquid_log.output.thermo, temp=temps_extended)


rule enthalpy_liquid:
    input:
        script = '../scripts/confidence_interval.py',
        data = rules.extract_NPT_liquid_log.output.thermo
    output:
        enthalpy = '../analysis/{temp}K/enthalpy_liquid_block_error_extrapolation_0.dat'
    params:
        outprefix = '../analysis/{temp}K/enthalpy_liquid_'
    threads: config['NTHREADS']
    shell:
        'python {input.script} {input.data} 3 -op {params.outprefix} -eq 0.0 -nb 1000 '
        '-np ' + str(config['NTHREADS'])

rule enthalpy_liquids:
    input:
        expand(rules.enthalpy_liquid.output.enthalpy, temp=temps_extended)


rule volume_liquid:
    input:
        script = '../scripts/confidence_interval.py',
        data = rules.extract_NPT_liquid_log.output.thermo
    output:
        volume = '../analysis/{temp}K/volume_liquid_block_error_extrapolation_0.dat'
    params:
        outprefix = '../analysis/{temp}K/volume_liquid_'
    threads: config['NTHREADS']
    shell:
        'python {input.script} {input.data} 4 -op {params.outprefix} -eq 0.0 -nb 1000 '
        '-np ' + str(config['NTHREADS'])

rule volume_liquids:
    input:
        expand(rules.volume_liquid.output.volume, temp=temps_extended)


#########################################################################################
# Latent heat, volume change on solidification
#########################################################################################
rule latent_heat:
    input:
        script = '../scripts/latent_heat.py',
        liquid = rules.enthalpy_liquid.output.enthalpy,
        solid = rules.enthalpy_solid.output.enthalpy,
        lattice_param = rules.lattice_param.output.lat_param,
        natoms_per_cell = 'natoms_per_cell.dat'
    output:
        latent_heat = '../analysis/{temp}K/latent_heat.dat'
    shell:
        'python {input.script} {input.liquid} {input.solid} {input.lattice_param} '
        '`cat {input.natoms_per_cell}` {output.latent_heat}'

rule latent_heats:
    input:
        expand(rules.latent_heat.output.latent_heat, temp=temps_extended)


rule solidification_volume_change:
    input:
        script = '../scripts/solidification_volume_change.py',
        liquid = rules.volume_liquid.output.volume,
        solid = rules.volume_solid.output.volume,
        lattice_param = rules.lattice_param.output.lat_param,
        natoms_per_cell = 'natoms_per_cell.dat'
    output:
        volume_change = '../analysis/{temp}K/solidification_volume_change.dat'
    shell:
        'python {input.script} {input.liquid} {input.solid} {input.lattice_param} '
        '`cat {input.natoms_per_cell}` {output.volume_change}'

rule solidification_volume_changes:
    input:
        expand(rules.solidification_volume_change.output.volume_change,
               temp=temps_extended)


#########################################################################################
# Solidification to measure interface velocity
#########################################################################################
rule n_temp_bins:
    input:
        temp_bin_width = 'temperature_bin_width_set.dat',
        data = lambda wildcards: rules.run_NPT_coexist_mp.output.data[int(wildcards.run)-1]
    output:
        temp_bin_width = '../analysis/{orientation}/{run}/temperature_bin_width_actual.dat',
        n_temp_bins = '../analysis/{orientation}/{run}/n_temperature_bins.dat'
    shell:
        '''
        temp_bin_width=`cat {input.temp_bin_width}`
        box_norm=`cat {input.data} | awk '$3 == "zlo" {{printf("%.20f", $2 - $1)}}'`
        n_temp_bins=`echo $box_norm $temp_bin_width | awk '{{print int($1/$2) + 1}}'`
        temp_bin_width=`echo "$box_norm/$n_temp_bins" | bc -l`

        echo $n_temp_bins > {output.n_temp_bins}
        echo $temp_bin_width > {output.temp_bin_width}
        '''

rule n_temp_binss:
    input:
        expand(rules.n_temp_bins.output.n_temp_bins, run=range(1, nruns+1),
               orientation=orientations)


rule edit_solidification_template:
    input:
        restart = '../output/{orientation}/NPT_coexist_mp.restart.{run}',
        script = '../scripts/edit_solidification_template.bash',
        box_sizes = rules.box_sizes.output.box_sizes,
        template = 'solidification_template.in',
        lmp_infile = rules.edit_NPT_solid_template_mp.output.lmp_infile,
        timestep = 'timestep.dat',
        n_temp_bins = rules.n_temp_bins.output.n_temp_bins
    output:
        lmp_infile = 'lmp_infiles/{orientation}/{temp}K/{run}/solidification.in',
    shell:
        'bash {input.script} {input.template} {input.restart} {input.box_sizes} '
        '{input.lmp_infile} {wildcards.temp} `cat {input.n_temp_bins}` '
        '`cat {input.timestep}` {wildcards.run} {wildcards.orientation} '
        '{output.lmp_infile}'

rule edit_solidification_templates:
    input:
        expand(rules.edit_solidification_template.output.lmp_infile,
               orientation=orientations, temp=temps, run=range(1, nruns+1))


rule run_solidification:
    input:
        python = '../scripts/fit_centrosymmetry_param.py',
        lmp_infile = rules.edit_solidification_template.output.lmp_infile
    output:
        log = '../output/{orientation}/{temp}K/{run}/solidification.log',
        restart = '../output/{orientation}/{temp}K/{run}/solidification.restart',
        area = '../output/{orientation}/{temp}K/{run}/area.dat',
        interface_fit = '../output/{orientation}/{temp}K/{run}/interface_fit.dat'
    params:
        LAMMPS_command = LAMMPS_command
    threads: config['NTHREADS']
    shell:
        '{params.LAMMPS_command} {input.lmp_infile}'

rule run_solidifications:
    input:
        expand(rules.run_solidification.output.log,
               temp=temps, orientation=orientations, run=range(1, nruns+1))


rule extract_solidification_log:
    input:
        script = '../scripts/data_from_log.lammps.bash',
        log = rules.run_solidification.output.log
    output:
        thermo = '../analysis/{orientation}/{temp}K/{run}/solidification_thermo_0.dat'
    params:
        thermo_prefix = '../analysis/{orientation}/{temp}K/{run}/solidification_thermo_'
    shell:
        'bash {input.script} {input.log} {params.thermo_prefix} 1'

rule extract_solidification_logs:
    input:
        expand(rules.extract_solidification_log.output.thermo,
               temp=temps, orientation=orientations, run=range(1, nruns+1))

####################################################################################################
# Interface velocities
####################################################################################################
rule interface_velocity:
    input:
        script = '../scripts/interface_velocity.py',
        thermo = rules.extract_solidification_log.output.thermo,
        latent_heat = rules.latent_heat.output.latent_heat,
        volume_change = rules.solidification_volume_change.output.volume_change,
        area = rules.run_solidification.output.area,
        interface_fit = rules.run_solidification.output.interface_fit
    output:
        velocity = '../results/{orientation}/{temp}K/{run}/velocity.dat'
    params:
        pe_col = 1,
        depth_col = 2
    shell:
        'python {input.script} {input.thermo} {params.pe_col} {params.depth_col} '
        '{input.latent_heat} {input.volume_change} {input.area} {input.interface_fit} '
        '{output.velocity}'

rule interface_velocities:
    input:
        expand(rules.interface_velocity.output.velocity,
               temp=temps, orientation=orientations, run=range(1, nruns+1))

rule interface_velocity_mean:
    input:
        velocities = expand('../results/{{orientation}}/{{temp}}K/{run}/velocity.dat',
                            run=range(1, nruns+1)),
        script = '../scripts/interface_velocity_mean.py'
    output:
        vel_mean = '../results/{orientation}/{temp}K/velocity_mean.dat'
    shell:
        'python {input.script} {output.vel_mean} {input.velocities}'

rule interface_velocity_means:
    input:
        expand(rules.interface_velocity_mean.output, temp=temps, orientation=orientations)


####################################################################################################
# Kinetic coefficient
####################################################################################################
rule kinetic_coefficient:
    input:
        vel_mean = expand('../results/{{orientation}}/{temp}K/velocity_mean.dat', temp=temps),
        script = '../scripts/kinetic_coefficient.py',
        temperatures = 'temperatures.dat',
        nruns = 'nruns.dat',
        mp = 'melting_point.dat'
    output:
        mu = '../results/{orientation}/kinetic_coefficient.dat',
        vel = '../results/{orientation}/interface_velocities_mean.dat'
    shell:
        'python {input.script} `cat {input.nruns}` {input.temperatures} `cat {input.mp}`'
        ' {output.vel} {output.mu} {input.vel_mean}'

rule kinetic_coefficients:
    input:
        expand(rules.kinetic_coefficient.output, orientation=orientations)

####################################################################################################
rule compile_results:
    input:
        mu = expand(rules.kinetic_coefficient.output.mu, orientation=orientations),
        vel = expand(rules.kinetic_coefficient.output.vel, orientation=orientations),
        script = '../scripts/compile_results.py'
    output:
        results = '../results/RESULTS.json'
    shell:
        'python {input.script} config.json {output.results} -v {input.vel} -k {input.mu}'


rule edit_submit_template:
    input:
        template = 'submit_template.pbs',
        script = '../scripts/edit_pbs_template.bash'
    output:
        'pbs_files/submit_{temp}K_{orientation}_{run}.pbs'
    shell:
        'bash {input.script} {input.template} {wildcards.temp} {wildcards.orientation} {wildcards.run}'

rule edit_submit_templates:
    input:
        expand(rules.edit_submit_template.output, temp=temps, orientation=orientations,
               run=range(1, nruns+1))

rule submission:
    input:
        pbs = rules.edit_submit_template.output
    output:
        '../output/submissions/submitted_{temp}K_{orientation}_{run}'
    shell:
        'qsub {input.pbs}; touch {output}'

rule submissions:
    input:
        expand(rules.submission.output, temp=temps, orientation=orientations,
               run=range(1, nruns+1))
