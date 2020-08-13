import re
import subprocess
import sys
import uncertainties
import numpy as np

sys.path.append('../scripts')
max_threads = int(subprocess.check_output('nproc').decode().split()[0])

configfile: "config.json"


rule graph:
    input:
        'simulations.Snakefile'
    output:
        '../results/rulegraph_simulations.png'
    shell:
        'snakemake -s {input} --rulegraph results | dot -Tpng > {output}'


rule edit_template:
    input:
        template = 'NPT_{phase}_template.in',
        script = '../scripts/edit_template.py'
    output:
        lmp_infile = 'input_files/{potential}/NPT_{phase}.in'
    params:
        temp = lambda wildcards: config['TEMPERATURES'][wildcards.potential]
    shell:
        'python {input.script} {input.template} {wildcards.potential} {params.temp}'
        ' {output.lmp_infile}'

rule edit_templates:
    input:
        [re.sub('{[a-z]*}', '{}', str(rules.edit_template.output)).format(potential, phase) \
         for potential in config['POTENTIALS'] for phase in config['PHASES'][potential]]


rule simulation:
    input:
        lmp_infile = rules.edit_template.output.lmp_infile,
        potential = 'input_files/{potential}/potential.in',
        crystal = 'input_files/{potential}/crystal.in'
    output:
        log = '../output/{potential}/{phase}/NPT.log',
        traj = '../output/{potential}/{phase}/NPT.lammpstrj'
    params:
        nprocs = int(max_threads)
    threads: max_threads
    shell:
        'mpirun -np {params.nprocs} lmp -in {input.lmp_infile}'

rule simulations:
    input:
        [re.sub('{[a-z]*}', '{}', str(rules.simulation.output.log)).format(potential, phase) \
         for potential in config['POTENTIALS'] for phase in config['PHASES'][potential]]


rule extract_log:
    input:
        log = rules.simulation.output.log,
        script = '../scripts/data_from_log.lammps.bash'
    output:
        data = '../analysis/{potential}/{phase}/NPT_0.dat'
    params:
        outprefix = '../analysis/{potential}/{phase}/NPT_'
    shell:
        'bash {input.script} {input.log} {params.outprefix} 1'

rule extract_logs:
    input:
        [re.sub('{[a-z]*}', '{}', str(rules.extract_log.output)).format(potential, phase) \
         for potential in config['POTENTIALS'] for phase in config['PHASES'][potential]]


rule mean_molar_volume:
    input:
        data = rules.extract_log.output.data,
        script = '../scripts/confidence_interval.py'
    output:
        mean = '../analysis/{potential}/{phase}/molarvol_block_error_extrapolation_0.dat'
    params:
        outprefix = '../analysis/{potential}/{phase}/molarvol_'
    threads: max_threads
    shell:
        'python {input.script} {input.data} 2 -op {params.outprefix} -nb 1000 -np {threads}'

rule mean_molar_volumes:
    input:
        [re.sub('{[a-z]*}', '{}', str(rules.mean_molar_volume.output)).format(potential, 'liquid') \
         for potential in config['POTENTIALS'] if 'liquid' in config['PHASES'][potential]]

rule mean_enthalpy:
    input:
        data = rules.extract_log.output.data,
        script = '../scripts/confidence_interval.py'
    output:
        mean = '../analysis/{potential}/{phase}/enthalpy_block_error_extrapolation_0.dat'
    params:
        outprefix = '../analysis/{potential}/{phase}/enthalpy_'
    threads: max_threads
    shell:
        'python {input.script} {input.data} 1 -op {params.outprefix} -nb 1000 -np {threads}'

rule mean_enthalpies:
    input:
        [re.sub('{[a-z]*}', '{}', str(rules.mean_enthalpy.output)).format(potential, phase) \
         for potential in config['POTENTIALS'] for phase in config['PHASES'][potential]]

rule latent_heat:
    input:
        enthalpy_liq = re.sub('{phase}', 'liquid', rules.mean_enthalpy.output.mean),
        enthalpy_sol = re.sub('{phase}', 'solid', rules.mean_enthalpy.output.mean)
    output:
        latent_heat = '../analysis/{potential}/latent_heat.dat'
    run:
        liquid_enth = np.loadtxt(input.enthalpy_liq, usecols=[0, 2])
        liquid_enth = uncertainties.ufloat(*liquid_enth)
        solid_enth = np.loadtxt(input.enthalpy_sol, usecols=[0, 2])
        solid_enth = uncertainties.ufloat(*solid_enth)
        latent_heat = liquid_enth - solid_enth
        np.savetxt(output.latent_heat, [latent_heat.nominal_value, latent_heat.std_dev])

rule latent_heats:
    input:
        [re.sub('{[a-z]*}', '{}', str(rules.latent_heat.output)).format(potential) \
         for potential in config['POTENTIALS'] \
         if 'solid' in config['PHASES'][potential] and 'liquid' in config['PHASES'][potential]]

rule results:
    input:
        rules.mean_molar_volumes.input,
        rules.latent_heats.input
