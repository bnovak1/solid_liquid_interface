import glob
import re
import numpy as np

concs = ['0.1', '0.045']

# Get pdb files with beta equal to bin number if atom is in a bin, otherwise 0
rule interface_concentration:
    input:
        json = 'analysis_infiles/conc_visualization/{x}/traj_analysis.json',
        script = '../scripts/solidification_concentration.py',
        lammpstrj_files = glob.glob('data_files/conc_visualization/{x}/solidification_{x}.*.lammpstrj')
    output:
        concs = '../analysis/conc_visualization/{x}/interface_concs.dat',
        pdb_files = [f.replace('.lammpstrj', '.pdb') for f in \
                     glob.glob('data_files/conc_visualization/{x}/solidification_{x}.*.lammpstrj')]
    threads: 40
    shell:
        'python {input.script} {input.json}'

rule interface_concentrations:
    input:
        expand(rules.interface_concentration.output, x=concs)


# Get average concentrations for non-overlapping bins from data with 6 bins per layer
rule convert_average_concs:
    input:
        concs = 'data_files/conc_visualization/{x}/conc_output_L.dat'
    output:
        concs = 'data_files/conc_visualization/{x}/conc_nonoverlapping_bins.dat'
    run:
        data = np.loadtxt(input.concs)
        np.savetxt(output.concs, data[0, ::6])

rule convert_average_concss:
    input:
        expand(rules.convert_average_concs.output, x=concs)


# Convert beta to concentration in pdb files
rule convert_beta_to_conc:
    input:
        pdb = 'data_files/conc_visualization/{x}/interfaces/solidification_{x}.{step}.pdb',
        concs = rules.convert_average_concs.output.concs,
        script = '../scripts/tempfactors_to_conc.py'
    output:
        pdb = '../analysis/conc_visualization/{x}/concs_{step}.pdb'
    shell:
        'python {input.script} {input.pdb} {input.concs} {output.pdb}'

rule convert_beta_to_concs:
    input:
        [re.sub('{[a-z0-9]*}', '{}', str(rules.convert_beta_to_conc.output)).format(x, step) \
         for x in concs for step in \
         [f.split('.')[3] for f in \
          glob.glob('data_files/conc_visualization/' + x + '/solidification_' + x + '.*.lammpstrj')]]
