configfile: 'config.json'

rule pdb_from_lammpstrj:
    input:
        lammpstrj = 'Ts-1613/{X}/kinetic/solidification_{X}.0.lammpstrj',
        script = '../scripts/lammpstrj_to_pdb.tcl'
    output:
        pdb = 'Ts-1613/{X}/kinetic/solidification_{X}.0.pdb',
    shell:
        '''
        echo "source {input.script}" > temp{wildcards.X}.tcl
        echo "lammpstrj_to_pdb {input.lammpstrj} {output.pdb} {config[ELEMENTS]}" >> temp{wildcards.X}.tcl
        echo "exit" >> temp{wildcards.X}.tcl
        vmd -e temp{wildcards.X}.tcl
        rm -f temp{wildcards.X}.tcl
        '''

rule pdb_from_lammpstrjs:
    input:
        expand(rules.pdb_from_lammpstrj.output.pdb, X=config['CONCS'])


rule edit_reference_structure_template:
    input:
        template = 'reference_solid_template.in',
        script = '../scripts/edit_template.py',
        latparam = 'Ts-1613/lattice-initial/lattice_1613.dat'
    output:
        lmp_infile = 'lmp_infiles/{X}/reference_solid.in'
    params:
        orientation = config['ORIENTATIONS']['1'],
        traj = 'Ts-1613/{X}/kinetic/reference.lammpstrj'
    shell:
        'python {input.script} {input.template} {config[LATTICE_TYPE]}'
        ' `cat {input.latparam}` "{params.orientation}" {params.traj} {output.lmp_infile}'

rule edit_reference_structure_templates:
    input:
        expand(rules.edit_reference_structure_template.output.lmp_infile, X=config['CONCS'])

rule create_reference_structure:
    input:
        lmp_infile = rules.edit_reference_structure_template.output.lmp_infile
    output:
        lammpstrj = rules.edit_reference_structure_template.params.traj
    shell:
        'lmp_stable -in {input.lmp_infile}'

rule create_reference_structures:
    input:
        expand(rules.create_reference_structure.output.lammpstrj, X=config['CONCS'])


rule pdb_from_lammpstrj_ref:
    input:
        lammpstrj = rules.create_reference_structure.output.lammpstrj,
        script = '../scripts/lammpstrj_to_pdb.tcl'
    output:
        pdb = str(rules.create_reference_structure.output.lammpstrj).replace('.lammpstrj', '.pdb')
    shell:
        '''
        echo "source {input.script}" > temp{wildcards.X}.tcl
        echo "lammpstrj_to_pdb {input.lammpstrj} {output.pdb} {config[ELEMENTS]}" >> temp{wildcards.X}.tcl
        echo "exit" >> temp{wildcards.X}.tcl
        vmd -e temp{wildcards.X}.tcl
        rm -f temp{wildcards.X}.tcl
        '''

rule pdb_from_lammpstrj_refs:
    input:
        expand(rules.pdb_from_lammpstrj_ref.output.pdb, X=config['CONCS'])


rule interface_concentration:
    input:
        json = 'analysis_infiles/{X}/traj_analysis.json',
        script = '../scripts/solidification_concentration.py'
    output:
        concs = '../analysis/{X}/interface_concs.dat'
    threads: 40
    shell:
        'python {input.script} {input.json}'

rule interface_concentrations:
    input:
        expand(rules.interface_concentration.output.concs, X=config['CONCS'])
