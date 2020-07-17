configfile: 'config.json'

rule copy_singularity:
    input:
        definition = '/home/bnovak1/singularity/CIMM.def'
    output:
        definition = '../scripts/CIMM.def'
    shell:
        'cp {input.definition} {output.definition}'

include: '/home/bnovak1/CODE/Work/misc/IO/FFS.Snakefile'

rule results:
    input:
        rules.copy_singularity.output,
        rules.edit_ffs_templates.input
