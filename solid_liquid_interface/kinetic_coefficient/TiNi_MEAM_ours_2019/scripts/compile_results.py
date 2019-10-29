'''
Compile mean interface velocity and kinetic coefficient data + metadata to a JSON file
'''

import argparse
import json
import numpy as np

RESULTS = {}

def main(args_input):
    '''
    main
    '''

    with open(args_input.json_file, 'r') as f:
        json_in = json.load(f)

    RESULTS['description'] = json_in['DESCRIPTION']
    RESULTS['description']['melting temperature'] = {}
    RESULTS['description']['melting temperature']['value'] = 1942.6
    RESULTS['description']['melting temperature']['method'] = \
        'NPH coexistence with about 25000 atoms'
    RESULTS['description']['velocity'] = {}
    RESULTS['description']['velocity']['methods'] = {}
    RESULTS['description']['velocity']['methods']['potential energy'] = \
        'Using potential energy of system which is proportional to the amount of solid due to ' + \
        'latent heat of fusion'
    RESULTS['description']['velocity']['methods']['interface positions'] = \
        'Direct method where interface positions are found using error function fits to ' + \
        'centrosymmetry parameter'
    RESULTS['description']['velocity']['steady state condition'] = \
        'Distance between two interfaces has reduced by more than 12 angstroms'
    RESULTS['description']['velocity']['mean'] = \
        'Mean of 10 independent free solidification simulations'
    RESULTS['description']['velocity']['uncertainty'] = \
        'Half width of 95% confidence interval using standard error of 10 simulations'
    RESULTS['description']['kinetic coefficient'] = {}
    RESULTS['description']['kinetic coefficient']['mean'] = \
        'Slope of interface velocity as a function of melting temperature - temperature'
    RESULTS['description']['kinetic coefficient']['fitting procedure'] = \
        'statsmodels weighted least squares (WLS) function with weights equal to 1/sigma^2 ' + \
        'where sigma are the standard deviations of the interface velocities. Forced through 0,0.'
    RESULTS['description']['kinetic coefficient']['uncertainty'] = \
        'Half width of confidence interval calculated using statsmodels weighted least squares'
    RESULTS['description']['units'] = {}
    RESULTS['description']['units']['temperature'] = 'Kelvin'
    RESULTS['description']['units']['velocity'] = 'meters/second'
    RESULTS['description']['units']['kinetic coefficient'] = 'meters/(second-Kelvin)'

    RESULTS['data'] = {}
    orientations = json_in['ORIENTATIONS']
    for orientation in orientations:
        RESULTS['data'][orientation] = {}

    for orientation, vel_file in zip(orientations, args_input.vel_files):
        data = np.loadtxt(vel_file)
        RESULTS['data'][orientation]['temperature'] = data[:, 0].tolist()
        RESULTS['data'][orientation]['velocity'] = {}
        RESULTS['data'][orientation]['velocity']['potential energy'] = \
            {'mean': data[:, 1].tolist(), 'uncertainty': data[:, 2].tolist()}
        RESULTS['data'][orientation]['velocity']['interface positions'] = \
            {'mean': data[:, 3].tolist(), 'uncertainty': data[:, 4].tolist()}

    for orientation, mu_file in zip(orientations, args_input.mu_files):
        data = np.loadtxt(mu_file)
        RESULTS['data'][orientation]['kinetic_coefficient'] = {}
        RESULTS['data'][orientation]['kinetic_coefficient']['potential energy'] = \
            {'mean': data[0], 'uncertainty': data[1]}
        RESULTS['data'][orientation]['kinetic_coefficient']['interface positions'] = \
            {'mean': data[2], 'uncertainty': data[3]}

    with open(args_input.outfile, 'w') as f:
        json.dump(RESULTS, f, indent=4)


if __name__ == '__main__':

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('json_file', help='Name of JSON config file.')
    PARSER.add_argument('outfile', help='Name of JSON file for output.')
    PARSER.add_argument('-v', '--vel_files', nargs='+',
                        help='Names of files with temperatures & mean interface velocities.')
    PARSER.add_argument('-k', '--mu_files', nargs='+',
                        help='Names of files with kinetic kinetic_coefficients.')

    ARGS_INPUT = PARSER.parse_args()

    main(ARGS_INPUT)
