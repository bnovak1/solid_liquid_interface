import sys

sys.path.append('../scripts')

rule fitting:
    input:
        excel = '../analysis/InterfacialEnergy.xlsx',
        script = '../scripts/fit.py'
    output:
        fits = ['../results/fit_all.json',
                '../results/fit_fcc.json'],
        plots = ['../results/fit_all_structure.png',
                 '../results/fit_all_type.png',
                 '../results/fit_fcc.png'],
        other = '../results/RMSE_compare_Kaptay_fcc.csv'
    run:
        from fit import InterfacialFreeEnergyFit
        fitter = InterfacialFreeEnergyFit(input.excel)
        fitter.fit_all()
        fitter.fit_exp()
        fitter.fit_fcc()
