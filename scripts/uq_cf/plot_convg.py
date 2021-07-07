#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import common


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def load_data(filename):
    j = load_json(filename)
    return j["nsamples"], np.array(j["mean"]), np.array(j["variance"])


# convergence results channel flow
def convgMeanVar(dir_sol, dir_fig, lx, show_plot, save_plot, plot_id):

    fn = dir_sol+'convergence_cf_lx'+str(lx)+'.json'
    M, meanL1Error, varL1Error = load_data(fn)
    
    # linear fitting
    linfit_mean = np.polyfit(np.log(M), np.log(meanL1Error), 1)
    linfit_var = np.polyfit(np.log(M), np.log(varL1Error), 1)
    ref_mean = np.exp(np.polyval(linfit_mean, np.log(M))-0.5)
    ref_var = np.exp(np.polyval(linfit_var, np.log(M))-0.5)
    slope_mean = linfit_mean[0]
    slope_var = linfit_var[0]
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    # plot error vs mesh size
    myPlotDict['xlabel'] = r'Number of samples $M$ [log]'
    myPlotDict['legend_loc'] = 'upper right'
    myPlotDict['data_markers'] = ['bo-', 'rs-']
    myPlotDict['data_labels'] = ['mean', 'variance']
    myPlotDict['ylim'] = [2e-5, 1e-2]
    #myPlotDict['ylim'] = [2e-5, 6e-3]
    myPlotDict['ref_data_markers'] = ['b-.', 'r--']
    
    myPlotDict['title'] = ''#r'$\left||\Delta \right||_{L^1}(\mathbf{\mu}^{\mathcal{N}, T})$ and $\left||\Delta\right||_{L^1}((\mathbf{s}^2)^{\mathcal{N}, T})$'
    myPlotDict['ylabel'] = 'Cauchy error'#r'$\left| \mu_{2N_{\mathbf{x}}} - \mu_{N_{\mathbf{x}}} \right|_{1}$ / $\left| s^2_{2N_{\mathbf{x}}} - s^2_{N_{\mathbf{x}}} \right|_{1}$ [log]'
    myPlotDict['out_filename'] =dir_fig+'convg_uq_cf_lx'+str(lx)+'.pdf'
    myPlotDict['xlim'] = [20, 600]
    myPlotDict['ref_data_labels'] = ['$O(M^{%4.2f})$'%slope_mean, '$O(M^{%4.2f})$'%slope_var]
    common.plotLogLogData([M, M], [meanL1Error, varL1Error], [ref_mean, ref_var], myPlotDict)


if __name__ == "__main__":

    show_plot = True
    save_plot = True
    
    # Re = 1600
    Re = 3200
    
    dir_sol = '../../output/uq_pincompNS_cf/Re'+str(Re)+'/convergence/'
    dir_fig = '../../figures/uq_incompNS/uq_cf/Re'+str(Re)+'/'
    
    lx = 3
    convgMeanVar(dir_sol, dir_fig, lx, show_plot, save_plot, 1)
    
    
# End of file
