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
    return j["Nx"], np.array(j["mean"]), np.array(j["variance"])


# convergence results for lid driven cavity on quadrilateral meshes
def convg_quadMeshes(dir_sol, dir_fig, show_plot, save_plot, plot_id):

    fn = dir_sol+'convergence_ldc.json'
    Nx, meanL1Error, varL1Error = load_data(fn)
    
    # linear fitting
    linfit_mean = np.polyfit(np.log(Nx), np.log(meanL1Error), 1)
    linfit_var = np.polyfit(np.log(Nx), np.log(varL1Error), 1)
    ref_mean = np.exp(np.polyval(linfit_mean, np.log(Nx))-0.5)
    ref_var = np.exp(np.polyval(linfit_var, np.log(Nx))-0.5)
    slope_mean = linfit_mean[0]
    slope_var = linfit_var[0]
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    # plot error vs mesh size
    myPlotDict['xlabel'] = r'Mesh $2N_{\mathbf{x}}\!\times\!2N_{\mathbf{x}}$ [log]'
    myPlotDict['legend_loc'] = 'upper right'
    myPlotDict['data_markers'] = ['bo-', 'rs-']
    myPlotDict['data_labels'] = ['mean', 'variance']
    myPlotDict['ylim'] = [4e-6, 1e-0]
    myPlotDict['ref_data_markers'] = ['b-.', 'r--']
    
    myPlotDict['title'] = r'Error between $N_{\mathbf{x}}\!\times\!N_{\mathbf{x}}$ and $2N_{\mathbf{x}}\!\times\! 2N_{\mathbf{x}}$'#r'$\left||\Delta \right||_{L^1}(\mathbf{\mu}^{\mathcal{N}, T})$ and $\left||\Delta\right||_{L^1}((\mathbf{s}^2)^{\mathcal{N}, T})$'
    myPlotDict['ylabel'] = 'Cauchy error'#r'$\left| \mu_{2N_{\mathbf{x}}} - \mu_{N_{\mathbf{x}}} \right|_{1}$ / $\left| s^2_{2N_{\mathbf{x}}} - s^2_{N_{\mathbf{x}}} \right|_{1}$ [log]'
    myPlotDict['out_filename'] =dir_fig+"convg_uq_ldc.pdf"
    myPlotDict['xlim'] = [32, 1024]
    myPlotDict['ref_data_labels'] = ['$O(N_{\mathbf{x}}^{%4.2f})$'%slope_mean, '$O(N_{\mathbf{x}}^{%4.2f})$'%slope_var]
    common.plotLogLogData([Nx, Nx], [meanL1Error, varL1Error], [ref_mean, ref_var], myPlotDict)


if __name__ == "__main__":

    show_plot = True
    save_plot = True
    
    dir_sol = '../../output/uq_pincompNS_ldc/convergence/'
    dir_fig = '../../figures/uq_incompNS/uq_ldc/'
    
    convg_quadMeshes(dir_sol, dir_fig, show_plot, save_plot, 1)
    
    
# End of file
