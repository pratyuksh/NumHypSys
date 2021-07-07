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
    return np.array(j["Nx"]), np.array(j["vx"]), np.array(j["vy"]), np.array(j["v"]), np.array(j["vector"])


# wasserstein metrics for smooth vortex sheet on quadrilateral meshes
def wasserstein_quadMeshes(dir_sol, dir_fig, metric, show_plot, save_plot):
    
    fn = dir_sol+'wasserstein_'+str(metric)+'pt.json'
    plotOutfile = dir_fig+'wasserstein_svs_'+str(metric)+'pt.pdf'
    # plotTitle = r'{Wasserstein distance between $N\times N$ and $2N\times 2N$, '+str(metric)+'-pt. dist.}'
    plotTitle = r'Wasserstein '+str(metric)+r'-pt. between $N\times N$ and $2N\times 2N$'
    yLabel = r'$W^{'+str(metric)+'}$ [log]'
    
    if metric == 1:
        yLim = [2e-3, 1e-1]
    elif metric == 2:
        yLim = [4e-3, 1e-1]
    
    # read file
    Nx, wdVx, wdVy, wdV, wdVec = load_data(fn)
    
    # linear fitting
    linfit_wdVx = np.polyfit(np.log(Nx), np.log(wdVx), 1)
    ref_wdVx = np.exp(np.polyval(linfit_wdVx, np.log(Nx))+0.1)
    slope_wdVx = linfit_wdVx[0]
    
    linfit_wdVy = np.polyfit(np.log(Nx), np.log(wdVy), 1)
    ref_wdVy = np.exp(np.polyval(linfit_wdVy, np.log(Nx))-0.1)
    slope_wdVy = linfit_wdVy[0]
    
    linfit_wdV = np.polyfit(np.log(Nx), np.log(wdV), 1)
    ref_wdV = np.exp(np.polyval(linfit_wdV, np.log(Nx))+0.1)
    slope_wdV = linfit_wdV[0]
    
    linfit_wdVec = np.polyfit(np.log(Nx), np.log(wdVec), 1)
    ref_wdVec = np.exp(np.polyval(linfit_wdVec, np.log(Nx))+0.1)
    slope_wdVec = linfit_wdVec[0]
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    myPlotDict['xlabel'] = r'Mesh $2N \times 2N$ [log]'
    myPlotDict['legend_loc'] = 'upper right'
    myPlotDict['data_markers'] = ['bo-', 'rs-', 'gd-', 'mo-']
    #myPlotDict['data_labels'] = ['$u_x$', '$u_y$', '$|\mathbf{u}|$', '$\mathbf{u}$']
    myPlotDict['data_labels'] = ['$|\mathbf{u}|$', '$\mathbf{u}$']
    myPlotDict['ylim'] = yLim
    myPlotDict['ref_data_markers'] = ['b-.', 'r--', 'g-.', 'm--']
    
    myPlotDict['title'] = plotTitle
    myPlotDict['ylabel'] = yLabel
    myPlotDict['out_filename'] = plotOutfile
    myPlotDict['xlim'] = [32, 1024]
    #myPlotDict['ref_data_labels'] = ['$O(N^{%4.2f})$'%slope_wdVx, '$O(N^{%4.2f})$'%slope_wdVy, '$O(N^{%4.2f})$'%slope_wdV, '$O(N^{%4.2f})$'%slope_wdVec]
    myPlotDict['ref_data_labels'] = ['$O(N^{%4.2f})$'%slope_wdV, '$O(N^{%4.2f})$'%slope_wdVec]
    #common.plotLogLogData([Nx, Nx, Nx, Nx], [wdVx, wdVy, wdV, wdVec], [ref_wdVx, ref_wdVy, ref_wdV, ref_wdVec], myPlotDict)
    common.plotLogLogData([Nx, Nx], [wdV, wdVec], [ref_wdV, ref_wdVec], myPlotDict)


if __name__ == "__main__":
    
    show_plot = True
    save_plot = True
    
    dir_sol = '../../output/uq_pincompNS_svs/convergence/'
    dir_fig = '../../figures/uq_incompNS/uq_svs/'
    
    metric = 1
    wasserstein_quadMeshes(dir_sol, dir_fig, metric, show_plot, save_plot)
    
# End of file
