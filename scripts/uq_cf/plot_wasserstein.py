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
    return np.array(j["M"]), np.array(j["vx"]), np.array(j["vy"]), np.array(j["v"]), np.array(j["vector"])


# wasserstein metrics
def wasserstein(dir_sol, dir_fig, lx, metric, show_plot, save_plot):
    
    fn = dir_sol+'wasserstein_cf_lx'+str(lx)+'_'+str(metric)+'pt.json'
    plotOutfile = dir_fig+'wasserstein_cf_lx'+str(lx)+'_'+str(metric)+'pt.pdf'
    plotTitle = r'Wasserstein '+str(metric)+'-pt. dist.'
    yLabel = r'Wasserstein dist. $W^{'+str(metric)+'}$ [log]'
    
    if metric == 1:
        yLim = [1e-3, 1e-2]
    elif metric == 2:
        yLim = [3e-3, 2e-2]
    
    # read file
    M, wdVx, wdVy, wdV, wdVec = load_data(fn)
    
    # linear fitting
    linfit_wdV = np.polyfit(np.log(M), np.log(wdV), 1)
    ref_wdV = np.exp(np.polyval(linfit_wdV, np.log(M))+0.1)
    slope_wdV = linfit_wdV[0]
    
    linfit_wdVec = np.polyfit(np.log(M), np.log(wdVec), 1)
    ref_wdVec = np.exp(np.polyval(linfit_wdVec, np.log(M))+0.1)
    slope_wdVec = linfit_wdVec[0]
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    myPlotDict['xlabel'] = r'Number of samples $M$ [log]'
    myPlotDict['legend_loc'] = 'upper right'
    myPlotDict['data_markers'] = ['bo-', 'rs-']
    myPlotDict['data_labels'] = ['$|\mathbf{u}|$', '$\mathbf{u}$']
    myPlotDict['ylim'] = yLim
    myPlotDict['ref_data_markers'] = ['b-.', 'r--']
    
    myPlotDict['title'] = plotTitle
    myPlotDict['ylabel'] = yLabel
    myPlotDict['out_filename'] = plotOutfile
    myPlotDict['xlim'] = [20, 500]
    myPlotDict['ref_data_labels'] = ['$O(M^{%4.2f})$'%slope_wdV, '$O(M^{%4.2f})$'%slope_wdVec]
    common.plotLogLogData([M, M], [wdV, wdVec], [ref_wdV, ref_wdVec], myPlotDict)


if __name__ == "__main__":
    
    show_plot = True
    save_plot = True
    
    # Re = 1600
    Re = 3200
    
    dir_sol = '../../output/uq_pincompNS_cf/Re'+str(Re)+'/convergence/'
    dir_fig = '../../figures/uq_incompNS/uq_cf/Re'+str(Re)+'/'
    
    lx_ = 3
    metric = 2
    wasserstein(dir_sol, dir_fig, lx_, metric, show_plot, save_plot)
    
    
# End of file
