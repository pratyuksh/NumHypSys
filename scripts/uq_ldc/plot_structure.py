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
    return np.array(j["offsets"]), np.array(j["p1"]), np.array(j["p2"]), np.array(j["p3"])


# Structure function of mean field
def structureMean(dir_sol, dir_fig, lx, show_plot, save_plot):
    
    all_offsets = []
    
    all_svalp1 = []
    all_ref_svalp1 = []
    all_slopes_svalp1 = []
    
    all_svalp2 = []
    all_ref_svalp2 = []
    all_slopes_svalp2 = []
    
    all_svalp3 = []
    all_ref_svalp3 = []
    all_slopes_svalp3 = []
    
    plotOutfile1 = dir_fig+'structure_mean_ldc_p1.pdf'
    plotTitle1 = r'Structure function of mean vs offset, $p=1$'
    
    plotOutfile2 = dir_fig+'structure_mean_ldc_p2.pdf'
    plotTitle2 = r'Structure function of mean vs offset, $p=2$'

    plotOutfile3 = dir_fig+'structure_mean_ldc_p3.pdf'
    plotTitle3 = r'Structure function of mean vs offset, $p=3$'
    
    for k in list(lx):
        fn = dir_sol+'structure_mean_lx'+str(k)+'.json'
    
        # read file
        offsets, svalp1, svalp2, svalp3 = load_data(fn)
        n = offsets.size
        m = 4
        if n==2:
            m = 2
        all_offsets.append(offsets)
    
        linfit_svalp1 = np.polyfit(np.log(offsets[0:m]), np.log(svalp1[0:m]), 1)
        ref_svalp1 = np.exp(np.polyval(linfit_svalp1, np.log(offsets))-0.5)
        slope_svalp1 = linfit_svalp1[0]
        #print('\nSlope p1: ', slope_svalp1)
        #print(svalp1)
        all_svalp1.append(svalp1)
        all_ref_svalp1.append(ref_svalp1)
        all_slopes_svalp1.append(slope_svalp1)
    
        linfit_svalp2 = np.polyfit(np.log(offsets[0:m]), np.log(svalp2[0:m]), 1)
        ref_svalp2 = np.exp(np.polyval(linfit_svalp2, np.log(offsets))-0.5)
        slope_svalp2 = linfit_svalp2[0]
        #print('\nSlope p2: ', slope_svalp2)
        #print(svalp2)
        all_svalp2.append(svalp2)
        all_ref_svalp2.append(ref_svalp2)
        all_slopes_svalp2.append(slope_svalp2)
    
        linfit_svalp3 = np.polyfit(np.log(offsets[0:m]), np.log(svalp3[0:m]), 1)
        ref_svalp3 = np.exp(np.polyval(linfit_svalp3, np.log(offsets))-0.5)
        slope_svalp3 = linfit_svalp3[0]
        #print('\nSlope p3: ', slope_svalp3)
        #print(svalp3)
        all_svalp3.append(svalp3)
        all_ref_svalp3.append(ref_svalp3)
        all_slopes_svalp3.append(slope_svalp3)
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    myPlotDict['xlabel'] = ''
    myPlotDict['ylabel'] = ''
    myPlotDict['legend_loc'] = 'upper left'
    myPlotDict['data_markers'] = ['b-', 'r-', 'k-', 'g-', 'y-']
    myPlotDict['data_labels'] = ['$N=32$', '$N=64$', '$N=128$', '$N=256$', '$N=512$']
    myPlotDict['ylim'] = [1E-3, 1.2E-1]
    myPlotDict['xlim'] = [1E-3, 2E-1]
    myPlotDict['ref_data_markers'] = ['b--', 'r--', 'k--', 'g--', 'y--']
    
    # for p=1
    myPlotDict['title'] = plotTitle1
    myPlotDict['out_filename'] = plotOutfile1
    myPlotDict['ref_data_labels'] = ['$O(r^{%4.2f})$'%all_slopes_svalp1[0], '$O(r^{%4.2f})$'%all_slopes_svalp1[1], '$O(r^{%4.2f})$'%all_slopes_svalp1[2], '$O(r^{%4.2f})$'%all_slopes_svalp1[3], '$O(r^{%4.2f})$'%all_slopes_svalp1[4]]
    common.plotLogLogData(all_offsets, all_svalp1, all_ref_svalp1, myPlotDict)
    
    # for p=2
    myPlotDict['title'] = plotTitle2
    myPlotDict['out_filename'] = plotOutfile2
    myPlotDict['ref_data_labels'] = ['$O(r^{%4.2f})$'%all_slopes_svalp2[0], '$O(r^{%4.2f})$'%all_slopes_svalp2[1], '$O(r^{%4.2f})$'%all_slopes_svalp2[2], '$O(r^{%4.2f})$'%all_slopes_svalp2[3], '$O(r^{%4.2f})$'%all_slopes_svalp2[4]]
    common.plotLogLogData(all_offsets, all_svalp2, all_ref_svalp2, myPlotDict)
    
    # for p=3
    myPlotDict['title'] = plotTitle3
    myPlotDict['out_filename'] = plotOutfile3
    myPlotDict['ref_data_labels'] = ['$O(r^{%4.2f})$'%all_slopes_svalp3[0], '$O(r^{%4.2f})$'%all_slopes_svalp3[1], '$O(r^{%4.2f})$'%all_slopes_svalp3[2], '$O(r^{%4.2f})$'%all_slopes_svalp3[3], '$O(r^{%4.2f})$'%all_slopes_svalp3[4]]
    common.plotLogLogData(all_offsets, all_svalp3, all_ref_svalp3, myPlotDict)

# Structure function of fluctuations
def structureFluctuations(dir_sol, dir_fig, lx, show_plot, save_plot):
    
    all_offsets = []
    
    all_svalp1 = []
    all_ref_svalp1 = []
    all_slopes_svalp1 = []
    
    all_svalp2 = []
    all_ref_svalp2 = []
    all_slopes_svalp2 = []
    
    all_svalp3 = []
    all_ref_svalp3 = []
    all_slopes_svalp3 = []
    
    plotOutfile1 = dir_fig+'structure_fluctuations_ldc_p1.pdf'
    plotTitle1 = r'Structure function of fluctuations vs offset, $p=1$'
    
    plotOutfile2 = dir_fig+'structure_fluctuations_ldc_p2.pdf'
    plotTitle2 = r'Structure function of fluctuations vs offset, $p=2$'

    plotOutfile3 = dir_fig+'structure_fluctuations_ldc_p3.pdf'
    plotTitle3 = r'Structure function of fluctuations vs offset, $p=3$'
    
    for k in list(lx):
        fn = dir_sol+'structure_fluctuations_lx'+str(k)+'.json'
    
        # read file
        offsets, svalp1, svalp2, svalp3 = load_data(fn)
        n = offsets.size
        m = 4
        if n==2:
            m = 2
        all_offsets.append(offsets)
    
        linfit_svalp1 = np.polyfit(np.log(offsets[0:m]), np.log(svalp1[0:m]), 1)
        ref_svalp1 = np.exp(np.polyval(linfit_svalp1, np.log(offsets))-0.5)
        slope_svalp1 = linfit_svalp1[0]
        #print('\nSlope p1: ', slope_svalp1)
        #print(svalp1)
        all_svalp1.append(svalp1)
        all_ref_svalp1.append(ref_svalp1)
        all_slopes_svalp1.append(slope_svalp1)
    
        linfit_svalp2 = np.polyfit(np.log(offsets[0:m]), np.log(svalp2[0:m]), 1)
        ref_svalp2 = np.exp(np.polyval(linfit_svalp2, np.log(offsets))-0.5)
        slope_svalp2 = linfit_svalp2[0]
        #print('\nSlope p2: ', slope_svalp2)
        #print(svalp2)
        all_svalp2.append(svalp2)
        all_ref_svalp2.append(ref_svalp2)
        all_slopes_svalp2.append(slope_svalp2)
    
        linfit_svalp3 = np.polyfit(np.log(offsets[0:m]), np.log(svalp3[0:m]), 1)
        ref_svalp3 = np.exp(np.polyval(linfit_svalp3, np.log(offsets))-0.5)
        slope_svalp3 = linfit_svalp3[0]
        #print('\nSlope p3: ', slope_svalp3)
        #print(svalp3)
        all_svalp3.append(svalp3)
        all_ref_svalp3.append(ref_svalp3)
        all_slopes_svalp3.append(slope_svalp3)
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    myPlotDict['xlabel'] = ''
    myPlotDict['ylabel'] = ''
    myPlotDict['legend_loc'] = 'upper left'
    myPlotDict['data_markers'] = ['b-', 'r-', 'k-', 'g-', 'y-']
    myPlotDict['data_labels'] = ['$N=32$', '$N=64$', '$N=128$', '$N=256$', '$N=512$']
    myPlotDict['ylim'] = [2E-4, 2E-2]
    myPlotDict['xlim'] = [1E-3, 2E-1]
    myPlotDict['ref_data_markers'] = ['b--', 'r--', 'k--', 'g--', 'y--']
    
    # for p=1
    myPlotDict['title'] = plotTitle1
    myPlotDict['out_filename'] = plotOutfile1
    myPlotDict['ref_data_labels'] = ['$O(r^{%4.2f})$'%all_slopes_svalp1[0], '$O(r^{%4.2f})$'%all_slopes_svalp1[1], '$O(r^{%4.2f})$'%all_slopes_svalp1[2], '$O(r^{%4.2f})$'%all_slopes_svalp1[3], '$O(r^{%4.2f})$'%all_slopes_svalp1[4]]
    common.plotLogLogData(all_offsets, all_svalp1, all_ref_svalp1, myPlotDict)
    
    # for p=2
    myPlotDict['title'] = plotTitle2
    myPlotDict['out_filename'] = plotOutfile2
    myPlotDict['ref_data_labels'] = ['$O(r^{%4.2f})$'%all_slopes_svalp2[0], '$O(r^{%4.2f})$'%all_slopes_svalp2[1], '$O(r^{%4.2f})$'%all_slopes_svalp2[2], '$O(r^{%4.2f})$'%all_slopes_svalp2[3], '$O(r^{%4.2f})$'%all_slopes_svalp2[4]]
    common.plotLogLogData(all_offsets, all_svalp2, all_ref_svalp2, myPlotDict)
    
    # for p=3
    myPlotDict['title'] = plotTitle3
    myPlotDict['out_filename'] = plotOutfile3
    myPlotDict['ref_data_labels'] = ['$O(r^{%4.2f})$'%all_slopes_svalp3[0], '$O(r^{%4.2f})$'%all_slopes_svalp3[1], '$O(r^{%4.2f})$'%all_slopes_svalp3[2], '$O(r^{%4.2f})$'%all_slopes_svalp3[3], '$O(r^{%4.2f})$'%all_slopes_svalp3[4]]
    common.plotLogLogData(all_offsets, all_svalp3, all_ref_svalp3, myPlotDict)

# Structure function cube
def structureCube(dir_sol, dir_fig, lx, show_plot, save_plot):
    
    all_offsets = []
    
    all_svalp1 = []
    all_ref_svalp1 = []
    all_slopes_svalp1 = []
    
    all_svalp2 = []
    all_ref_svalp2 = []
    all_slopes_svalp2 = []
    
    all_svalp3 = []
    all_ref_svalp3 = []
    all_slopes_svalp3 = []
    
    plotOutfile1 = dir_fig+'structure_cube_ldc_p1.pdf'
    plotTitle1 = r'Structure function vs offset, $p=1$'
    
    plotOutfile2 = dir_fig+'structure_cube_ldc_p2.pdf'
    plotTitle2 = r'Structure function vs offset, $p=2$'

    plotOutfile3 = dir_fig+'structure_cube_ldc_p3.pdf'
    plotTitle3 = r'Structure function vs offset, $p=3$'
    
    for k in list(lx):
        fn = dir_sol+'structure_cube_lx'+str(k)+'.json'
    
        # read file
        offsets, svalp1, svalp2, svalp3 = load_data(fn)
        n = offsets.size
        m = 4
        if n==2:
            m = 2
        all_offsets.append(offsets)
    
        linfit_svalp1 = np.polyfit(np.log(offsets[0:m]), np.log(svalp1[0:m]), 1)
        ref_svalp1 = np.exp(np.polyval(linfit_svalp1, np.log(offsets))-0.5)
        slope_svalp1 = linfit_svalp1[0]
        #print('\nSlope p1: ', slope_svalp1)
        #print(svalp1)
        all_svalp1.append(svalp1)
        all_ref_svalp1.append(ref_svalp1)
        all_slopes_svalp1.append(slope_svalp1)
    
        linfit_svalp2 = np.polyfit(np.log(offsets[0:m]), np.log(svalp2[0:m]), 1)
        ref_svalp2 = np.exp(np.polyval(linfit_svalp2, np.log(offsets))-0.5)
        slope_svalp2 = linfit_svalp2[0]
        #print('\nSlope p2: ', slope_svalp2)
        #print(svalp2)
        all_svalp2.append(svalp2)
        all_ref_svalp2.append(ref_svalp2)
        all_slopes_svalp2.append(slope_svalp2)
    
        linfit_svalp3 = np.polyfit(np.log(offsets[0:m]), np.log(svalp3[0:m]), 1)
        ref_svalp3 = np.exp(np.polyval(linfit_svalp3, np.log(offsets))-0.5)
        slope_svalp3 = linfit_svalp3[0]
        #print('\nSlope p3: ', slope_svalp3)
        #print(svalp3)
        all_svalp3.append(svalp3)
        all_ref_svalp3.append(ref_svalp3)
        all_slopes_svalp3.append(slope_svalp3)
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    myPlotDict['xlabel'] = r'offset $r$'
    myPlotDict['ylabel'] = r'structure function $S^{p}_{r}$'
    myPlotDict['legend_loc'] = 'upper left'
    myPlotDict['data_markers'] = ['b-', 'r-', 'k-', 'g-', 'y-']
    myPlotDict['data_labels'] = ['$N_{\mathbf{x}}=32$', '$N_{\mathbf{x}}=64$', '$N_{\mathbf{x}}=128$', '$N_{\mathbf{x}}=256$', '$N_{\mathbf{x}}=512$']
    myPlotDict['ylim'] = [1E-3, 2E-1]
    myPlotDict['xlim'] = [2E-4, 2E-1]
    myPlotDict['ref_data_markers'] = ['b--', 'r--', 'k--', 'g--', 'y--']
    
    # for p=1
    myPlotDict['title'] = plotTitle1
    myPlotDict['out_filename'] = plotOutfile1
    myPlotDict['ref_data_labels'] = ['$O(r^{%4.2f})$'%all_slopes_svalp1[0], '$O(r^{%4.2f})$'%all_slopes_svalp1[1], '$O(r^{%4.2f})$'%all_slopes_svalp1[2], '$O(r^{%4.2f})$'%all_slopes_svalp1[3], '$O(r^{%4.2f})$'%all_slopes_svalp1[4]]
    common.plotLogLogData(all_offsets, all_svalp1, all_ref_svalp1, myPlotDict)
    
    # for p=2
    myPlotDict['title'] = plotTitle2
    myPlotDict['out_filename'] = plotOutfile2
    myPlotDict['ref_data_labels'] = ['$O(r^{%4.2f})$'%all_slopes_svalp2[0], '$O(r^{%4.2f})$'%all_slopes_svalp2[1], '$O(r^{%4.2f})$'%all_slopes_svalp2[2], '$O(r^{%4.2f})$'%all_slopes_svalp2[3], '$O(r^{%4.2f})$'%all_slopes_svalp2[4]]
    common.plotLogLogData(all_offsets, all_svalp2, all_ref_svalp2, myPlotDict)
    
    # for p=3
    myPlotDict['title'] = plotTitle3
    myPlotDict['out_filename'] = plotOutfile3
    myPlotDict['ref_data_labels'] = ['$O(r^{%4.2f})$'%all_slopes_svalp3[0], '$O(r^{%4.2f})$'%all_slopes_svalp3[1], '$O(r^{%4.2f})$'%all_slopes_svalp3[2], '$O(r^{%4.2f})$'%all_slopes_svalp3[3], '$O(r^{%4.2f})$'%all_slopes_svalp3[4]]
    common.plotLogLogData(all_offsets, all_svalp3, all_ref_svalp3, myPlotDict)

if __name__ == "__main__":

    show_plot = True
    save_plot = True
    
    dir_sol = '../../output/uq_pincompNS_ldc/stats/structure_noBdry/'
    dir_fig = '../../figures/uq_incompNS/uq_ldc/'
    
    lx = [5,6,7,8,9]
    
    #structureMean(dir_sol, dir_fig, lx, show_plot, save_plot)
    #structureFluctuations(dir_sol, dir_fig, lx, show_plot, save_plot)
    structureCube(dir_sol, dir_fig, lx, show_plot, save_plot)
    
    
# End of file
