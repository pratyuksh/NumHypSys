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
    return j["nsamples"], np.array(j["vx"]), np.array(j["vy"])


# PDF for velocity components, uq ldc
def velocity_pdfs(dir_sol, dir_fig, lx, show_plot, save_plot, plot_id):

    fn = dir_sol+'measuredData_lx'+str(lx)+'.json'
    nsamples, data_vx, data_vy = load_data(fn)
    
    titles = [r'Measured at $(x,y) = (0.50, 0.50)$',
              r'Measured at $(x,y) = (0.25, 0.25)$',
              r'Measured at $(x,y) = (0.25, 0.75)$',
              r'Measured at $(x,y) = (0.75, 0.25)$',
              r'Measured at $(x,y) = (0.75, 0.75)$']
    
    yMaxLims = [35, 70, 140, 280, 560]
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    myPlotDict['xlim'] = [-0.3, 0.3]
    # myPlotDict['xlim'] = [-1, 1]
    
    if plot_id == 1:
        nbins = [10, 10, 10, 10, 10]
        myPlotDict['xlabel'] = r'velocity in $x$'
        myPlotDict['ylabel'] = r'number of samples'
        myPlotDict['ylim'] = [0, yMaxLims[lx-5]]
        for n in range(0,5):
            myPlotDict['title'] = titles[n]
            myPlotDict['nbins'] = nbins[n]
            myPlotDict['out_filename'] = dir_fig+"ldc_vx"+str(n)+"_lx"+str(lx)+".pdf"
            common.plotHist(data_vx[n*nsamples:(n+1)*nsamples], myPlotDict)
    
    elif plot_id == 2:
        nbins = [10, 10, 10, 10, 10]
        myPlotDict['xlabel'] = r'velocity in $y$'
        myPlotDict['ylabel'] = r'number of samples'
        myPlotDict['ylim'] = [0, yMaxLims[lx-5]]
        for n in range(0,5):
            myPlotDict['title'] = titles[n]
            myPlotDict['nbins'] = nbins[n]
            myPlotDict['out_filename'] = dir_fig+"ldc_vy"+str(n)+"_lx"+str(lx)+".pdf"
            common.plotHist(data_vy[n*nsamples:(n+1)*nsamples], myPlotDict)


if __name__ == "__main__":

    show_plot = True
    save_plot = True
    
    dir_sol = '../../output/uq_pincompNS_ldc/'
    dir_fig = '../../figures/uq_incompNS/uq_ldc/pdfs/'
    
    lx = 5
    plot_id = 2
    velocity_pdfs(dir_sol, dir_fig, lx, show_plot, save_plot, plot_id)
    
    
# End of file
