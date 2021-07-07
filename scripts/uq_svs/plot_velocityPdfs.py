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


# PDFs for velocity components
def velocity_pdfs(dir_sol, dir_fig, lx, show_plot, save_plot, plot_id):

    fn = dir_sol+'measuredData_lx'+str(lx)+'.json'
    nsamples, data_vx, data_vy = load_data(fn)
    
    titles = [r'Measured at $(x,y) = (0.50, 0.25)$',
              r'Measured at $(x,y) = (0.50, 0.50)$',
              r'Measured at $(x,y) = (0.50, 0.75)$']
    
    yMaxLims = [35, 70, 140, 280, 560]
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    #myPlotDict['xlim'] = [-0.4, 1.2]
    myPlotDict['xlim'] = [-0.8, 1.25]
    
    if plot_id == 1:
        nbins = [14, 4, 12]
        # nbins = [30, 6, 30]
        myPlotDict['xlabel'] = r'velocity in $x$'
        myPlotDict['ylabel'] = r'number of samples'
        myPlotDict['ylim'] = [0, yMaxLims[lx-3]]
        for n in range(0,3):
            myPlotDict['title'] = titles[n]
            myPlotDict['nbins'] = nbins[n]
            myPlotDict['out_filename'] = dir_fig+"svs_vx"+str(n)+"_lx"+str(lx)+".pdf"
            common.plotHist(data_vx[n*nsamples:(n+1)*nsamples], myPlotDict)
    
    elif plot_id == 2:
        nbins = [12, 12, 12]
        # nbins = [20, 20, 20]
        myPlotDict['xlabel'] = r'velocity in $y$'
        myPlotDict['ylabel'] = r'number of samples'
        myPlotDict['ylim'] = [0, yMaxLims[lx-3]]
        for n in range(0,3):
            myPlotDict['title'] = titles[n]
            myPlotDict['nbins'] = nbins[n]
            myPlotDict['out_filename'] = dir_fig+"svs_vy"+str(n)+"_lx"+str(lx)+".pdf"
            common.plotHist(data_vy[n*nsamples:(n+1)*nsamples], myPlotDict)


if __name__ == "__main__":

    show_plot = True
    save_plot = True
    
    dir_sol = '../../output/uq_pincompNS_svs/'
    dir_fig = '../../figures/uq_incompNS/uq_svs/pdfs/'
    
    lx = 4
    plot_id = 1
    velocity_pdfs(dir_sol, dir_fig, lx, show_plot, save_plot, plot_id)
    
    
# End of file
