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


# PDF for velocity components, Reynolds number 1600
def velocity_pdfs(dir_sol, dir_fig, lx, show_plot, save_plot, plot_id):

    fn = dir_sol+'measuredData_lx'+str(lx)+'.json'
    nsamples, data_vx, data_vy = load_data(fn)
    
    titles = [r'Measured at $(x,y) = (1, 0.1)$',
              r'Measured at $(x,y) = (1, 0.2)$',
              r'Measured at $(x,y) = (1, 0.3)$',
              r'Measured at $(x,y) = (1, 0.4)$']
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    if plot_id == 1:
        xLims = np.array([[0.8, 1.10],
                         [1.3, 1.60],
                         [1.3, 1.60],
                         [0.875, 1.05]])
    
        nbins = [10]*4
        myPlotDict['xlabel'] = r'velocity in $x$'
        myPlotDict['ylabel'] = r'number of samples'
        myPlotDict['ylim'] = [0, nsamples]
        
        for n in range(0,4):
            myPlotDict['title'] = titles[n]
            myPlotDict['xlim'] = xLims[n,:]
            myPlotDict['nbins'] = nbins[n]
            myPlotDict['out_filename'] = dir_fig+"cf_vx"+str(n)+"_lx"+str(lx)+".pdf"
            common.plotHist(data_vx[n*nsamples:(n+1)*nsamples], myPlotDict)
    
    elif plot_id == 2:
        
        xLims = np.array([[-0.004, 0.004],
                         [-0.006, 0.006],
                         [-0.006, 0.006],
                         [-0.005, 0.005]])
    
        nbins = [10]*4
        myPlotDict['xlabel'] = r'velocity in $y$'
        myPlotDict['ylabel'] = r'number of samples'
        myPlotDict['ylim'] = [0, nsamples]
        for n in range(0,4):
            myPlotDict['title'] = titles[n]
            myPlotDict['xlim'] = xLims[n,:]
            myPlotDict['nbins'] = nbins[n]
            myPlotDict['out_filename'] = dir_fig+"cf_vy"+str(n)+"_lx"+str(lx)+".pdf"
            common.plotHist(data_vy[n*nsamples:(n+1)*nsamples], myPlotDict)


if __name__ == "__main__":

    show_plot = True
    save_plot = True
    
    # Re = 1600
    Re = 3200
    
    nsamples = 480
    
    lx = 3
    plot_id = 2
    
    dir_sol = '../../output/uq_pincompNS_cf/Re'+str(Re)+'/samples'+str(nsamples)+'/'
    dir_fig = '../../figures/uq_incompNS/uq_cf/Re'+str(Re)+'/pdfs/samples'+str(nsamples)+'/'
    velocity_pdfs(dir_sol, dir_fig, lx, show_plot, save_plot, plot_id)
    
    
# End of file
