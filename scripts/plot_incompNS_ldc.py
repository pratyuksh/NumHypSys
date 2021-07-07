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
    return np.array(j["y"]), np.array(j["vx"]), np.array(j["x"]), np.array(j["vy"])


# Velocity profiles for lid-driven cavity
# Reynold's number 100
def velocity_profile_Re100(dir_fig, show_plot, save_plot, plot_id):

    fn = "../output/incompNS/ldc_Re100_velocity_profile.json"
    y, vx, x,  vy = load_data(fn)
        
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    myPlotDict['legend_loc'] = 'lower right'
    myPlotDict['data_markers'] = ['b.-']
    
    if plot_id == 1:
        myPlotDict['title'] = '$u_{x_1}$ on the line $x_{1} = 0.5$, Reynolds number $100$'
        myPlotDict['xlabel'] = r'$u_{x_{1}}$'
        myPlotDict['ylabel'] = r'$x_{2}$'
        myPlotDict['data_labels'] = ['$u_{x_{1}}$']
        myPlotDict['out_filename'] =dir_fig+"ldc_Re100_vx.pdf"
        myPlotDict['xlim'] = [-0.4, 1.1]
        myPlotDict['ylim'] = [-0.1, 1.1]
        common.plotData([vx], [y], myPlotDict)
    elif plot_id == 2:
        myPlotDict['title'] = '$u_{x_{2}}$ on the line $x_{2} = 0.5$, Reynolds number $100$'
        myPlotDict['xlabel'] = r'$x_{1}$'
        myPlotDict['ylabel'] = r'$u_{x_{2}}$'
        myPlotDict['data_labels'] = ['$u_{x_{2}}$']
        myPlotDict['out_filename'] =dir_fig+"ldc_Re100_vy.pdf"
        myPlotDict['xlim'] = [-0.1, 1.1]
        myPlotDict['ylim'] = [-0.3, 0.2]
        common.plotData([x], [vy], myPlotDict)

# Velocity profiles for lid-driven cavity
# Reynold's number 100
def velocity_profile_Re3200(dir_fig, show_plot, save_plot, plot_id):

    fn = "../output/incompNS/ldc_Re3200_velocity_profile.json"
    y, vx, x,  vy = load_data(fn)
        
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    myPlotDict['legend_loc'] = 'lower right'
    myPlotDict['data_markers'] = ['b.-']
    
    if plot_id == 1:
        myPlotDict['title'] = '$u_{x_{1}}$ on the line $x_{1} = 0.5$, Reynolds number $3200$'
        myPlotDict['xlabel'] = r'$u_{x_{1}}$'
        myPlotDict['ylabel'] = r'$x_{2}$'
        myPlotDict['data_labels'] = ['$u_{x_{1}}$']
        myPlotDict['out_filename'] =dir_fig+"ldc_Re3200_vx.pdf"
        myPlotDict['xlim'] = [-0.6, 1.1]
        myPlotDict['ylim'] = [-0.1, 1.1]
        common.plotData([vx], [y], myPlotDict)
    elif plot_id == 2:
        myPlotDict['title'] = '$u_{x_{2}}$ on the line $x_{2} = 0.5$, Reynolds number $3200$'
        myPlotDict['xlabel'] = r'$x_{1}$'
        myPlotDict['ylabel'] = r'$u_{x_{2}}$'
        myPlotDict['data_labels'] = ['$u_{x_{2}}$']
        myPlotDict['out_filename'] =dir_fig+"ldc_Re3200_vy.pdf"
        myPlotDict['xlim'] = [-0.1, 1.1]
        myPlotDict['ylim'] = [-0.6, 0.5]
        common.plotData([x], [vy], myPlotDict)

if __name__ == "__main__":

    dir_fig = '../figures/incompNS/'
    show_plot = True
    save_plot = True
    
    velocity_profile_Re100(dir_fig, show_plot, save_plot, 2)
    velocity_profile_Re3200(dir_fig, show_plot, save_plot, 2)
    
    
# End of file
