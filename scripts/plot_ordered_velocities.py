#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import LogLocator
from matplotlib import rc
from matplotlib import rcParams
font = {'family' : 'Dejavu Sans',
        'weight' : 'normal',
        'size'   : 30}
rc('font', **font)
rcParams['lines.linewidth'] = 4
rcParams['lines.markersize'] = 18
rcParams['markers.fillstyle'] = 'none'


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def load_data(filename):
    j = load_json(filename)
    return np.array(j["vx"]), np.array(j["vy"])


def plot_ordered_velocity(dir_fig, show_plot, save_plot, plot_id):

    # fn = "../output_test/uq_pincompNS_ldc/pp_lx5/velocity_s0.json"
    fn = "../output/uq_pincompNS_svs/velocity_mean_lx3.json"
    vx_, vy_ = load_data(fn)
    
    N = 32
    vx = np.reshape(vx_, [32, 32])
    vy = np.reshape(vy_, [32, 32])
    
    plt.contour(vx, 200)
    plt.show()

if __name__ == "__main__":

    dir_fig = '../figures/incompNS/'
    show_plot = True
    save_plot = False
    
    plot_ordered_velocity(dir_fig, show_plot, save_plot, 1)


    
# End of file
