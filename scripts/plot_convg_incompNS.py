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
    return j["h_max"], np.array(j["velocity"]), np.array(j["pressure"])


# convergence results for test 1 on meshes with quadrilateral elements
# central numerical flux
def convg_quadMeshes_central_test1(dir_fig, show_plot, save_plot, plot_id):

    deg = []
    h_max = []   
    
    errL2_v = []
    ref_errL2_v = []
    slopes_v = []
    
    errL2_p = []
    ref_errL2_p = []
    slopes_p = []
    
    for k in range(0,3):
        fn = "../output/pincompNS/convergence_quadMesh_central_tgv_test1_deg"+str(k)+".json"
        h_max_, errL2_v_, errL2_p_ = load_data(fn)
        
        # linear fitting
        linfit_v = np.polyfit(np.log(h_max_), np.log(errL2_v_), 1)
        linfit_p = np.polyfit(np.log(h_max_), np.log(errL2_p_), 1)
        ref_errL2_v_ = np.exp(np.polyval(linfit_v, np.log(h_max_))-1)
        ref_errL2_p_ = np.exp(np.polyval(linfit_p, np.log(h_max_))-1)
        slopes_v.append(linfit_v[0])
        slopes_p.append(linfit_p[0])

        deg.append(k)
        h_max.append(h_max_)
        errL2_v.append(errL2_v_)
        errL2_p.append(errL2_p_)
        ref_errL2_v.append(ref_errL2_v_)
        ref_errL2_p.append(ref_errL2_p_)
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    # plot error vs mesh size
    myPlotDict['xlabel'] = r'Mesh size $h$ [log]'
    myPlotDict['legend_loc'] = 'lower right'
    myPlotDict['data_markers'] = ['ko-', 'bs-', 'rd-']
    myPlotDict['data_labels'] = ['$k=0$', '$k=1$', '$k=2$']
    myPlotDict['ylim'] = [1e-6, 1e+0]
    myPlotDict['ref_data_markers'] = ['k-.', 'b-.', 'r-.']
    
    if plot_id == 1:
        myPlotDict['title'] = 'Velocity error vs Mesh size, quadrilateral mesh, central'
        myPlotDict['ylabel'] = r'Rel. error $\left|| v - v_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_quadMesh_central_tgv_test1_velocity.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_v[0], '$O(h^{%4.2f})$'%slopes_v[1], '$O(h^{%4.2f})$'%slopes_v[2]]
        common.plotLogLogData(h_max, errL2_v, ref_errL2_v, myPlotDict)
        
    if plot_id == 2:
        myPlotDict['title'] = 'Pressure error vs Mesh size, quadrilateral mesh, central'
        myPlotDict['ylabel'] = r'Rel. error $\left|| p - p_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_quadMesh_central_tgv_test1_pressure.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_p[0], '$O(h^{%4.2f})$'%slopes_p[1], '$O(h^{%4.2f})$'%slopes_p[2]]
        common.plotLogLogData(h_max, errL2_p, ref_errL2_p, myPlotDict)

# convergence results for test 1 on meshes with quadrilateral elements
# upwind numerical flux
def convg_quadMeshes_upwind_test1(dir_fig, show_plot, save_plot, plot_id):

    deg = []
    h_max = []   
    
    errL2_v = []
    ref_errL2_v = []
    slopes_v = []
    
    errL2_p = []
    ref_errL2_p = []
    slopes_p = []
    
    for k in range(0,3):
        fn = "../output/pincompNS/convergence_quadMesh_upwind_tgv_test1_deg"+str(k)+".json"
        h_max_, errL2_v_, errL2_p_ = load_data(fn)
        
        # linear fitting
        linfit_v = np.polyfit(np.log(h_max_), np.log(errL2_v_), 1)
        linfit_p = np.polyfit(np.log(h_max_), np.log(errL2_p_), 1)
        ref_errL2_v_ = np.exp(np.polyval(linfit_v, np.log(h_max_))-1)
        ref_errL2_p_ = np.exp(np.polyval(linfit_p, np.log(h_max_))-1)
        slopes_v.append(linfit_v[0])
        slopes_p.append(linfit_p[0])

        deg.append(k)
        h_max.append(h_max_)
        errL2_v.append(errL2_v_)
        errL2_p.append(errL2_p_)
        ref_errL2_v.append(ref_errL2_v_)
        ref_errL2_p.append(ref_errL2_p_)
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    # plot error vs mesh size
    myPlotDict['xlabel'] = r'Mesh size $h$ [log]'
    myPlotDict['legend_loc'] = 'lower right'
    myPlotDict['data_markers'] = ['ko-', 'bs-', 'rd-']
    myPlotDict['data_labels'] = ['$k=0$', '$k=1$', '$k=2$']
    myPlotDict['ylim'] = [1e-6, 1e+0]
    myPlotDict['ref_data_markers'] = ['k-.', 'b-.', 'r-.']
    
    if plot_id == 1:
        myPlotDict['title'] = 'Velocity error vs Mesh size, quadrilateral mesh, upwind'
        myPlotDict['ylabel'] = r'Rel. error $\left|| v - v_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_quadMesh_upwind_tgv_test1_velocity.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_v[0], '$O(h^{%4.2f})$'%slopes_v[1], '$O(h^{%4.2f})$'%slopes_v[2]]
        common.plotLogLogData(h_max, errL2_v, ref_errL2_v, myPlotDict)
        
    if plot_id == 2:
        myPlotDict['title'] = 'Pressure error vs Mesh size, quadrilateral mesh, upwind'
        myPlotDict['ylabel'] = r'Rel. error $\left|| p - p_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_quadMesh_upwind_tgv_test1_pressure.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_p[0], '$O(h^{%4.2f})$'%slopes_p[1], '$O(h^{%4.2f})$'%slopes_p[2]]
        common.plotLogLogData(h_max, errL2_p, ref_errL2_p, myPlotDict)

# convergence results for test 1 on meshes with triangular elements
# central numerical flux
def convg_triMeshes_central_test1(dir_fig, show_plot, save_plot, plot_id):

    deg = []
    h_max = []   
    
    errL2_v = []
    ref_errL2_v = []
    slopes_v = []
    
    errL2_p = []
    ref_errL2_p = []
    slopes_p = []
    
    for k in range(0,3):
        fn = "../output/pincompNS/convergence_triMesh_central_tgv_test1_deg"+str(k)+".json"
        h_max_, errL2_v_, errL2_p_ = load_data(fn)

        # linear fitting
        linfit_v = np.polyfit(np.log(h_max_), np.log(errL2_v_), 1)
        linfit_p = np.polyfit(np.log(h_max_), np.log(errL2_p_), 1)
        ref_errL2_v_ = np.exp(np.polyval(linfit_v, np.log(h_max_))-1)
        ref_errL2_p_ = np.exp(np.polyval(linfit_p, np.log(h_max_))-1)
        slopes_v.append(linfit_v[0])
        slopes_p.append(linfit_p[0])

        deg.append(k)
        h_max.append(h_max_)
        errL2_v.append(errL2_v_)
        errL2_p.append(errL2_p_)
        ref_errL2_v.append(ref_errL2_v_)
        ref_errL2_p.append(ref_errL2_p_)
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    # plot error vs mesh size
    myPlotDict['xlabel'] = r'Mesh size $h$ [log]'
    myPlotDict['legend_loc'] = 'lower right'
    myPlotDict['data_markers'] = ['ko-', 'bs-', 'rd-']
    myPlotDict['data_labels'] = ['$k=0$', '$k=1$', '$k=2$']
    myPlotDict['ref_data_markers'] = ['k-.', 'b-.', 'r-.']
    myPlotDict['ylim'] = [1e-6, 1e+0]
    
    if plot_id == 1:
        myPlotDict['title'] = 'Velocity error vs Mesh size, triangular mesh, central'
        myPlotDict['ylabel'] = r'Rel. error $\left|| v - v_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_triMesh_central_tgv_test1_velocity.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_v[0], '$O(h^{%4.2f})$'%slopes_v[1], '$O(h^{%4.2f})$'%slopes_v[2]]
        common.plotLogLogData(h_max, errL2_v, ref_errL2_v, myPlotDict)
        
    if plot_id == 2:    
        myPlotDict['title'] = 'Pressure error vs Mesh size, triangular mesh, central'
        myPlotDict['ylabel'] = r'Rel. error $\left|| p - p_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_triMesh_central_tgv_test1_pressure.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_p[0], '$O(h^{%4.2f})$'%slopes_p[1], '$O(h^{%4.2f})$'%slopes_p[2]]
        common.plotLogLogData(h_max, errL2_p, ref_errL2_p, myPlotDict)

# convergence results for test 1 on meshes with triangular elements
# upwind numerical flux
def convg_triMeshes_upwind_test1(dir_fig, show_plot, save_plot, plot_id):

    deg = []
    h_max = []   
    
    errL2_v = []
    ref_errL2_v = []
    slopes_v = []
    
    errL2_p = []
    ref_errL2_p = []
    slopes_p = []
    
    for k in range(0,3):
        fn = "../output/pincompNS/convergence_triMesh_upwind_tgv_test1_deg"+str(k)+".json"
        h_max_, errL2_v_, errL2_p_ = load_data(fn)

        # linear fitting
        linfit_v = np.polyfit(np.log(h_max_), np.log(errL2_v_), 1)
        linfit_p = np.polyfit(np.log(h_max_), np.log(errL2_p_), 1)
        ref_errL2_v_ = np.exp(np.polyval(linfit_v, np.log(h_max_))-1)
        ref_errL2_p_ = np.exp(np.polyval(linfit_p, np.log(h_max_))-1)
        slopes_v.append(linfit_v[0])
        slopes_p.append(linfit_p[0])

        deg.append(k)
        h_max.append(h_max_)
        errL2_v.append(errL2_v_)
        errL2_p.append(errL2_p_)
        ref_errL2_v.append(ref_errL2_v_)
        ref_errL2_p.append(ref_errL2_p_)
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    # plot error vs mesh size
    myPlotDict['xlabel'] = r'Mesh size $h$ [log]'
    myPlotDict['legend_loc'] = 'lower right'
    myPlotDict['data_markers'] = ['ko-', 'bs-', 'rd-']
    myPlotDict['data_labels'] = ['$k=0$', '$k=1$', '$k=2$']
    myPlotDict['ref_data_markers'] = ['k-.', 'b-.', 'r-.']
    myPlotDict['ylim'] = [1e-6, 1e+0]
    
    if plot_id == 1:
        myPlotDict['title'] = 'Velocity error vs Mesh size, triangular mesh, upwind'
        myPlotDict['ylabel'] = r'Rel. error $\left|| v - v_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_triMesh_upwind_tgv_test1_velocity.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_v[0], '$O(h^{%4.2f})$'%slopes_v[1], '$O(h^{%4.2f})$'%slopes_v[2]]
        common.plotLogLogData(h_max, errL2_v, ref_errL2_v, myPlotDict)
        
    if plot_id == 2:
        myPlotDict['title'] = 'Pressure error vs Mesh size, triangular mesh, upwind'
        myPlotDict['ylabel'] = r'Rel. error $\left|| p - p_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_triMesh_upwind_tgv_test1_pressure.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_p[0], '$O(h^{%4.2f})$'%slopes_p[1], '$O(h^{%4.2f})$'%slopes_p[2]]
        common.plotLogLogData(h_max, errL2_p, ref_errL2_p, myPlotDict)


# convergence results for test 2 on meshes with quadrilateral elements
# upwind numerical flux
def convg_quadMeshes_upwind_test2(dir_fig, show_plot, save_plot, plot_id):

    deg = []
    h_max = []   
    
    errL2_v = []
    ref_errL2_v = []
    slopes_v = []
    
    errL2_p = []
    ref_errL2_p = []
    slopes_p = []
    
    for k in range(0,3):
        fn = "../output/pincompNS/convergence_quadMesh_upwind_tgv_test2_deg"+str(k)+".json"
        h_max_, errL2_v_, errL2_p_ = load_data(fn)
        
        # linear fitting
        linfit_v = np.polyfit(np.log(h_max_), np.log(errL2_v_), 1)
        linfit_p = np.polyfit(np.log(h_max_), np.log(errL2_p_), 1)
        ref_errL2_v_ = np.exp(np.polyval(linfit_v, np.log(h_max_))-1)
        ref_errL2_p_ = np.exp(np.polyval(linfit_p, np.log(h_max_))-1)
        slopes_v.append(linfit_v[0])
        slopes_p.append(linfit_p[0])

        deg.append(k)
        h_max.append(h_max_)
        errL2_v.append(errL2_v_)
        errL2_p.append(errL2_p_)
        ref_errL2_v.append(ref_errL2_v_)
        ref_errL2_p.append(ref_errL2_p_)
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    # plot error vs mesh size
    myPlotDict['xlabel'] = r'Mesh size $h$ [log]'
    myPlotDict['legend_loc'] = 'lower right'
    myPlotDict['data_markers'] = ['ko-', 'bs-', 'rd-']
    myPlotDict['data_labels'] = ['$k=0$', '$k=1$', '$k=2$']
    myPlotDict['ylim'] = [1e-6, 1e+0]
    myPlotDict['ref_data_markers'] = ['k-.', 'b-.', 'r-.']
    
    if plot_id == 1:
        myPlotDict['title'] = 'Velocity error vs Mesh size, quadrilateral mesh, upwind'
        myPlotDict['ylabel'] = r'Rel. error $\left|| v - v_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_quadMesh_upwind_tgv_test2_velocity.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_v[0], '$O(h^{%4.2f})$'%slopes_v[1], '$O(h^{%4.2f})$'%slopes_v[2]]
        common.plotLogLogData(h_max, errL2_v, ref_errL2_v, myPlotDict)
        
    if plot_id == 2:
        myPlotDict['title'] = 'Pressure error vs Mesh size, quadrilateral mesh, upwind'
        myPlotDict['ylabel'] = r'Rel. error $\left|| p - p_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_quadMesh_upwind_tgv_test2_pressure.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_p[0], '$O(h^{%4.2f})$'%slopes_p[1], '$O(h^{%4.2f})$'%slopes_p[2]]
        common.plotLogLogData(h_max, errL2_p, ref_errL2_p, myPlotDict)

# convergence results for test 2 on meshes with triangular elements
# upwind numerical flux
def convg_triMeshes_upwind_test2(dir_fig, show_plot, save_plot, plot_id):

    deg = []
    h_max = []   
    
    errL2_v = []
    ref_errL2_v = []
    slopes_v = []
    
    errL2_p = []
    ref_errL2_p = []
    slopes_p = []
    
    for k in range(0,3):
        fn = "../output/pincompNS/convergence_triMesh_upwind_tgv_test2_deg"+str(k)+".json"
        h_max_, errL2_v_, errL2_p_ = load_data(fn)

        # linear fitting
        linfit_v = np.polyfit(np.log(h_max_), np.log(errL2_v_), 1)
        linfit_p = np.polyfit(np.log(h_max_), np.log(errL2_p_), 1)
        ref_errL2_v_ = np.exp(np.polyval(linfit_v, np.log(h_max_))-1)
        ref_errL2_p_ = np.exp(np.polyval(linfit_p, np.log(h_max_))-1)
        slopes_v.append(linfit_v[0])
        slopes_p.append(linfit_p[0])

        deg.append(k)
        h_max.append(h_max_)
        errL2_v.append(errL2_v_)
        errL2_p.append(errL2_p_)
        ref_errL2_v.append(ref_errL2_v_)
        ref_errL2_p.append(ref_errL2_p_)
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    # plot error vs mesh size
    myPlotDict['xlabel'] = r'Mesh size $h$ [log]'
    myPlotDict['legend_loc'] = 'lower right'
    myPlotDict['data_markers'] = ['ko-', 'bs-', 'rd-']
    myPlotDict['data_labels'] = ['$k=0$', '$k=1$', '$k=2$']
    myPlotDict['ref_data_markers'] = ['k-.', 'b-.', 'r-.']
    myPlotDict['ylim'] = [1e-6, 1e+0]
    
    if plot_id == 1:
        myPlotDict['title'] = 'Velocity error vs Mesh size, triangular mesh, upwind'  
        myPlotDict['ylabel'] = r'Rel. error $\left|| v - v_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_triMesh_upwind_tgv_test2_velocity.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_v[0], '$O(h^{%4.2f})$'%slopes_v[1], '$O(h^{%4.2f})$'%slopes_v[2]]
        common.plotLogLogData(h_max, errL2_v, ref_errL2_v, myPlotDict)
        
    if plot_id == 2:
        myPlotDict['title'] = 'Pressure error vs Mesh size, triangular mesh, upwind'
        myPlotDict['ylabel'] = r'Rel. error $\left|| p - p_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_triMesh_upwind_tgv_test2_pressure.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_p[0], '$O(h^{%4.2f})$'%slopes_p[1], '$O(h^{%4.2f})$'%slopes_p[2]]
        common.plotLogLogData(h_max, errL2_p, ref_errL2_p, myPlotDict)


# convergence results for test 4 on meshes with quadrilateral elements
# upwind numerical flux
def convg_quadMeshes_upwind_test4(dir_fig, show_plot, save_plot, plot_id):

    deg = []
    h_max = []   
    
    errL2_v = []
    ref_errL2_v = []
    slopes_v = []
    
    errL2_p = []
    ref_errL2_p = []
    slopes_p = []
    
    for k in range(0,3):
        fn = "../output/pincompNS/convergence_quadMesh_upwind_tgv_test4_deg"+str(k)+".json"
        h_max_, errL2_v_, errL2_p_ = load_data(fn)
        
        # linear fitting
        linfit_v = np.polyfit(np.log(h_max_), np.log(errL2_v_), 1)
        linfit_p = np.polyfit(np.log(h_max_), np.log(errL2_p_), 1)
        ref_errL2_v_ = np.exp(np.polyval(linfit_v, np.log(h_max_))-1)
        ref_errL2_p_ = np.exp(np.polyval(linfit_p, np.log(h_max_))-1)
        slopes_v.append(linfit_v[0])
        slopes_p.append(linfit_p[0])

        deg.append(k)
        h_max.append(h_max_)
        errL2_v.append(errL2_v_)
        errL2_p.append(errL2_p_)
        ref_errL2_v.append(ref_errL2_v_)
        ref_errL2_p.append(ref_errL2_p_)
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    # plot error vs mesh size
    myPlotDict['xlabel'] = r'Mesh size $h$ [log]'
    myPlotDict['legend_loc'] = 'lower right'
    myPlotDict['data_markers'] = ['ko-', 'bs-', 'rd-']
    myPlotDict['data_labels'] = ['$k=0$', '$k=1$', '$k=2$']
    myPlotDict['ylim'] = [1e-6, 1e+0]
    myPlotDict['ref_data_markers'] = ['k-.', 'b-.', 'r-.']
    
    if plot_id == 1:
        myPlotDict['title'] = 'Velocity error vs Mesh size, quadrilateral mesh, upwind'
        myPlotDict['ylabel'] = r'Rel. error $\left|| v - v_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_quadMesh_upwind_tgv_test4_velocity.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_v[0], '$O(h^{%4.2f})$'%slopes_v[1], '$O(h^{%4.2f})$'%slopes_v[2]]
        common.plotLogLogData(h_max, errL2_v, ref_errL2_v, myPlotDict)
        
    if plot_id == 2:
        myPlotDict['title'] = 'Pressure error vs Mesh size, quadrilateral mesh, upwind'
        myPlotDict['ylabel'] = r'Rel. error $\left|| p - p_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_quadMesh_upwind_tgv_test4_pressure.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_p[0], '$O(h^{%4.2f})$'%slopes_p[1], '$O(h^{%4.2f})$'%slopes_p[2]]
        common.plotLogLogData(h_max, errL2_p, ref_errL2_p, myPlotDict)

# convergence results for test 4 on meshes with triangular elements
# upwind numerical flux
def convg_triMeshes_upwind_test4(dir_fig, show_plot, save_plot, plot_id):

    deg = []
    h_max = []   
    
    errL2_v = []
    ref_errL2_v = []
    slopes_v = []
    
    errL2_p = []
    ref_errL2_p = []
    slopes_p = []
    
    for k in range(0,3):
        fn = "../output/pincompNS/convergence_triMesh_upwind_tgv_test4_deg"+str(k)+".json"
        h_max_, errL2_v_, errL2_p_ = load_data(fn)

        # linear fitting
        linfit_v = np.polyfit(np.log(h_max_), np.log(errL2_v_), 1)
        linfit_p = np.polyfit(np.log(h_max_), np.log(errL2_p_), 1)
        ref_errL2_v_ = np.exp(np.polyval(linfit_v, np.log(h_max_))-1)
        ref_errL2_p_ = np.exp(np.polyval(linfit_p, np.log(h_max_))-1)
        slopes_v.append(linfit_v[0])
        slopes_p.append(linfit_p[0])

        deg.append(k)
        h_max.append(h_max_)
        errL2_v.append(errL2_v_)
        errL2_p.append(errL2_p_)
        ref_errL2_v.append(ref_errL2_v_)
        ref_errL2_p.append(ref_errL2_p_)
    
    myPlotDict = {}
    myPlotDict['show_plot'] = show_plot
    myPlotDict['save_plot'] = save_plot
    
    # plot error vs mesh size
    myPlotDict['xlabel'] = r'Mesh size $h$ [log]'
    myPlotDict['legend_loc'] = 'lower right'
    myPlotDict['data_markers'] = ['ko-', 'bs-', 'rd-']
    myPlotDict['data_labels'] = ['$k=0$', '$k=1$', '$k=2$']
    myPlotDict['ref_data_markers'] = ['k-.', 'b-.', 'r-.']
    myPlotDict['ylim'] = [1e-6, 1e+0]
    
    if plot_id == 1:
        myPlotDict['title'] = 'Velocity error vs Mesh size, triangular mesh, upwind'  
        myPlotDict['ylabel'] = r'Rel. error $\left|| v - v_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_triMesh_upwind_tgv_test4_velocity.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_v[0], '$O(h^{%4.2f})$'%slopes_v[1], '$O(h^{%4.2f})$'%slopes_v[2]]
        common.plotLogLogData(h_max, errL2_v, ref_errL2_v, myPlotDict)
        
    if plot_id == 2:
        myPlotDict['title'] = 'Pressure error vs Mesh size, triangular mesh, upwind'
        myPlotDict['ylabel'] = r'Rel. error $\left|| p - p_h \right||_{L^2(\Omega)}$ [log]'
        myPlotDict['out_filename'] =dir_fig+"convg_triMesh_upwind_tgv_test4_pressure.pdf"
        myPlotDict['xlim'] = [6e-2,2e+0]
        myPlotDict['ref_data_labels'] = ['$O(h^{%4.2f})$'%slopes_p[0], '$O(h^{%4.2f})$'%slopes_p[1], '$O(h^{%4.2f})$'%slopes_p[2]]
        common.plotLogLogData(h_max, errL2_p, ref_errL2_p, myPlotDict)


if __name__ == "__main__":

    dir_fig = '../figures/incompNS/'
    show_plot = True
    save_plot = True
    
    plot_id = 2
    
    convg_quadMeshes_central_test1(dir_fig, show_plot, save_plot, plot_id)
    convg_quadMeshes_upwind_test1(dir_fig, show_plot, save_plot, plot_id)
    
    convg_triMeshes_central_test1(dir_fig, show_plot, save_plot, plot_id)
    convg_triMeshes_upwind_test1(dir_fig, show_plot, save_plot, plot_id)
    
    convg_quadMeshes_upwind_test2(dir_fig, show_plot, save_plot, plot_id)
    convg_triMeshes_upwind_test2(dir_fig, show_plot, save_plot, plot_id)
    
    convg_quadMeshes_upwind_test4(dir_fig, show_plot, save_plot, plot_id)
    convg_triMeshes_upwind_test4(dir_fig, show_plot, save_plot, plot_id)
    
    
# End of file
