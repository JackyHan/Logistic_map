################################################################################################################
# Logistic_map
# Description: Model, simulate, and visualize logistic map and chaos
# Author: Jacky Han
#
# The MIT License (MIT)
#
# Copyright (c) 2017 Jacky Han
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#################################################################################################################

import pandas as pd, numpy as np, math
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
np.set_printoptions(threshold=np.nan)


# define the fonts to use for plot titles and labels
_font_family = ['Helvetica', 'Arial', 'sans-serif']
title_font = fm.FontProperties(family=_font_family, style='normal', size=20, weight='normal', stretch='normal')
label_font = fm.FontProperties(family=_font_family, style='normal', size=16, weight='normal', stretch='normal')


def save_fig(filename='image', folder='/Users/jackyhan/Desktop/', dpi=300, bbox_inches='tight', pad=0.1):
    """
    Save the current figure as a file to disk.
    
    Arguments
    ---------
    filename: string, filename of image file to be saved
    folder: string, folder in which to save the image file
    dpi: int, resolution at which to save the image
    bbox_inches: string, tell matplotlib to figure out the tight bbox of the figure
    pad: float, inches to pad around the figure
    
    Returns
    -------
    None
    """
    
    plt.savefig('{}/{}.png'.format(folder, filename), dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad)

    
    
    
def save_and_show(fig, ax, save, show, filename='image', folder='/Users/jackyhan/Desktop/', dpi=300, bbox_inches='tight', pad=0.1):
    """
    Consistently handle plot completion by saving then either displaying or returning the figure.
    
    Arguments
    ---------
    fig: matplotlib figure
    ax: matplotlib axis
    save: bool, whether to save the image to disk, or not
    show: bool, whether to display the image or instead just return the figure and axis
    filename: string, filename of image file to be saved
    folder: string, folder in which to save the image file
    dpi: int, resolution at which to save the image
    bbox_inches: string, tell matplotlib to figure out the tight bbox of the figure
    pad: float, inches to pad around the figure
    
    Returns
    -------
    fig, ax: tuple (if show=False, otherwise returns None)
    """
    
    if save:  
        save_fig(filename=filename, folder=folder, dpi=dpi, bbox_inches=bbox_inches, pad=pad)
        
    if show:
        plt.show()   
    else:
        return fig, ax
    
    
    
def logistic_map(pop, rate):
    """
    Define the equation for the logistic map.
    
    Arguments
    ---------
    pop: float, current population value at time t
    rate: float, growth rate parameter values
    
    Returns
    -------
    scalar result of logistic map at time t+1
    """
    
    return pop * rate * (1 - pop)


def logistic_map_coupled(p1, p2, r1, r2, e):
    """
    Define the equation for the coupled logistic map.
    
    Arguments
    ---------
    p1, p2: float, current population value at time t 
    r1, r2: float, growth rate parameter values
	e: float, coupling factor
    
    Returns
    -------
    scalar result of logistic map at time t+1
    """
    
    return (1-e) * logistic_map(p1,r1) + e * logistic_map(p2,r2)

    
def simulate_points_range(num_gens, rate_one, rate_two, num_discard, initial_pop_one, initial_pop_two, e_min, e_max, num_e):
    """
    Create a DataFrame with columns for each system, row labels for each time step, and values computed by the model (without JIT compilation).
    
    Arguments
    ---------
    num_gens: int, number of iterations to run the model
    rate_one: float, the first growth rate for the model, between 0 and 4
    rate_two: float, the second growth rate for the model, between 0 and 4
    num_discard: int, number of generations to discard before keeping population values
    initial_pop_one/two: float, starting population when you run the model, between 0 and 1
    e_min: min value of e
    e_max: max value of e
    num_e: number of e to be used between the min and max
    
    Returns
    -------
    df: pandas DataFrame
    """
    pops = []
    es = np.linspace(e_min, e_max, num_e)
    
    for e in es:
        # for each rate, run the function repeatedly, starting at the initial_pop
        pop = initial_pop_one
        pop2 = initial_pop_two
            
        # first run it num_discard times and ignore the results
        for _ in range(num_discard):
            pop_new = logistic_map_coupled(pop, pop2, rate_one, rate_two, e)
            pop2_new = logistic_map_coupled(pop2, pop, rate_two, rate_one, e)
            pop = pop_new
            pop2 = pop2_new 
            
        # now that those gens are discarded, run it num_gens times and keep the results
        for gen_num in range(num_gens):
            pops.append([e,pop, pop2])
            pop_new = logistic_map_coupled(pop, pop2, rate_one, rate_two, e)
            pop2_new = logistic_map_coupled(pop2, pop, rate_two, rate_one, e)
            pop = pop_new
            pop2 = pop2_new 
        
    # return a DataFrame with one column for each growth rate and one row for each timestep (aka generation)
    df = pd.DataFrame(data=pops, columns=['e', 'pop1', 'pop2'])
    df.index = pd.MultiIndex.from_arrays([num_e * list(range(num_gens)), df['e'].values])
    return df.drop(labels='e', axis=1).unstack()

def simulate_points_single(num_gens, rate_one, rate_two, num_discard, initial_pop_one, initial_pop_two, e):
    """
    Create a DataFrame with columns for each system, row labels for each time step, and values computed by the model (without JIT compilation).
    
    Arguments
    ---------
    num_gens: int, number of iterations to run the model
    rate_one: float, the first growth rate for the model, between 0 and 4
    rate_two: float, the second growth rate for the model, between 0 and 4
    num_discard: int, number of generations to discard before keeping population values
    initial_pop_one/two: float, starting population when you run the model, between 0 and 1
    e: coupling coefficient
    
    Returns
    -------
    df: pandas DataFrame
    """
    pops = []
    
    # for each rate, run the function repeatedly, starting at the initial_pop
    pop = initial_pop_one
    pop2 = initial_pop_two
            
    # first run it num_discard times and ignore the results
    for _ in range(num_discard):
        pop_new = logistic_map_coupled(pop, pop2, rate_one, rate_two, e)
        pop2_new = logistic_map_coupled(pop2, pop, rate_two, rate_one, e)
        pop = pop_new
        pop2 = pop2_new 
            
    # now that those gens are discarded, run it num_gens times and keep the results
    for gen_num in range(num_gens):
        pops.append([pop, pop2])
        pop_new = logistic_map_coupled(pop, pop2, rate_one, rate_two, e)
        pop2_new = logistic_map_coupled(pop2, pop, rate_two, rate_one, e)
        pop = pop_new
        pop2 = pop2_new 
        
    # return a DataFrame with one column for each growth rate and one row for each timestep (aka generation)
    df = pd.DataFrame(data=pops, columns=['pop1', 'pop2'])
    return df


def get_bifurcation_plot_points_range(pops, one_or_two):
    """
    Convert a DataFrame of values from the model into a set of xy points that you can plot as a bifurcation diagram.
    
    Arguments
    ---------
    pops: DataFrame, population data output from the model
    one_or_two: display pop1 or pop2, 1 for pop1, 2 for pop2
    
    Returns
    -------
    xy_points: DataFrame
    """

    # create a new DataFrame to contain our xy points
    xy_points = pd.DataFrame(columns=['x', 'y'])
    if one_or_two == 1:
        pops = pops['pop1']
    else:
        pops = pops['pop2']
    
   	# for each column in the populations DataFrame
    for e in pops.columns:
        # append the growth rate as the x column and all the population values as the y column
        xy_points = xy_points.append(pd.DataFrame({'x':e, 'y':pops[e]}))
    
        # reset the index and drop the old index before returning the xy point data
        xy_points = xy_points.reset_index().drop(labels='index', axis=1)
    return xy_points


def get_bifurcation_plot_points_single(pops, one_or_two):
    """
    Convert a DataFrame of values from the model into a set of xy points that you can plot as a bifurcation diagram.
    
    Arguments
    ---------
    pops: DataFrame, population data output from the model
    one_or_two: display pop1 or pop2, 1 for pop1, 2 for pop2
    
    Returns
    -------
    xy_points: DataFrame
    """

    # create a new DataFrame to contain our xy points
    xy_points = pd.DataFrame(columns=['y'])

    if one_or_two == 1:
        xy_points['y'] = pops['pop1']
    else:
        xy_points['y'] = pops['pop2']

    # reset the index
    xy_points = xy_points.reset_index()
    xy_points.columns = ['x', 'y']
    return xy_points

    
def bifurcation_plot_range(pops, xmin=0, xmax=4, ymin=0, ymax=1, figsize=(10,6),
                     title='', xlabel='Epsilon', ylabel='Population', 
                     color='#003399', filename='', save=True, show=True, title_font=title_font, label_font=label_font,
                     folder='/Users/jackyhan/Desktop/', dpi=300, bbox_inches='tight', pad=0.1, correlation = False):
    """
    Plot the results of the model as a bifurcation diagram.
    
    Arguments
    ---------
    pops: DataFrame population data output from the model
    xmin: float, minimum value on the x axis
    xmax: float, maximum value on the x axis
    ymin: float, minimum value on the y axis
    ymax: float, maximum value on the y axis
    figsize: tuple, (width, height) of figure
    title: string, title of the plot
    xlabel: string, label of the x axis
    ylabel: string, label of the y axis
    color: string, color of the points in the scatter plot
    filename: string, name of image file to be saved, if applicable
    save: bool, whether to save the image to disk or not
    show: bool, whether to display the image on screen or not
    title_font: matplotlib.font_manager.FontProperties, font properties for figure title
    label_font: matplotlib.font_manager.FontProperties,  font properties for axis labels
    folder: string, folder in which to save the image file
    dpi: int, resolution at which to save the image
    bbox_inches: string, tell matplotlib to figure out the tight bbox of the figure
    pad: float, inches to pad around the figure
    
    Returns
    -------
    fig, ax: tuple (if show=False, otherwise returns None)
    """
    
    # create a new matplotlib figure and axis and set its size
    fig, ax = plt.subplots(figsize=figsize)
    
    # plot the xy data
    point1 = get_bifurcation_plot_points_range(pops, 1)
    point2 = get_bifurcation_plot_points_range(pops, 2)
    bifurcation_scatter = ax.scatter(point1['x'], point1['y'], c='#00a9ff', edgecolor='None', alpha=1, s=1)
    bifurcation_scatter = ax.scatter(point2['x'], point2['y'], c='#ffa500', edgecolor='None', alpha=1, s=1)
    
    # plot the correlation
    if correlation:
        point3 = get_correlation_points(pops)
        point4 = np.array(point3)
        print(point4)
        ax.plot(point3['x'], point3['y'])
    
    # set x and y limits, title, and x and y labels
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(title, fontproperties=title_font)
    ax.set_xlabel(xlabel, fontproperties=label_font)
    ax.set_ylabel(ylabel, fontproperties=label_font)
    
    return save_and_show(fig=fig, ax=ax, save=save, show=show, filename=filename, folder=folder, dpi=dpi, bbox_inches=bbox_inches, pad=pad)

    
def bifurcation_plot_single(pops, xmin=0, xmax=4, ymin=0, ymax=1, figsize=(10,6),
                     title='', xlabel='Time step', ylabel='Population', 
                     color='#003399', filename='', save=True, show=True, title_font=title_font, label_font=label_font,
                     folder='/Users/jackyhan/Desktop/', dpi=300, bbox_inches='tight', pad=0.1):
    """
    Plot the results of the model as a bifurcation diagram.
    
    Arguments
    ---------
    pops: DataFrame population data output from the model
    xmin: float, minimum value on the x axis
    xmax: float, maximum value on the x axis
    ymin: float, minimum value on the y axis
    ymax: float, maximum value on the y axis
    figsize: tuple, (width, height) of figure
    title: string, title of the plot
    xlabel: string, label of the x axis
    ylabel: string, label of the y axis
    color: string, color of the points in the scatter plot
    filename: string, name of image file to be saved, if applicable
    save: bool, whether to save the image to disk or not
    show: bool, whether to display the image on screen or not
    title_font: matplotlib.font_manager.FontProperties, font properties for figure title
    label_font: matplotlib.font_manager.FontProperties,  font properties for axis labels
    folder: string, folder in which to save the image file
    dpi: int, resolution at which to save the image
    bbox_inches: string, tell matplotlib to figure out the tight bbox of the figure
    pad: float, inches to pad around the figure
    
    Returns
    -------
    fig, ax: tuple (if show=False, otherwise returns None)
    """
    
    # create a new matplotlib figure and axis and set its size
    fig, ax = plt.subplots(figsize=figsize)
    
    # plot the xy data
    point1 = get_bifurcation_plot_points_single(pops, 1)
    point2 = get_bifurcation_plot_points_single(pops, 2)
    bifurcation_scatter = ax.plot(point1['x'], point1['y'], c='#00a9ff')
    bifurcation_scatter = ax.plot(point2['x'], point2['y'], c='#ffa500')
    
    # set x and y limits, title, and x and y labels
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontproperties=title_font)
    ax.set_xlabel(xlabel, fontproperties=label_font)
    ax.set_ylabel(ylabel, fontproperties=label_font)
    
    return save_and_show(fig=fig, ax=ax, save=save, show=show, filename=filename, folder=folder, dpi=dpi, bbox_inches=bbox_inches, pad=pad)
    
def correlation(a, b):
    #a = np.array(pop['pop1'])
    #b = np.array(pop['pop2'])
    meanOne = np.mean(a)
    meanTwo = np.mean(b)
    stdOne = np.std(a)
    stdTwo = np.std(b)
    s = a.size
    sum = 0
    for i in range(0,s):
        sum += (a[i]-meanOne)/stdOne * (b[i]-meanTwo)/stdTwo
    if stdOne < 0.0001 or stdTwo < 0.0001:
        return 0
    return float(sum)/s
    
def get_correlation_points(pops):
    xy_points = pd.DataFrame(columns=['x', 'y'])
    pop1 = pops['pop1']
    pop2 = pops['pop2']
    for e in pop1.columns:
        pop_temp1 = np.array(pop1[e])
        pop_temp2 = np.array(pop2[e])
        cor = correlation(pop_temp1, pop_temp2)
        xy_points = xy_points.append(pd.DataFrame({'x':[e], 'y':[cor]}))
        xy_points = xy_points.reset_index().drop(labels='index', axis=1)
    return xy_points
        

    
def main(argv=None):  
  pops = simulate_points_range(num_gens = 300, rate_one = 3.838, rate_two = 3.976, num_discard = 300, initial_pop_one = 0.5, initial_pop_two = 0.7, e_min = 0, e_max = 1, num_e = 400)
  
  
  bifurcation_plot_range(pops, xmin=0, xmax=1, filename='logistic-map-bifurcation-1', correlation = True)
  
  #pop = simulate_points_single(num_gens = 100, rate_one = 3.8, rate_two = 3.5, num_discard = 300, initial_pop_one = 0.6, initial_pop_two = 0.7, e=0.05)
  #print(pop)
  #bifurcation_plot_single(pop, xmin=0, xmax=100, filename='logistic-map-bifurcation-2') 
  
  

if __name__ == '__main__':
  main()
    
