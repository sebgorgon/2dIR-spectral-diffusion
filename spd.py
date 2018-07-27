### SET UP OF ENVIRONMENT ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize as opt
import os



### SET UP OF GENERAL DATA PRE-PROCESSING ###

def df_separate(df):
    
    """Separates dataframe into 1D lists of x and y values, and a 2D matrix of z values"""

    x1d = np.array(df[df.columns[0]])[1:]  
    df = df.drop(df.columns[0], axis=1)
    y1d = np.array(df)[0,:]  
    z2d = np.array(df)[1:,:]

    return x1d, y1d, z2d

def df_to_xyz(df):
    
    """Converts dataframe to 1D arrays of XYZ with one-to-one correspondence - for fitting data"""
    
    df = np.array(df)
    X, Y, Z = [], [], []
    for i, y in enumerate(df[0][1:], 1):
        for z in df[1:]:
            X.append(z[0])
            Y.append(y)
            Z.append(z[i])
    return np.vstack((X,Y,Z))

def df_join(x, y, z):    
    
    """Creates dataframe from 1D lists of x and y values, and a 2D matrix of z values"""
    
    combmap = np.insert(z, 0, x, axis=1) 
    y = np.insert(y, 0, 0)
    combmap = np.insert(combmap, 0, y, axis=0)
    return pd.DataFrame(combmap)
    
def val_to_index(val, array):
    
    """Returns index of number inside the array that is closest to set value"""
    
    return np.abs(array-val).argmin()   

def crop_df(df, ranges):

    """Crops the dataset to selected region"""
    
    x1d, y1d, z2d = df_separate(df)
    xlimits = (val_to_index(ranges['xlo'], x1d), val_to_index(ranges['xhi'], x1d))
    ylimits = (val_to_index(ranges['ylo'], y1d), val_to_index(ranges['yhi'], y1d))

    x1d = x1d[xlimits[0]:xlimits[1]+1]    
    y1d = y1d[ylimits[0]:ylimits[1]+1]
    z2d = z2d[xlimits[0]:xlimits[1]+1,ylimits[0]:ylimits[1]+1]
    return df_join(x1d, y1d, z2d)

def contour_plot(df, zlim=200):
    
    """Generates a 2D plot of a dataset"""
    
    x1d, y1d, z2d = df_separate(df)
    ztop = np.max([np.max(z2d), np.abs(np.min(z2d))])
    levels = np.linspace(-ztop, ztop, zlim+1)
    
    plt.contourf(x1d, y1d, z2d.transpose(), levels, cmap='RdBu_r')
    plt.colorbar()
    plt.xlabel('Probe / cm-1')
    plt.ylabel('Pump / cm-1')
    return 

def extrema_find(df):
    
    """Returns coordinates of extrema in dataset"""
    
    x1d, y1d, z2d = df_separate(df)
    (ximx, yimx) = np.unravel_index(z2d.argmax(), z2d.shape)
    xpos, ypos, zpos = x1d[ximx], y1d[yimx], np.max(z2d)
    (ximn, yimn) = np.unravel_index(z2d.argmin(), z2d.shape)
    xneg, yneg, zneg = x1d[ximn], y1d[yimn], np.min(z2d)
    
    p0idea = (zpos, -zneg, xpos, xneg, ypos, '?', '?', '?', '?')
    return p0idea

def timestring(path):
    
    """Returns population time from filename"""
    
    rpatr = path[::-1]
    ind = rpatr.find('sp')
    cut = rpatr[ind+2:]
    end = cut.find('_')
    return float(cut[:end][::-1])

def polarisation_select(dataset, polarisation):
   
    """Selects LifeTIME dataset based on polarisation chosen and reverses if needed"""

    x, y, z = df_separate(dataset)
    mid = int(0.5*len(x))
    if y[1]<y[0]:
        y = np.flip(y, axis=0)
        z = np.flip(z, axis=1)
    if polarisation is 'perpendicular':
        dataset = df_join(x[:mid], y, z[:mid,:])
    elif polarisation is 'parallel':
        dataset = df_join(x[mid:], y, z[mid:,:])
    return dataset

def display_dataset(direct, polarisation, ranges=None):
    
    """Displays first frame of dataset so that x and y limits can be set"""
    
    paths = []
    for filename in os.listdir(direct):
        if filename.endswith("2DIR.csv"):
             paths.append(str(os.path.join(direct, filename)))
    frame = paths[0] 
    dataset = pd.read_csv(frame, header=None, delimiter='\t') 
    dataset = polarisation_select(dataset, polarisation)

    if ranges:
        dataset = crop_df(dataset, ranges)
        print('Suggested p0:')
        print(extrema_find(dataset))
        dataset = crop_df(dataset, ranges)
        
    fig = plt.figure()
    contour_plot(dataset)
    fig.patch.set_facecolor('w')
    return



### DEFINES 2 TYPES OF 2D GAUSSIAN FUNCTIONS ###

def uni_gauss(xy, *param):
    
    if len(param) == 1:
        param = param[0]
    
    amplitude, xo, yo, sigma_x, sigma_y, theta, offset = \
    param[0], param[1], param[2], param[3], param[4], param[5], param[6]
    
    mid = int(0.5 * len(xy))
    x, y = xy[:mid], xy[mid:]
    xo, yo = float(xo), float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)

    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def bi_gauss(xy, *param):
    
    if len(param) == 1:
        param = param[0]
    
    amplitude1, amplitude2, xo1, xo2, yo, sigma_x, sigma_y, theta, offset = \
    param[0], param[1], param[2], param[3], param[4], param[5], param[6], param[7], param[8]    
    
    mid = int(0.5 * len(xy))
    x, y = xy[:mid], xy[mid:]
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    g = offset + amplitude1*np.exp( - (a*((x-xo1)**2) + 2*b*(x-xo1)*(y-yo) + c*((y-yo)**2))) \
    + amplitude2*np.exp( - (a*((x-xo2)**2) + 2*b*(x-xo2)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()



### SET UP OF FITING ROUTINE ###

def fit_prepare(peakselect, data, p0):

    """Prepares data for fitting given the chosen parameters"""
    
    if peakselect == 1:
        zfin = [0 if j < 0 else j for j in data]
        zfin = np.array(zfin)
        p0 = np.delete(p0, [1,3])
        func = uni_gauss
        
    elif peakselect == -1:
        zfin = [0 if j > 0 else j for j in data]
        zfin = - np.array(zfin)         
        p0 = np.delete(p0, [1,2])
        func = uni_gauss

    elif peakselect == 0:
        zfin = np.abs(data)
        func = bi_gauss
        
    return zfin, p0, func

def fit_calc(popt, pcov, time, func):
    
    """Calculates frame fit results from optimized parameters"""
    
    std = np.sqrt(np.diag(pcov))
    
    if func is uni_gauss:
        popt = np.insert(popt, [1,2], [popt[0], popt[1]])
        std = np.insert(std, [1,2], [std[0], std[1]])

    sx, sy, ex, ey = abs(popt[5]), abs(popt[6]) ,std[5],std[6]
    if sx < sy:
        sx, sy, ex, ey = sy, sx, ey, ex
    
    elipt = abs((sy)**2 - (sx)**2)/((sx)**2 + (sy)**2)
    elipterr = ((((2*sx-sy**2)/(2*sx+sy**2))*ex)**2+(((sx**2-2*sy)/(sx**2+2*sy))*ey)**2)**0.5

    theta = popt[7] - (popt[7] // np.pi) * np.pi
    if theta > 0.5 * np.pi:
        theta = np.pi - theta
    if theta < 0.25 * np.pi:
        theta = 0.5 * np.pi - theta
    thetaerr = std[7]                     

    igrad = 1 / np.tan(theta)
    igraderr = ((np.cos(theta)) ** 0.5) * thetaerr                     

    arrres = (time, popt[0], popt[1], popt[2], popt[3], popt[4], sx, ex, sy, ey, elipt,\
              elipterr, theta, thetaerr, igrad, igraderr)
    cols = ['time','Ampl_1','Ampl_2','x0_1','x0_2','y_0', 'sigma_x', 'sigma_x_err','sigma_y',\
            'sigma_y_err', 'elipt','elipt_err', 'theta', 'theta_err', 'igrad', 'igrad_err']

    dfres = pd.DataFrame(columns=cols)
    dfres.loc[0] = np.array(arrres)
    dfres = dfres.set_index('time')
    return dfres

def fit_frame(xyzdat, peakselect, time, p0):

    """Orders processing of a single timedelay frame."""
    
    try:
        xy = np.append(xyzdat[0], xyzdat[1])
        zfin, p0, func = fit_prepare(peakselect, xyzdat[2], p0)
        popt, pcov = opt.curve_fit(f=func, xdata=xy, ydata=zfin, p0=p0, method='dogbox', maxfev=1000)
        fitraveled = func(xy, popt)

        resrow = fit_calc(popt, pcov, time, func)
        
        diff = zfin - fitraveled
        qfit = sum((abs(diff))/popt[0])  
        resrow.insert(0, 'qfit', qfit)
        
        datforfig = xyzdat, fitraveled, diff
        errorcheck = False
        
    except RuntimeError:
        print("Error - curve_fit failed at t = " + str(time) )
        errorcheck = True
        datforfig, resrow, qfit = 0, 0, 0.001

    return datforfig, resrow, errorcheck, qfit

def plot_frame(df, datforfig, frame):

    """Generates the figure"""
    
    xyzdat, fitraveled, diff = datforfig
    extx = (np.min(xyzdat[0]), np.max(xyzdat[0]), np.min(xyzdat[1]), np.max(xyzdat[1]))
    
    fig = plt.figure(figsize=(8,3))

    plt.subplot(1, 2, 1)

    contour_plot(df) 
    plt.tricontour(xyzdat[0], xyzdat[1], fitraveled, 5, extent=extx, linewidths=1, colors='k')
    plt.title('Fit')
    
    x1d, y1d, z2draw = df_separate(df)
    z2d = np.reshape(diff, (len(y1d), len(x1d)))
    diffdf = df_join(x1d, y1d, z2d.transpose())

    plt.subplot(1, 2, 2)
    contour_plot(diffdf)
    plt.title('Difference')
    
    plt.subplots_adjust(top=0.8, bottom=0.196, left=0.138, right=0.898, hspace=0.2, wspace=0.55)
    fig.patch.set_facecolor('w')
    plt.suptitle(str(frame), verticalalignment = 'top')
    plt.show()
    return fig

def fit_dataset(direct, peakselect, p0, ranges, polarisation, savedata=False, oneframe=False, showfig=False):
    
    """Fits the entire dataset"""
    
    paths, times = [], []
    combres = pd.DataFrame()

    for filename in os.listdir(direct):
        if filename.endswith("2DIR.csv"):
            frame = str(os.path.join(direct, filename))
            time = timestring(frame)
            paths.append(frame)
            times.append(time)
            
    pt = pd.DataFrame(data={'times': times, 'paths': paths}) 
    pt = pt.set_index('times')
    pt = pt.sort_index(axis=0)
    paths = np.array(pt['paths'])

    if oneframe is True:
        paths = paths[0:1]

    qfit = 0
    for i in range(len(paths)):
        frame = paths[i]
        time = np.array(pt.index)[i]

        dffr = pd.read_csv(frame, header=None, delimiter='\t')
        dffr = polarisation_select(dffr, polarisation)
        dfframe = crop_df(dffr, ranges) 
        xyzdat = df_to_xyz(dfframe)
        datforfig, resrow, error2, qfitcur = fit_frame(xyzdat, peakselect, time, p0)
        
        if error2 is False:
            qrat = abs((qfitcur-qfit)/qfitcur)
            error1 = False
            if qrat > 50:
                error1 = True
            else:
                qfit = qfitcur
                combres = pd.concat([combres, resrow])                                
                combres = combres.sort_index(axis=0)
                if showfig is True: 
                    print('-----------------------------')
                    fig = plot_frame(dfframe, datforfig, frame) 
                    if savedata is True:
                        fig.savefig((direct + '/' + str(time) + '.png'), dpi=200, facecolor='w')
                        combres.to_csv((direct + '/' + 'fit_results.csv'), sep=',')
    
    cars = {'dir': direct,  'pol': polarisation, 'ranges': ranges, 'p0': p0, 'peak': peakselect}
    for k, v in cars.items():
        print('%s: %s' % (k, v))
    return combres



### SIMPLE FFCF GRAPHING ###

def graph_ffcf(results, param, p0exp):
    
    """Plots a result column and fits it to an exponential"""
    
    x = np.array(results.index)
    y = np.array(results[param])

    def exp(x, a, b):
        return a*np.exp(-x/b) 

    popt, pcov = opt.curve_fit(exp, x, y, p0=p0exp)
    t = np.linspace(0, np.max(x), 100)
    yfit = exp(t, float(popt[0]), float(popt[1]))
    
    fig = plt.figure()
    plt.plot(x, y, 's')
    plt.plot(t, yfit, 'r--')
    plt.xlabel('time / ps')
    plt.ylabel('FFCF')
    fig.patch.set_facecolor('w')
    plt.show()
    
    print('y = ' + "%.5g" % (popt[0]) + ' * exp(- t / ' + "%.5g" % (popt[1]) + ')')
    return