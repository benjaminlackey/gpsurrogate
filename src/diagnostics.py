import numpy as np
import matplotlib.pyplot as plt

########################## 2d error plots ##############################
def max_error_2d_projection_plot(axes, x, y, error, threshold=None,
                                 x_label='x', y_label='y', colorbar=None, colorbarlabel='error'):
    """2d plot of the errors.
    Parameters
    ----------
    x : array
    y : array
    error : array
    threshold : float
        Only plot points with error above this value.
    """
    # Sort errors so largest errors are plotted on top of smaller errors
    error_params = np.array([error, x, y]).T
    error_params_sort = error_params[error_params[:, 0].argsort()]

    # Remove points below threshold
    if threshold!=None:
        ep = error_params_sort[error_params_sort[:, 0]>=threshold]
    else:
        ep = error_params_sort

    # min and max errors after threshold cut
    e_min = ep[0, 0]
    e_max = ep[-1, 0]

    # Scatter plot with colorbar
    # Point sizes from 2 to 100
    sc = axes.scatter(ep[:, 1], ep[:, 2], c=ep[:, 0], s=2+98*(ep[:, 0]-e_min)/e_max,
                      edgecolor='', alpha=1.0)

    if colorbar:
        cb = plt.colorbar(mappable=sc, ax=axes)
        cb.set_label(label=colorbarlabel, fontsize=18)
        cb.ax.tick_params(labelsize=14)

    # buffers for plot
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    bufx = 0.05*(xmax - xmin)
    bufy = 0.05*(ymax - ymin)

    axes.set_xlim([xmin-bufx, xmax+bufx])
    axes.set_ylim([ymin-bufy, ymax+bufy])
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.minorticks_on()


def error_2d_triangle_plot(params, error, labels, threshold=None, figsize=(10, 10)):
    """(nparams-1) x (nparams-1) triangle plot of 2d errors.
    """
    # Make params an array
    ps = np.array(params)
    # Number of parameters
    nparams = len(ps[0])
    fig, ax = plt.subplots(nparams-1, nparams-1, figsize=(12, 12))
    # 1st index y_i: top to bottom
    # 2nd index x_i: left to right
    for y_i in range(nparams-1):
        for x_i in range(nparams-1):
            if x_i <= y_i:
                # Left triangle
                ax[y_i, x_i].plot(0, 0, label=labels[x_i]+labels[y_i+1])
                x = ps[:, x_i]
                y = ps[:, y_i+1]
                max_error_2d_projection_plot(ax[y_i, x_i], x, y, error, threshold=threshold,
                                             x_label=labels[x_i], y_label=labels[y_i+1],
                                             colorbar=None)
            else:
                # turn off axes for right triangle
                ax[y_i, x_i].axis('off')

    for y_i in range(nparams-2):
        for x_i in range(nparams-1):
            ax[y_i, x_i].xaxis.set_ticklabels([])
            ax[y_i, x_i].set_xlabel('')

    for x_i in range(nparams-1):
        plt.setp(ax[-1, x_i].get_xticklabels(), rotation=45)
    #tick = ax[-1, x_i].get_xticklabels()
    #tick.set_rotation(45)

    for y_i in range(nparams-1):
        for x_i in range(1, nparams-1):
            ax[y_i, x_i].yaxis.set_ticklabels([])
            ax[y_i, x_i].set_ylabel('')

    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    return fig, ax

#################### 1d cross section plots ###########################

def plot_1d_amp_phase_variation(ax_amp, ax_phase, sur, params, xi, xlow, xhigh, nx, mfs, legend=True):
    """Plot amplitude(1parameter) and phase(1parameter) as the single parameter is varied.
    Do this for all the frequencies mfs.

    Parameters
    ----------
    params : 1d array
        Point in full parameter space where the 1d slice should pass through
    xi : int
        Index of the parameter you want to vary
    xlow, xhigh, nx : bounds and number of samples for the feature
    mfs : 1d array
        Frequencies at which to plot the 1d variation.
    """
    xs = np.linspace(xlow, xhigh, nx)
    ps = np.array([params for j in range(len(xs))])
    for j in range(len(xs)):
        ps[j, xi] = xs[j]

    dhs = [sur.amp_phase_difference(p) for p in ps]

    amp_list = []
    phase_list = []
    for i in range(len(dhs)):
        amps = dhs[i].interpolate('amp')(mfs)
        phases = dhs[i].interpolate('phase')(mfs)
        amp_list.append(amps)
        phase_list.append(phases)
    amp_array = np.array(amp_list).T
    phase_array = np.array(phase_list).T

    for j in range(len(mfs)):
        ax_amp.plot(xs, amp_array[j], label=mfs[j])

    for j in range(len(mfs)):
        ax_phase.plot(xs, phase_array[j])

    if legend:
        ax_amp.legend(ncol=2, frameon=False)


def plot_all_1d_slices(sur, params, limits, labels, mfs, nx=20):
    """Plot amplitude(1parameter) and phase(1parameter) as the single parameter is varied.
    Repeat this for all parameters.

    Parameters
    ----------
    params : 1d array
        Point in full parameter space where the 1d slice should pass through
    xi : int
        Index of the parameter you want to vary
    xlow, xhigh, nx : bounds and number of samples for the feature
    mfs : 1d array
        Frequencies at which to plot the 1d variation.
    """
    nparams = len(limits)
    fig, ax = plt.subplots(2, nparams, figsize=(min(8*nparams, 20), 6))
    title = '{:.3}, {:.2}, {:.2}, {:.1f}, {:.1f}'.format(params[0], params[1], params[2], params[3], params[4])
    fig.suptitle(title)
    for xi in range(nparams):
        ax1 = ax[0, xi]
        ax2 = ax[1, xi]
        xlow = limits[xi, 0]
        xhigh = limits[xi, 1]
        if xi==0:
            legend = True
            ax1.set_ylabel(r'$\ln(A/A_{\rm F2})$')
            ax2.set_ylabel(r'$\Phi-\Phi_{\rm F2}$')
        else:
            legend = False
        plot_1d_amp_phase_variation(ax1, ax2, sur, params, xi, xlow, xhigh, nx, mfs, legend=legend)
        ax1.minorticks_on()
        ax2.minorticks_on()
        ax1.xaxis.set_ticklabels([])
        ax2.set_xlabel(labels[xi])
        # Manually rotate every single label
        for label in ax2.get_xticklabels():
            label.set_rotation(45)
    fig.subplots_adjust(hspace=0.0, wspace=0.25)

    
