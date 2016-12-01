import numpy as np
import matplotlib.pyplot as plt

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
