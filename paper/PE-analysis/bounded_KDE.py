import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects as PathEffects

import pandas as pd
from six import string_types
from scipy.stats import gaussian_kde as kde



class Bounded_2d_kde(kde):
    r"""Represents a two-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, xlow=None, xhigh=None, ylow=None, yhigh=None, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param xlow: The lower x domain boundary.

        :param xhigh: The upper x domain boundary.

        :param ylow: The lower y domain boundary.

        :param yhigh: The upper y domain boundary.
        """
        pts = np.atleast_2d(pts)

        assert pts.ndim == 2, 'Bounded_kde can only be two-dimensional'

        super(Bounded_2d_kde, self).__init__(pts.T, *args, **kwargs)

        self._xlow = xlow
        self._xhigh = xhigh
        self._ylow = ylow
        self._yhigh = yhigh

    @property
    def xlow(self):
        """The lower bound of the x domain."""
        return self._xlow

    @property
    def xhigh(self):
        """The upper bound of the x domain."""
        return self._xhigh

    @property
    def ylow(self):
        """The lower bound of the y domain."""
        return self._ylow

    @property
    def yhigh(self):
        """The upper bound of the y domain."""
        return self._yhigh

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points."""
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'points must be two-dimensional'

        pdf = super(Bounded_2d_kde, self).evaluate(pts.T)
        
        if np.any([self.xlow, self.xhigh, self.ylow, self.yhigh]):
            pts_p = pts.copy()
            
        if self.ylow is not None:
            pts_p[:, 1] = 2.0*self.ylow - pts_p[:, 1]
            pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
            pts_p[:, 1] = 2.0*self.ylow - pts_p[:, 1]
        
        if self.yhigh is not None:
            pts_p[:, 1] = 2.0*self.yhigh - pts_p[:, 1]
            pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
            pts_p[:, 1] = 2.0*self.yhigh + pts_p[:, 1]
            
        if self.xlow is not None:
            pts_p[:, 0] = 2.0*self.xlow - pts[:, 0]
            pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
            if self.ylow is not None:
                pts_p[:, 1] = 2.0*self.ylow - pts_p[:, 1]
                pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
                pts_p[:, 1] = 2.0*self.ylow - pts_p[:, 1]
            if self.yhigh is not None:
                pts_p[:, 1] = 2.0*self.yhigh - pts_p[:, 1]
                pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
                pts_p[:, 1] = 2.0*self.yhigh - pts_p[:, 1]

        if self.xhigh is not None:
            pts_p[:, 0] = 2.0*self.xhigh - pts[:, 0]
            pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
            if self.ylow is not None:
                pts_p[:, 1] = 2.0*self.ylow - pts_p[:, 1]
                pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
                pts_p[:, 1] = 2.0*self.ylow - pts_p[:, 1]
            if self.yhigh is not None:
                pts_p[:, 1] = 2.0*self.yhigh - pts_p[:, 1]
                pdf += super(Bounded_2d_kde, self).evaluate(pts_p.T)
                pts_p[:, 1] = 2.0*self.yhigh - pts_p[:, 1]
   
        return pdf

    def __call__(self, pts):
        pts = np.atleast_2d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')
        
        if self.xlow is not None:
            out_of_bounds[pts[:, 0] < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts[:, 0] > self.xhigh] = True
        if self.ylow is not None:
            out_of_bounds[pts[:, 1] < self.ylow] = True
        if self.yhigh is not None:
            out_of_bounds[pts[:, 1] > self.yhigh] = True
            
        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results

def ms2q(pts):
    pts = np.atleast_2d(pts)
    
    m1 = pts[:, 0]
    m2 = pts[:, 1]
    mc = np.power(m1 * m2, 3./5.) * np.power(m1 + m2, -1./5.)
    q = m2/m1
    return np.column_stack([mc, q])


# MP 07/16 changes to plot_bounded_2d_kde:
# * add show_clabel
# * return cset
# * bugfix for mpl >= 1.5.1
def plot_bounded_2d_kde(data, data2=None, levels=None, xlow=None, xhigh=None, ylow=None, yhigh=None, transform=None,
                shade=False, vertical=False, 
                bw="scott", gridsize=500, cut=3, clip=None, legend=True, show_clabel=True,
                shade_lowest=True, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if transform is None:
        transform = lambda x: x

    data = data.astype(np.float64)
    if data2 is not None:
        data2 = data2.astype(np.float64)

    bivariate = False
    if isinstance(data, np.ndarray) and np.ndim(data) > 1:
        bivariate = True
        x, y = data.T
    elif isinstance(data, pd.DataFrame) and np.ndim(data) > 1:
        bivariate = True
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
    elif data2 is not None:
        bivariate = True
        x = data
        y = data2

    if bivariate==False:
        raise TypeError("Bounded 2D KDE are only available for"
                        "bivariate distributions.")
    else:
        # Determine the clipping
        if clip is None:
            clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
        elif np.ndim(clip) == 1:
            clip = [clip, clip]

        # Calculate the KDE
        if levels==None:
            kde_pts = transform(np.array([x, y]).T)
            post_kde = Bounded_2d_kde(kde_pts, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)
        else:
            pts = np.array([x, y]).T
            Npts = pts.shape[0]
            kde_pts = transform(pts[:Npts//2,:])
            den_pts = transform(pts[Npts//2:,:])

            Nden = den_pts.shape[0]

            post_kde=Bounded_2d_kde(kde_pts, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)
            den=post_kde(den_pts)
            densort=np.sort(den)[::-1]

            zvalues=[]
            for level in levels:
                ilevel = int(Nden*level + 0.5)
                if ilevel >= Nden:
                    ilevel = Nden-1
                zvalues.append(densort[ilevel])

        deltax = x.max() - x.min()
        deltay = y.max() - y.min()
        x_pts = np.linspace(x.min() - .1*deltax, x.max() + .1*deltax, gridsize)
        y_pts = np.linspace(y.min() - .1*deltay, y.max() + .1*deltay, gridsize)

        xx, yy = np.meshgrid(x_pts, y_pts)

        positions = np.column_stack([xx.ravel(), yy.ravel()])

        z = np.reshape(post_kde(transform(positions)), xx.shape)

        # Black (thin) contours with while outlines by default
        kwargs['linewidths'] = kwargs.get('linewidths', 1.)

        # Plot the contours
        n_levels = kwargs.pop("n_levels", 10)
        cmap = kwargs.get("cmap", None)

        if cmap is None:
            kwargs['colors'] = kwargs.get('colors', 'k')
        if isinstance(cmap, string_types):
            if cmap.endswith("_d"):
                pal = ["#333333"]
                pal.extend(sns.palettes.color_palette(cmap.replace("_d", "_r"), 2))
                cmap = sns.palettes.blend_palette(pal, as_cmap=True)
            else:
                cmap = plt.cm.get_cmap(cmap)

        kwargs["cmap"] = cmap
        contour_func = ax.contourf if shade else ax.contour
        if levels:
            # MP: change in recent matplotlib: sort zvalues and reverse levels
            if mpl.__version__ > '1.5.1':
                levels = levels[::-1]
                zvalues.sort()
            cset = contour_func(xx, yy, z, zvalues, **kwargs)

            # Add white outlines
            if kwargs['colors'] == 'k':
                plt.setp(cset.collections,
                         path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="w")])
            fmt = {}
            strs = ['{}%'.format(int(100*level)) for level in levels]
            for l, s in zip(cset.levels, strs):
                fmt[l] = s

            #plt.clabel(cset, cset.levels, fmt=fmt, fontsize=11, manual=True, **kwargs)#, use_clabeltext=True)
            if show_clabel:
                plt.clabel(cset, cset.levels, fmt=fmt, fontsize=11, **kwargs)#, use_clabeltext=True)
                plt.setp(cset.labelTexts, color='k', path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="w")])
        else:
            cset = contour_func(xx, yy, z, n_levels, **kwargs)

        if shade and not shade_lowest:
            cset.collections[0].set_alpha(0)

        # Avoid gaps between patches when saving as PDF
        if shade:
            for c in cset.collections:
                c.set_edgecolor("face")

        kwargs["n_levels"] = n_levels

        # Label the axes
        if hasattr(x, "name") and legend:
            ax.set_xlabel(x.name)
        if hasattr(y, "name") and legend:
            ax.set_ylabel(y.name)

    return ax, cset
