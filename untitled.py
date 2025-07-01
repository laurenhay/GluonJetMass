import hist
import awkward as ak
from hist import intervals
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib as mpl
##############################################################################################    
## Rebin function for hists
##############################################################################################
## Reference: https://github.com/kdlong/WRemnants/blob/main/utilities/boostHistHelpers.py#L171-L213 
def rebin_hist(h, axis_name, edges):
    if type(edges) == int:
        return h[{axis_name : hist.rebin(edges)}]

    ax = h.axes[axis_name]
    ax_idx = [a.name for a in h.axes].index(axis_name)
    if not all([np.isclose(x, ax.edges).any() for x in edges]):
        raise ValueError(f"Cannot rebin histogram due to incompatible edges for axis '{ax.name}'\n"
                            f"Edges of histogram are {ax.edges}, requested rebinning to {edges}")
        
    # If you rebin to a subset of initial range, keep the overflow and underflow
    overflow = ax.traits.overflow or (edges[-1] < ax.edges[-1] and not np.isclose(edges[-1], ax.edges[-1]))
    underflow = ax.traits.underflow or (edges[0] > ax.edges[0] and not np.isclose(edges[0], ax.edges[0]))
    flow = overflow or underflow
    new_ax = hist.axis.Variable(edges, name=ax.name, overflow=overflow, underflow=underflow)
    axes = list(h.axes)
    axes[ax_idx] = new_ax
    
    hnew = hist.Hist(*axes, name=h.name, storage=h._storage_type())

    offset = 0.5*np.min(ax.edges[1:]-ax.edges[:-1])
    edges_eval = edges+offset
    edge_idx = ax.index(edges_eval)

    if len(np.unique(edge_idx)) != len(edge_idx):
        raise ValueError("Did not find a unique binning. Probably this is a numeric issue with bin boundaries")

    if underflow:
        # Only if the original axis had an underflow should you offset
        if ax.traits.underflow:
            edge_idx += 1
        edge_idx = np.insert(edge_idx, 0, 0)

# Take is used because reduceat sums i:len(array) for the last entry, in the case
    # where the final bin isn't the same between the initial and rebinned histogram, you
    # want to drop this value. 
    hnew.values(flow=flow)[...] = np.add.reduceat(h.values(flow=flow), edge_idx, 
            axis=ax_idx).take(indices=range(new_ax.size+underflow+overflow), axis=ax_idx)
    if hnew._storage_type() == hist.storage.Weight():
        hnew.variances(flow=flow)[...] = np.add.reduceat(h.variances(flow=flow), edge_idx, 
                axis=ax_idx).take(indices=range(new_ax.size+underflow+overflow), axis=ax_idx)
    return hnew

##############################################################################################
## Plotting Script for systematics.
##############################################################################################
def systPlot(output, axName, edges, scale, factor, xVar, axisVar, systematics, label, IOV = '2018',
             doOverflow = True, ratio=True, mcOnly=True, xlabel='',
             ymin = 0.00001, ylim=0.5, logy=False):
    
    #systematic = array contaning strings for up down systematics
    xlim = edges[-1]
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    if ratio:
        fig, (ax, rax) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(7,7),
            gridspec_kw={"height_ratios": (3, 1)},
            sharex=True
        )
        fig.subplots_adjust(hspace=.07)
    else:
        fig, ax, = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(7,7)
        )
            
    # Here is an example of setting up a color cycler to color the various fill patches
    # We get the colors from this useful utility: http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=6
    from cycler import cycler
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c', '#9F79EE']
    #colors = ['#fb9a99','#33a02c','#b2df8a','#1f78b4','#a6cee3']
    #colors = ['#DC141C','#1f78b4','#FF6EB4','#33a02c','#9F79EE','#b2df8a']#'#fb9a99']#,'#e31a1c']

    ax.set_prop_cycle(cycler(color=colors))

    fill_opts = {
        'edgecolor': (0,0,0,0.3),
        'alpha': 0.8
    }
    error_opts = {
        'label': 'Stat. Unc.',
        'hatch': '///',
        'facecolor': 'none',
        'edgecolor': (0,0,0,.5),
        'linewidth': 0
    }
    data_err_opts = {
        'linestyle': 'none',
        'marker': '.',
        'markersize': 10.,
        'color': 'k',
        'elinewidth': 1,
    }        
            
        mc_hist = {}
        for syst in systematics:
            mc_sum = []
            for out in output:
                name = "MC"
                h = out[xVar]
                print("Histogram chosen: ", h)
                print("Histogram axes: ", h.axes)
                if (len(h.axes) ==5) & (syst in h.axes[3]):
                    h1 = h[:, [hist.loc(j) for j in axVar], [hist.loc(sys)], :]
                    h1.view().value = np.nan_to_num(h1.values())
                    h1.view().variance = np.nan_to_num(h1.variances())
h1 = h1[sum, sum, sum, sum, :]
                    h1 = h1 * scale[name]
                    if doOverflow: h1.values()[-1] += h1.values(flow=True)[-1]
                    h1 = np.nan_to_num(h1.values())
                    mc_sum.append(h1)
            mc_hist[syst]= np.nansum(np.array(mc_sum), axis = 0)
        if len(systematics)>2:
            mc = [mc_hist[systematic[0]],  mc_hist[systematic[1]], mc_hist["nominal"]]
            lines = ["-", "-", "-"]
            cols = ['green', 'red', 'black']
        else:
            mc = [mc_hist[systematic[0]], mc_hist["nominal"]]
            lines = ["-", "-"]
            cols = ['red', 'black']

#-------------------------------------Background plotting-------------------------------------#


        # plot the MC first
        hep.histplot(mc, edges, stack=False, histtype='step', binwnorm=1,
                     ax=ax, density=False, linestyle =lines,
                     color = cols, linewidth=1,
                     label=systematic)
                    
#-----------------------------------------Ratio plot-----------------------------------------#

        ax.autoscale(axis='x', tight=True)
        leg = ax.legend(labels=systematic, loc='best', labelspacing=0.25)
        leg.set_visible(True)

        # now we build the ratio plot
        if ratio:
            ax.set_xlabel(None)
ratio_D = np.divide(
                mc_hist[systematic[1]],
                mc_hist["nominal"],
                out=np.empty(np.array(mc_hist["nominal"]).shape).fill(np.nan),
                #out=np.ones_like(data),
                where=mc_hist["nominal"]!= 0,
            )
            if len(systematic)>2:
                ratio_U = np.divide(
                    mc_hist[systematic[0]],
                    mc_hist["nominal"],
                    out=np.empty(np.array(mc_hist["nominal"]).shape).fill(np.nan),
                    #out=np.ones_like(data),
                    where=mc_hist["nominal"]!= 0,
                )

                hep.histplot(ratio_U, edges, stack=False, histtype='step',
                         ax=rax, density=False, linestyle ="-", color = 'green', linewidth=1)
            hep.histplot(ratio_D, edges, stack=False, histtype='step',
                     ax=rax, density=False, linestyle ="-", color = 'red', linewidth=1)
            
            hep.histplot(np.ones_like(ratio_D), edges, stack=False, histtype='step',
                     ax=rax, density=False, linestyle ="--", color = 'black', linewidth=1)
            
            rax.set_ylabel(r'$\frac{Up/Down}{Nominal}$', loc = 'center')
            rax.set_ylim(0.5,1.5)
            if 'q2' in systematic[0]:
                rax.set_ylim(0.,2.0)
                hep.histplot(np.ones_like(ratio_D)*0.5, edges, stack=False, histtype='step',
                             ax=rax, density=False, linestyle ="dotted", color = 'black', linewidth=0.8)
                hep.histplot(np.ones_like(ratio_D)*1.5, edges, stack=False, histtype='step',
                             ax=rax, density=False, linestyle ="dotted", color = 'black', linewidth=0.8)
            else:
                hep.histplot(np.ones_like(ratio_D)*0.8, edges, stack=False, histtype='step',
                             ax=rax, density=False, linestyle ="dotted", color = 'black', linewidth=0.8)
                hep.histplot(np.ones_like(ratio_D)*1.2, edges, stack=False, histtype='step',
                             ax=rax, density=False, linestyle ="dotted", color = 'black', linewidth=0.8)
            
            if ("eta" in xVar) | ("phi" in xVar):
rax.set_xlim(-xlim, xlim)
            else:
                rax.set_xlim(0, xlim)
            rax.set_xlabel(xlabel)
        else:
            if ("eta" in xVar) | ("phi" in xVar):
                ax.set_xlim(-xlim, xlim)
            else:
                ax.set_xlim(0, xlim)
            ax.set_xlabel(xlabel)
        
        if logy: 
            ax.set_yscale('log')
            ax.set_ylim(ymin, ylim[sc])
        else:
            ax.set_ylim(0, ylim[sc])

        ax.set_ylabel(r'Events/bin', loc = 'top')
        # add some labels
        cms = plt.text(0.04, 0.87, 'CMS $\it{Simulation}$',
                          fontsize=22,
                          fontfamily='sans',
                          fontweight='bold',
                          horizontalalignment='left',
                          verticalalignment='bottom',
                          transform=ax.transAxes
                         )
        subc = sc.replace('m', '\mu')
        subchannel = plt.text(0.04, 0.78, '${}$'.format(subc),
                          fontsize=16,
                          fontfamily='sans',
                          fontweight='bold',
                          horizontalalignment='left',
                          verticalalignment='bottom',
                          transform=ax.transAxes
                         )
        regions = ['SR_nonboosted', 'CR0', 'SR_boosted']
        txt = [item for item in axisVar if item in regions]
        if (len(txt)==1) & (txt[0] == 'CR0'):
typeTxt = 'Control Region'
        elif (len(txt)==1) & (txt[0] == 'SR_nonboosted'):
            typeTxt = 'Resolved Signal Region'
        elif (len(txt)==1) & (txt[0] == 'SR_boosted'):
            typeTxt = 'Boosted Signal Region'
        # elif (len(txt)==3):
        #     typeTxt = ', Pre-selection'
        else:
            typeTxt = ''
    
        if len(typeTxt):
            region = plt.text(0.045, 0.83, '{}'.format(typeTxt),
                              fontsize=12,
                              fontfamily='sans',
                              horizontalalignment='left',
                              verticalalignment='bottom',
                              transform=ax.transAxes
                             )
        lumi = plt.text(1., 1., label,
                        fontsize=16,
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        transform=ax.transAxes
                       )
        ax.yaxis.get_minor_locator().set_params(numticks=999, subs=(.2, .4, .6, .8))
        sy = systematic[0].split("_")[0]
        if ("corr" in systematic[0])&("_U" in systematic[0]):
            sy = systematic[0].split("_U")[0]
        elif ("corr" in systematic[0])&("_D" in systematic[0]):
            sy = systematic[0].split("_D")[0]
        print("Saving plots as plots/UL{}/sys/UL{}_{}_{}_{}_{}.pdf".format(IOV, IOV, xVar, sc, sy, '_'.join(axisVar)))
        fig.savefig("plots/UL{}/sys/UL{}_{}_{}_{}_{}.pdf".format(IOV, IOV, xVar, sc, sy, '_'.join(axisVar)), bbox_inches='tight')
        
        


