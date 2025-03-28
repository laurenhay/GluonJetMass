# Plotting functions

import awkward as ak
import numpy as np
import pickle
import hist
import coffea
from python.plugins import *

# from hist import intervals
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib as mpl
###### Kevin's script for nonunifom rebinning https://github.com/scikit-hep/hist/issues/345
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

    # Offset from bin edge to avoid numeric issues
    offset = 0.5*np.min(ax.edges[1:]-ax.edges[:-1])
    edges_eval = edges+offset
    edge_idx = ax.index(edges_eval)
    # Avoid going outside the range, reduceat will add the last index anyway
    if edge_idx[-1] == ax.size+ax.traits.overflow:
        edge_idx = edge_idx[:-1]

    if underflow:
        # Only if the original axis had an underflow should you offset
        if ax.traits.underflow:
            edge_idx += 1
        edge_idx = np.insert(edge_idx, 0, 0)

    # Take is used because reduceat sums i:len(array) for the last entry, in the case
    # where the final bin isn't the same between the initial and rebinned histogram, you
    # want to drop this value. Add tolerance of 1/2 min bin width to avoid numeric issues
    hnew.values(flow=flow)[...] = np.add.reduceat(h.values(flow=flow), edge_idx, 
            axis=ax_idx).take(indices=range(new_ax.size+underflow+overflow), axis=ax_idx)
    if hnew._storage_type() == hist.storage.Weight():
        hnew.variances(flow=flow)[...] = np.add.reduceat(h.variances(flow=flow), edge_idx, 
                axis=ax_idx).take(indices=range(new_ax.size+underflow+overflow), axis=ax_idx)
    return hnew

def getTotSyst(result, histname, axis='mreco'):
    hist = result[histname]
    availSysts = [ax for ax in result[histname].project("syst").axes[0]]
#     syst_uncorr = [
#     'jerUp','jerDown'
#     'PUSFUp','PUSFDown', #PDFs WIP
#     'L1prefiringDown', 'L1prefiringUp', 
#     'AbsoluteStatUp','AbsoluteStatDown',
#     'FlavorQCDUp','FlavorQCDDown',
#     'FragmentationUp','FragmentationDown',
#     'PileUpDataMCUp','PileUpDataMCDown',
#     'RelativeBalUp','RelativeBalDown',
#     'RelativeJEREC1Up','RelativeJEREC1Down',
#     'RelativeJEREC2Up','RelativeJEREC2Down',
#     'RelativePtEC1Up','RelativePtEC1Down',
#     'RelativePtEC2Up','RelativePtEC2Down',
#     'RelativeSampleUp','RelativeSampleDown',
#     'RelativeStatECUp','RelativeStatECDown',
#     'RelativeStatFSRUp','RelativeStatFSRDown',
#     'RelativeStatHFUp','RelativeStatHFDown',
# ]

#     absolute = ['AbsoluteMPFBiasUp', 'AbsoluteMPFBiasDown',
#                 'AbsoluteScaleUp', 'AbsoluteScaleDown']
#     pileuppt = ['PileUpPtBBUp', 'PileUpPtBBDown', 
#                 'PileUpPtEC1Up', 'PileUpPtEC1Down',
#                 'PileUpPtEC2Up', 'PileUpPtEC2Down', 
#                 'PileUpPtHFUp', 'PileUpPtHFDown', 
#                 'PileUpPtRefUp', 'PileUpPtRefDown']
#     relpt = ['RelativePtEC1Up', 'RelativePtEC1Down','RelativePtHFUp', 'RelativePtHFDown']
#     singlepion =     ['SinglePionECALUp', 'SinglePionECALDown','SinglePionHCALUp', 'SinglePionHCALDown']
#     timepteta = ['TimePtEtaUp', 'TimePtEtaDown']

    availSysts = [syst for syst in availSysts if syst!="nominal"]
    # syst_list = [syst for syst in availSysts if syst!="nominal" and syst not in syst_uncorr]
    # syst_uncorr = [syst for syst in syst_uncorr if syst in availSysts]
    # corr_grps = [absolute, pileuppt, relpt, singlepion, timepteta]
    sysErr = {}  
    for syst in availSysts:
        sysErr.update({syst: hist[{'syst':syst}].project(axis).values()})
    nom_values = hist[{'syst':'nominal'}].project(axis).values()
    sysErrTot_up = np.zeros_like(nom_values)
    sysErrTot_dn = np.zeros_like(nom_values)
    #### Loop through uncorrelated groups and make 
    # for grp in corr_grps:
    #     grp_unc_total_up  = np.zeros_like(nom_values)
    #     grp_unc_total_dn  = np.zeros_like(nom_values)
    #     for syst in grp:
    #         if syst in availSysts:
    #             syst_vals = sysErr[syst]
    #             if "Up" in syst:
    #                 deltasys = abs(syst_vals-nom_values)
    #                 grp_unc_total_up += deltasys
    #             if "Down" in syst:
    #                 deltasys = abs(nom_values-syst_vals)
    #                 grp_unc_total_dn += deltasys
    #     sysErrTot_up += grp_unc_total_up**2
    #     sysErrTot_dn += grp_unc_total_dn**2
    #### Loop through uncorrelated uncertainties and add together with correlated groups
    for syst, syst_vals in sysErr.items():
        if "Up" in syst:
            deltasys = syst_vals-nom_values
            sysErrTot_up = sysErrTot_up + deltasys**2
        if "Down" in syst:
            deltasys = nom_values-syst_vals
            sysErrTot_dn = sysErrTot_dn + deltasys**2
    sysErrTot_up = sysErrTot_up**0.5
    sysErrTot_dn = sysErrTot_dn**0.5
    return sysErrTot_up, sysErrTot_dn
def plotDataMCwErrorsBinned(result_mc, result_data, hist_mc, hist_data, IOV, channel = "", axVar="mreco", norm = True, rax_lim=None, os_path='plots/syst/', binwnorm=True, trim = False):
    pt_edges = [bin[0] for bin in hist_mc.project('ptreco').axes[0]] + [hist_mc.project('ptreco').axes[0][-1][1]]
    for i in range(len(pt_edges)-1):
        stat_unc_up = result_mc[hist_mc][{'syst':'nominal'}].project(axVar).variances()**0.5
        stat_unc_down = result_mc[hist_mc][{'syst':'nominal'}].project(axVar).variances()**0.5
        syst_unc_up, syst_unc_down = getTotSyst(result_mc, hist_mc, axis=axVar)
        print("Syst unc up vals: ", syst_unc_up)
        print("Syst unc down vals: ", syst_unc_down)
        #### following opts may be unnecessary
        tot_error_opts = {
                'label': 'Stat. + Syst. Unc.',
                'facecolor': 'orange',
    
                'linewidth': 0
            }
        stat_error_opts = {
                'label': 'Stat. Unc.',
                        'hatch': '///',
                        'edgecolor': 'black',
                'facecolor': 'none',
                'linewidth': 0
            }
        data_err_opts = {
                'linestyle': 'none',
                'marker': '.',
                'markersize': 10.,
                'color': 'k',
                'elinewidth': 1,
            }
        datahist = result_data[hist_data][{'syst':'nominal', 'ptreco':i}]
        mchist = result_mc[hist_mc][{'syst':'nominal', 'ptreco':i}]
        availAxes = [ax.name for ax in result_mc[hist_mc].axes]
        edges = [bin[0] for bin in result_mc[hist_mc].project(axVar).axes[0]] + [result_mc[hist_mc].project(axVar).axes[0][-1][1]]
        widths = result_mc[hist_mc].project(axVar).axes[0].widths
        if trim:
            print("Trimming last bin to be size of second to last bin")
            print("Last two widths ", widths[-2], " ", widths[-1])
            print("Last two edges ", edges[-2], " ", edges[-1])
            widths[-1] = widths[-2]
            edges[-1] = edges[2] + width[-1]
            ax.set_xticklabels(edges)
        print("widths", widths)
        xlim = edges[-1]
        fig, (ax, rax) = plt.subplots(
                    nrows=2,
                    ncols=1,
                    figsize=(8,7),
                    gridspec_kw={"height_ratios": (3, 1)},
                    sharex=True)
        ax.yaxis.get_minor_locator().set_params(numticks=999, subs=(.2, .4, .6, .8))
        ax.set_ylabel(r'$Events/GeV$', loc = 'top')
        if not binwnorm:
            ax.set_ylabel(r'$Events$', loc = 'top')
        ax.set_yscale('log')
        if "_g" in hist_mc and "m"==axVar[0]:
            rax.set_xlabel(r'$m_{SD, RECO} (GeV)$' )
        else:
            rax.set_xlabel(r'$m_{SD, RECO} (GeV)$' )
        ratio = np.ones_like(result_mc[hist_mc].project(axVar).values())
        #### Fill ratio plot
        ax.set_xlabel("")
        if norm:
            mcvals = mchist.project(axVar)*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
            datavals = datahist.project(axVar)
            stat_unc_up = stat_unc_up*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
            stat_unc_down = stat_unc_down*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
            syst_unc_up = syst_unc_up*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
            syst_unc_down = syst_unc_down*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        else:
            mcvals = mchist.project(axVar)
            datavals = datahist.project(axVar)
        ratio = np.divide(mcvals.values(),datavals.values(),
                          out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                          where=datavals.values()!= 0,)
        #### Add MC error bars
        print("Total unc up: ", mcvals.values()+stat_unc_up+syst_unc_up)
        ratio_up_tot = mcvals.values()+(stat_unc_up**2+syst_unc_up**2)**0.5
        ratio_dn_tot = mcvals.values()-(stat_unc_down**2+syst_unc_down**2)**0.5
        ratio_up_syst = mcvals.values()+stat_unc_up
        ratio_dn_syst = mcvals.values()-stat_unc_down
        print("Unc. ratio upper bound ", ratio_up_tot)
        if binwnorm:
            ratio_up_tot=ratio_up_tot/widths
            print("Unc. ratio upper bound ", ratio_up_tot , " after dividing by bin widths")
            ratio_dn_tot=ratio_dn_tot/widths
            ratio_up_syst=ratio_up_syst/widths
            ratio_dn_syst=ratio_dn_syst/widths
        hep.histplot(datavals, stack=False, histtype='errorbar', binwnorm=binwnorm, 
                     ax=ax, marker =["."], color = 'Black', linewidth=1, 
                     label=channel + " Data")
        hep.histplot(mcvals, stack=False, histtype='step',
                     ax=ax, linestyle ='-', color = 'Black', linewidth=1, binwnorm = binwnorm,
                     label=channel + " MC "+" pt "+str(pt_edges[i])+"-"+str(pt_edges[i+1]))
        ax.stairs(values=ratio_up_tot, edges = edges, baseline= ratio_dn_tot, fill=True,
                  **tot_error_opts,
                )
        ax.stairs(values=ratio_up_syst, edges = edges, baseline= ratio_dn_syst, fill=True,
                    **stat_error_opts,
                )
        print("Data vals ", datavals.values(), " and MC vals ", mcvals.values())
        ax.autoscale(axis='x', tight=True)
        #### Want to stack uncertainties
            # print("Values in bins: ", mchist.project(axVar).values(), " errors of bins ", mchist.project(axVar).variances())
        if rax_lim != None:
            rax.set_ylim(rax_lim[0], rax_lim[1])
        leg = ax.legend(loc='upper right', labelspacing=0.25)
        leg.set_visible(True)
        #### Get ratio err values and plot
        ratio_totterr_up = np.divide((mcvals.values()+(stat_unc_up**2+syst_unc_up**2)**0.5),datavals.values(),
                          out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                          where=datavals.values()!= 0,)
        ratio_totterr_down = np.divide(mcvals.values()-(stat_unc_down**2+syst_unc_down**2)**0.5,datavals.values(),
                          out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                          where=datavals.values()!= 0,)
        ratio_statterr_up = np.divide(mcvals.values()+stat_unc_up,datavals.values(),
                          out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                          where=datavals.values()!= 0,)
        ratio_statterr_down = np.divide(mcvals.values()-stat_unc_up,datavals.values(),
                          out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                          where=datavals.values()!= 0,)

        rax.stairs(values=ratio_totterr_up, edges = edges, baseline= ratio_totterr_down,
                    fill=True,
                    **tot_error_opts,
                )
        rax.stairs(values=ratio_statterr_up, edges = edges, baseline= ratio_statterr_down,
                    fill=True,
                    **stat_error_opts,
                )
        hep.histplot(ratio, edges, histtype='step', ax=rax, linestyle ="-", color = 'black', linewidth=1)
        hep.histplot(np.ones_like(ratio), edges, histtype='step',ax=rax,linestyle ="--", color = 'black', linewidth=1)
        
        rax.set_ylabel(r'$MC/Data', loc = 'center')
        if ("rapidity" in axVar) | ("phi" in axVar):
            rax.set_xlim(-xlim, xlim)
        elif "pt" in axVar:
            rax.set_xlim(0, 2000)

            # rax.set_ylim(0.0,1.0)
        else:
            rax.set_xlim(0, xlim)
        hep.cms.label("Preliminary", data = True, loc=0, ax=ax, fontsize=18)
        ax.set_xlabel(None) 
        plt.show()
        if norm:
            fig.savefig(os_path+"ULwErrs{}{}_{}_{}_pt{}_{}normed.png".format(IOV,channel, hist_mc, axVar, pt_edges[i], pt_edges[i+1]), bbox_inches="tight") 
        else:
            fig.savefig(os_path+"ULwErrs{}{}_{}_{}_pt{}_{}.png".format(IOV,channel, hist_mc, axVar, pt_edges[i], pt_edges[i+1]), bbox_inches="tight") 
def plotSyst(result, histname, axVar, label, logy=True, IOV = '', channel='', os_path=""):
    from cycler import cycler
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c', '#9F79EE']
    fill_opts = {
            'edgecolor': (0,0,0,0.3),
            'alpha': 0.8}
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
    fig_tot, (ax_tot, rax_tot) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(8,7),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True)
    if "pt" in axVar:
        rax_tot.set_xlim(0, 2000)
    ax_tot.set_prop_cycle('color',plt.cm.Spectral(np.linspace(0,1,30)))
    rax_tot.set_prop_cycle('color',plt.cm.Spectral(np.linspace(0,1,30)))
    availAxes = [ax.name for ax in result[histname].axes]
    availSysts = [ax for ax in result[histname].project("syst").axes[0]]
    print("Available axes: ", availAxes)
    print("Available systs ", availSysts)
    # fig.subplots_adjust(hspace=.1)
    mc = []
    rax_lim_dict = {}
    edges = [bin[0] for bin in result[histname].project(axVar).axes[0]] + [result[histname].project(axVar).axes[0][-1][1]]
    xlim = edges[-1]
    print("Plotting all systematics available")
    systematics = [syst for syst in availSysts if syst !="nominal"]
    for i, syst in enumerate(systematics):
        lines = ["-", "-", "-"]
        cols = ['green', 'red', 'black']
        ax.set_xlabel("")
        #### Set up ratio plot
        if (syst in availSysts) and ("nominal" in availSysts) and (syst[-2:]=="Up"):
            fig, (ax, rax) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(8,7),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True)
            ax.yaxis.get_minor_locator().set_params(numticks=999, subs=(.2, .4, .6, .8))
            ax.set_ylabel(r'Events', loc = 'top')
            if logy: 
                ax.set_yscale('log')
            syst = syst[:-2]
            ratio_D = np.ones_like(result[histname][{"syst":syst+"Up",}].project(axVar).values())
            ratio_U = np.ones_like(result[histname][{"syst":syst+"Up",}].project(axVar).values())
            #### Get mc hist
            # print("Up values: ", result[histname][{'syst':syst+"Up"}].project(axVar).values())
            if (syst+"Down") in availSysts:
                ratio_D = np.divide(result[histname][{'syst':syst+"Down"}].project(axVar).values(),result[histname][{"syst":"nominal"}].project(axVar).values(),
                                    out=np.empty(np.array(result[histname][{"syst":"nominal"}].project(axVar).values()).shape).fill(np.nan),
                                    where=result[histname][{"syst":"nominal"}].project(axVar).values()!= 0,)
                hep.histplot(ratio_D, edges, stack=True, histtype='step', ax=rax, density=False, linestyle ="-", color = 'red', linewidth=1)
                # print("Down values: ", result[histname][{'syst':syst+"Down"}].project(axVar).values())
                hep.histplot(ratio_D, edges, stack=True, histtype='step', ax=rax_tot, density=False, linestyle ="-", linewidth=1)
                hep.histplot(result[histname][{'syst':syst+"Down"}].project(axVar), stack=False, histtype='step', binwnorm=True, ax=ax, density=False, linestyle ='-', color = 'red', linewidth=1,label=syst+"Down")
                hep.histplot(result[histname][{'syst':syst+"Down"}].project(axVar), stack=False, histtype='step', binwnorm=True, ax=ax_tot, density=False, linestyle ='-', linewidth=1,label=syst+"Down")
            #### Fill ratio plot
            # print("Nom values: ", result[histname][{'syst':"nominal"}].project(axVar).values())
            ratio_U = np.divide(
                result[histname][{'syst':syst+"Up"}].project(axVar).values(),
                result[histname][{"syst":"nominal"}].project(axVar).values(),
                out=np.empty(np.array(result[histname][{"syst":"nominal"}].project(axVar).values()).shape).fill(np.nan),
                where=result[histname][{"syst":"nominal"}].project(axVar).values()!= 0,)
            hep.histplot(result[histname][{'syst':syst+"Up"}].project(axVar), stack=False, histtype='step', binwnorm=True, ax=ax, density=False, linestyle ='-', color = 'green', linewidth=1,label=syst+"Up")
            hep.histplot(result[histname][{'syst':syst+"Up"}].project(axVar), stack=False, histtype='step', binwnorm=True, ax=ax_tot, density=False, linestyle ='-', linewidth=1,label=syst+"Up")
            ax.autoscale(axis='x', tight=True)
            leg = ax.legend(loc='upper right', labelspacing=0.25)
            leg.set_visible(True)
            hep.histplot(ratio_U, edges, stack=False, histtype='step', ax=rax, linestyle ="-", color = 'green', linewidth=1)
            hep.histplot(ratio_U, edges, stack=False, histtype='step', ax=rax_tot, linestyle ="-", linewidth=1)
            #### plotting nominal values
            hep.histplot(np.ones_like(ratio_U), edges, stack=False, histtype='step',ax=rax, linestyle ="--", color = 'black', linewidth=1)
            hep.histplot(result[histname][{'syst':"nominal"}].project(axVar), stack=False, histtype='step', binwnorm=True, ax=ax, density=False, linestyle ='-', color = 'black', linewidth=1,label="nominal")
            hep.histplot(result[histname][{'syst':"nominal"}].project(axVar), stack=False, histtype='step', binwnorm=True, ax=ax_tot, density=False, linestyle ='-', color = 'black', linewidth=1,label="nominal")
            rax.set_ylabel(r'$\frac{Up/Down}{Nominal}$', loc = 'center')
            rax.set_ylim(0.9,1.1)
            rax_tot.set_ylabel(r'$\frac{Up/Down}{Nominal}$', loc = 'center')
            # rax_tot.set_ylim(0.9,1.1)
            if ("rapidity" in axVar) | ("phi" in axVar):
                rax.set_xlim(-xlim, xlim)
            elif "pt" in axVar:
                rax.set_xlim(0, 2000)
            else:
                rax.set_xlim(0, xlim)
            cms = plt.text(0.04, 0.87, 'CMS $\it{Simulation}$',
                          fontsize=22,
                          fontfamily='sans',
                          fontweight='bold',
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
            fig.savefig(os_path+"/UL{}{}_{}_{}_{}.png".format(IOV,channel, histname, axVar, syst), bbox_inches='tight') 
        elif syst in availSysts and "nominal" not in availSysts:
            fig, ax, = plt.subplots(nrows=1,ncols=1,figsize=(8,7))
            mc = [result[histname][{'syst':syst}].project(axVar),  result[histname][{'syst':syst}].project(axVar)]
            hep.histplot(mc, stack=False, histtype='step', binwnorm=True, ax=ax, density=False, linestyle =lines, color = cols, linewidth=1,label=syst)
            fig.savefig(os_path+"{}/UL{}_{}_{}_{}.png".format(IOV,IOV,channel, histname, axVar, syst), bbox_inches='tight')
        elif (syst in availSysts) and ("nominal" in availSysts) and (syst[-4:]=="Down"):
            print("Down -- plot both variations for up")
        else:
            print("Systematic not in desired hist")
        ax_tot.set_ylabel(r'Events', loc = 'top')
        ax.set_xlabel(None) 
        # leg_tot = ax_tot.legend(loc='best', labelspacing=0.25)
        # add some labels
        fig_tot.savefig(os_path+"UL{}{}_{}_{}_allSyst.png".format(IOV,channel, histname, axVar), bbox_inches='tight')   
from hist.intervals import ratio_uncertainty
def plotDataMCwErrors(result_mc, result_data, hist_mc, hist_data, axVar, IOV, channel = "", norm = False, rax_lim=None, os_path="plots/", ylim = None, xlim = None, trim=None):
    stat_unc_up = result_mc[hist_mc][{'syst':'nominal'}].project(axVar).variances()**0.5
    stat_unc_down = result_mc[hist_mc][{'syst':'nominal'}].project(axVar).variances()**0.5
    syst_unc_up, syst_unc_down = getTotSyst(result_mc, hist_mc, axis=axVar)
    #### following opts may be unnecessary
    tot_error_opts = {
            'label': 'Stat. + Syst. Unc.',
                    'hatch': '///',
                    'edgecolor': 'black',
        'facecolor': 'none',

            'linewidth': 0
        }
    stat_error_opts = {
            'label': 'Stat. Unc.',
                    'hatch': '///',
                    'edgecolor': 'grey',
            'facecolor': 'none',
            'linewidth': 0
        }
    data_err_opts = {
            'linestyle': 'none',
            'marker': '.',
            'markersize': 10.,
            'color': 'k',
            'elinewidth': 1,
        }
    datahist = result_data[hist_data][{'syst':'nominal'}]
    mchist = result_mc[hist_mc][{'syst':'nominal'}]
    availAxes = [ax.name for ax in result_mc[hist_mc].axes]
    edges = [bin[0] for bin in result_mc[hist_mc].project(axVar).axes[0]] + [result_mc[hist_mc].project(axVar).axes[0][-1][1]]
    widths = result_mc[hist_mc].project(axVar).axes[0].widths
    if trim!=None:
        print("Trimming last bin to be size of second to last bin")
        print("Original widths ", widths)
        print("Original edges ", edges)
        edges[-1] = trim
        widths[-1] = edges[-1] - edges[-2]
        print("New widths ", widths)
        print("New edges ", edges)
    if xlim != None:
        xlim = xlim
    elif trim != None:
        xlim = trim
    else:
        xlim = edges[-1]
    fig, (ax, rax) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True)
    rax.set_xlabel(mchist.project(axVar).axes.name[0])
    ax.yaxis.get_minor_locator().set_params(numticks=999, subs=(.2, .4, .6, .8))
    ax.set_ylabel(r'Events/GeV', loc = 'top')
    ax.set_yscale('log')
    if ylim != None:
        ax.set_ylim(ylim[0], ylim[1])
    if "pt" in axVar or "m"==axVar[0]: 
        rax.set_xlabel(r'$p_{T, RECO} \, [GeV]$' )
    if "_g" in hist_mc and "m"==axVar[0]:
        rax.set_xlabel(r'$m_{SD, RECO} \, [GeV]$' )
    elif "mreco"==axVar:
        rax.set_xlabel(r'$m_{RECO} \, [GeV]$' )
    ratio = np.ones_like(result_mc[hist_mc].project(axVar).values())
    #### Fill ratio plot
    if norm:
        mcvals = mchist.project(axVar)*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        datavals = datahist.project(axVar)
        stat_unc_up = stat_unc_up*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        stat_unc_down = stat_unc_down*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        syst_unc_up = syst_unc_up*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        syst_unc_down = syst_unc_down*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
    else:
        mcvals = mchist.project(axVar)
        datavals = datahist.project(axVar)
    ratio = np.divide(datavals.values(),mcvals.values(),
                      out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                      where=mcvals.values()!= 0,)
    ratio_err = np.divide(np.sqrt(datavals.values()),mcvals.values(),
                      out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                      where=mcvals.values()!= 0,)
    #### Add MC error bars
    hep.histplot(mcvals.values(), edges,yerr = datavals.variances()**0.5, stack=False, histtype='fill',
                 ax=ax, linestyle ='-', color = 'Orange', linewidth=1, binwnorm=True,
                 label="MG+Pythia8")
    ax.stairs(values=(mcvals.values()+(stat_unc_up**2+syst_unc_up**2)**0.5)/widths, edges = edges, baseline= (mcvals.values()-(stat_unc_down**2+syst_unc_down**2)**0.5)/widths,
                fill=True,
                **tot_error_opts,
            )
    # ax.stairs(values=(mcvals.values()+stat_unc_up)/widths, edges = edges, baseline= (mcvals.values()-stat_unc_down)/widths,
    #             fill=True,
    #             **stat_error_opts,
    #         )
    print("Data values ", datavals.values(), " and variances ", datavals.variances()**0.5)
    hep.histplot(datavals.values(), edges, stack=False, histtype='errorbar',
                 ax=ax, marker =["."], color = 'Black', linewidth=1, binwnorm=True,
                 label="Data")
    ax.autoscale(axis='x', tight=True)
    #### Want to stack uncertainties
        # print("Values in bins: ", mchist.project(axVar).values(), " errors of bins ", mchist.project(axVar).variances())
    if rax_lim != None:
        rax.set_ylim(rax_lim[0], rax_lim[1])
    leg = ax.legend(loc='upper right', labelspacing=0.25)
    leg.set_visible(True)
    #### Get ratio err values and plot
    ratio_totterr_up = np.divide((mcvals.values()+(stat_unc_up**2+syst_unc_up**2)**0.5),mcvals.values(),
                      out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                      where=datavals.values()!= 0,)
    ratio_totterr_down = np.divide(mcvals.values()-(stat_unc_down**2+syst_unc_down**2)**0.5,mcvals.values(),
                      out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                      where=datavals.values()!= 0,)
    # ratio_staterr_up = np.divide(mcvals.values()+stat_unc_up,mcvals.values(),
    #                   out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
    #                   where=datavals.values()!= 0,)
    # ratio_staterr_down = np.divide(mcvals.values()-stat_unc_up,mcvals.values(),
    #                   out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
    #                   where=datavals.values()!= 0,)
    rax.stairs(values=ratio_totterr_up, edges = edges, baseline= ratio_totterr_down,
                fill=True,
                **tot_error_opts,
            )
    # rax.stairs(values=ratio_staterr_up, edges = edges, baseline= ratio_staterr_down,
    #             fill=True,
    #             **stat_error_opts,
    #         )
    
    hep.histplot(np.ones_like(ratio), edges, histtype='step',ax=rax,linestyle ="--", color = 'black', linewidth=1)
    hep.histplot(ratio, edges, histtype='errorbar',ax=rax, yerr=ratio_err, marker =["."], color = 'Black', linewidth=1)
    rax.set_ylabel(r'Data/MC', loc = 'center')
    if ("rapidity" in axVar):
        print(xlim)
        ax.set_xlim(-xlim, xlim)
        rax.set_xlim(-xlim, xlim)
        rax.set_xlabel(r"$y$")
    elif ("phi" in axVar):
        ax.set_xlim(-xlim, xlim)
        rax.set_xlim(-xlim, xlim)
        rax.set_xlabel(r"$\phi$")
    # elif "pt" in axVar:
    #     rax.set_xlim(200, 810)
    #     ax.set_xlim(200, 810)
    else:
        ax.set_xlim(0, xlim)
        rax.set_xlim(0, xlim)
    if trim:
        newticks = ax.get_xticks().tolist()
        newticks[-1] = r'$\infty$'
        print("new ticks ", newticks)
        rax.set_xticks(rax.get_xticks().tolist(),
               labels=newticks)
    if IOV == "2018": lumi = 59.83
    elif IOV == "2017": lumi = 41.48
    elif IOV == "2016": lumi = 16.8
    elif IOV == "2016APV": lumi = 19.5
    else: lumi = 138
    hep.cms.label("Preliminary "+IOV, com = 13, lumi = lumi, data = True, loc=0, ax=ax);
    ax.set_xlabel(None) 
    plt.show()
    if norm:
        fig.savefig(os_path+"ULwErrs{}{}_{}_{}_normed.png".format(IOV,channel, hist_mc, axVar), bbox_inches="tight") 
    else:
        fig.savefig(os_path+"ULwErrs{}{}_{}_{}.png".format(IOV,channel, hist_mc, axVar), bbox_inches="tight") 
        
def plotDataMC(result_mc, result_data, hist_mc, hist_data, axVar, result_herwig = None, IOV="", channel = "", rax_lim = [0.,2.0], norm = False, x_lim = None):
    if result_herwig!=None:
        herwig=True
    else:
        herwig=False
    fill_opts = {
            'edgecolor': (0,0,0,0.3),
            'alpha': 0.8}
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
    datahist = result_data[hist_data][{'syst':'nominal'}]
    mchist = result_mc[hist_mc][{'syst':'nominal'}]
    if result_herwig!=None:
        herwighist = result_herwig[hist_mc][{'syst':'nominal'}]
    # if "m"==axVar[0]:
    #     datahist = result_data[hist_data].rebin(2)
    #     edges_data = [bin[0] for bin in result_data[hist_data].project(axVar).axes[0]] + [result_mc[hist_data].project(axVar).axes[0][-1][1]]
    #     print("Data mass bins after merging pt bins: ", edges_data)
    availAxes = [ax.name for ax in result_mc[hist_mc].axes]
    edges = [bin[0] for bin in result_mc[hist_mc].project(axVar).axes[0]] + [result_mc[hist_mc].project(axVar).axes[0][-1][1]]
    print("Edges dervied from mc hist: ", edges)
    print("Edges from data hist: ", [bin[0] for bin in result_data[hist_mc].project(axVar).axes[0]] + [result_data[hist_mc].project(axVar).axes[0][-1][1]])
    if x_lim == None:
        xlim = edges[-1]
    else: xlim = x_lim
    fig, (ax, rax) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(8,7),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True)
    ax.yaxis.get_minor_locator().set_params(numticks=999, subs=(.2, .4, .6, .8))
    ax.set_ylabel(r'Events/GeV', loc = 'top')
    ax.set_yscale('log')
    if "_g" in hist_mc and "m"==axVar[0]:
        rax.set_xlabel(r'$m_{SD, RECO} \, [GeV]$' )
    ratio = np.ones_like(result_mc[hist_mc].project(axVar).values())
    #### Fill ratio plot
    if norm:
        mcvals = mchist.project(axVar)*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        datavals = datahist.project(axVar)
        # stat_unc_up = stat_unc_up*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        # stat_unc_down = stat_unc_down*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        # syst_unc_up = syst_unc_up*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        # syst_unc_down = syst_unc_down*datahist.project(axVar).integrate(axVar).value/mchist.project(axVar).integrate(axVar).value
        if herwig:
            herwigvals = herwighist.project(axVar)*datahist.project(axVar).integrate(axVar).value/herwighist.project(axVar).integrate(axVar).value
            
    else:
        mcvals = mchist.project(axVar)
        datavals = datahist.project(axVar)
        if herwig:
            herwigvals = herwighist.project(axVar)
    ratio = np.divide(mcvals.values(),datavals.values(),
                      out=np.empty(np.array(mcvals.values()).shape).fill(np.nan),
                      where=mcvals.values()!= 0,)
    if herwig:
        ratio_h = np.divide(herwigvals.values(),datavals.values(),
                      out=np.empty(np.array(herwigvals.values()).shape).fill(np.nan),
                      where=herwigvals.values()!= 0,)
    print("Integral of hist ", datahist.project(axVar).integrate(axVar).value)
    # normedHistVals = np.divide(datahist.project(axVar).values(),/datahist.project(axVar).integrate(axVar),
    #                   out=np.empty(np.array(mchist.project(axVar).values()).shape).fill(np.nan),
    #                   where=mchist.project(axVar).values()!= 0,)
    hep.histplot(datavals, stack=False, histtype='errorbar',
                 binwnorm=1, ax=ax, marker =["."], color = 'Black', linewidth=1,
                 label=channel + " Data")
    # print("Values in bins: ", datahist.project(axVar), " errors of bins ", datahist.project(axVar).variances(), )
    hep.histplot(mcvals, stack=False, histtype='step',
                 binwnorm=1, ax=ax, linestyle ='-', color = 'Blue', linewidth=1,
                 label=channel + " MC")
    if herwig:
        hep.histplot(herwigvals, stack=False, histtype='step',
                 binwnorm=1, ax=ax, linestyle ='-', color = 'Red', linewidth=1,
                 label=channel + " Herwig")
    # print("Values in bins: ", mchist.project(axVar).values(), " errors of bins ", mchist.project(axVar).variances())
    ax.autoscale(axis='x', tight=True)
    if rax_lim != None:
        rax.set_ylim(rax_lim[0], rax_lim[1])
    leg = ax.legend(loc='upper right', labelspacing=0.25)
    leg.set_visible(True)
    hep.histplot(ratio, edges, stack=False, histtype='step', ax=rax, density=False, linestyle ="-", color = 'blue', linewidth=1)
    if herwig:
        hep.histplot(ratio_h, edges, stack=False, histtype='step', ax=rax, density=False, linestyle ="-", color = 'red', linewidth=1)
    hep.histplot(np.ones_like(ratio), edges, stack=False, histtype='step',ax=rax, density=False, linestyle ="--", color = 'black', linewidth=1)
    rax.set_ylabel(r'MC/Data', loc = 'center')
    if ("rapidity" in axVar) | ("phi" in axVar):
        rax.set_xlim(-xlim, xlim)
    # elif "pt" in axVar:
    #     # rax.set_xlim(0, 2000)
    #     # rax.set_ylim(0.0,1.0)
    else:
        rax.set_xlim(0, xlim)
        ax.set_xlim(0, xlim)
    
    ax.set_xlabel(None)        
    cms = plt.text(0.25, 0.88, 'CMS $\it{Preliminary}$',
                  fontsize=22,
                  fontfamily='sans',
                  fontweight='bold',
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  transform=ax.transAxes
                 )
    lumi = plt.text(1., 1., IOV,
                fontsize=16,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
               )
    if norm:
        fig.savefig("plots/comparison/{}/UL{}{}_{}_{}_normed.png".format(channel,IOV,channel, hist_mc, axVar)) 
    else:
        fig.savefig("plots/comparison/{}/UL{}{}_{}_{}.png".format(channel,IOV,channel, hist_mc, axVar)) 
    