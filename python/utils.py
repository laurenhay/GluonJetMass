from coffea.lumi_tools import LumiMask
import correctionlib
import awkward as ak
import numpy as np

turnOnPts_JetHT = {'2016': {'AK8PFJet40':0.,
                            'AK8PFJet60': 100., 
                            'AK8PFJet80': 200., 
                            'AK8PFJet140': 260., 
                            'AK8PFJet200': 350., 
                            'AK8PFJet260': 420., 
                            'AK8PFJet320': 530., 
                            'AK8PFJet400': 620., 
                            'AK8PFJet450': 690., 
                            'AK8PFJet500': 740.},
                   '2016APV': {'AK8PFJet40':0., 
                               'AK8PFJet60': 0, 
                               'AK8PFJet80': 220., 
                               'AK8PFJet140': 270., 
                               'AK8PFJet200': 360., 
                               'AK8PFJet260': 440., 
                               'AK8PFJet320': 530., 
                               'AK8PFJet400': 630., 
                               'AK8PFJet450': 690., 
                               'AK8PFJet500': 750.},
                   '2017' : {'AK8PFJet40':0.,
                             'AK8PFJet60': 0., 
                             'AK8PFJet80': 200., 
                             'AK8PFJet140': 260.,                 
                             'AK8PFJet200': 340., 
                             'AK8PFJet260': 420., 
                             'AK8PFJet320': 530., 
                             'AK8PFJet400': 620., 
                             'AK8PFJet450': 680., 
                             'AK8PFJet500': 745., 
                             'AK8PFJet550': 807.},
                   '2018' : {'AK8PFJet15': 0.,
                             'AK8PFJet25': 0.,
                             'AK8PFJet40': 0.,
                             'AK8PFJet60': 0.,
                             'AK8PFJet80': 0., 
                             'AK8PFJet140': 260., 
                             'AK8PFJet200': 360., 
                             'AK8PFJet260': 440., 
                             'AK8PFJet320': 550., 
                             'AK8PFJet400': 630., 
                             'AK8PFJet450': 700., 
                             'AK8PFJet500': 750., 
                             'AK8PFJet550': 810.}
            }

#### LATEST VALUES TO SWITCH TO
# JetHT2016_L1 = {'AK8PFJet40':0.,
#                  'AK8PFJet60': 140., 
#                  'AK8PFJet80': 210., 
#                  'AK8PFJet140': 290., 
#                  'AK8PFJet200': 380., 
#                  'AK8PFJet260': 450., 
#                  'AK8PFJet320': 550., 
#                  'AK8PFJet400': 640., 
#                  'AK8PFJet450': 690., 
#                  'AK8PFJet500': 820.}
# JetHT2016APV_L1 = {'AK8PFJet40':0., 
#                      'AK8PFJet60': 140, 
#                      'AK8PFJet80': 210., 
#                      'AK8PFJet140': 290., 
#                      'AK8PFJet200': 380., 
#                      'AK8PFJet260': 450., 
#                      'AK8PFJet320': 550., 
#                      'AK8PFJet400': 640., 
#                      'AK8PFJet450': 730., 
#                      'AK8PFJet500': 820.}
# JetHT2017_L1 = {'AK8PFJet40':0.,
#                  'AK8PFJet60': 0., 
#                  'AK8PFJet80': 160., 
#                  'AK8PFJet140': 270., 
#                  'AK8PFJet200': 310., 
#                  'AK8PFJet260': 450., 
#                  'AK8PFJet320': 560., 
#                  'AK8PFJet400': 640., 
#                  'AK8PFJet450': 700., 
#                  'AK8PFJet500': 760., 
#                  'AK8PFJet550': 810.}
# JetHT2018_L1 = {'AK8PFJet15': 0.,
#                  'AK8PFJet25': 0.,
#                  'AK8PFJet40': 0.,
#                  'AK8PFJet60': 0.,
#                  'AK8PFJet80': 160., 
#                  'AK8PFJet140': 270., 
#                  'AK8PFJet200': 390., 
#                  'AK8PFJet260': 470., 
#                  'AK8PFJet320': 570., 
#                  'AK8PFJet400': 650., 
#                  'AK8PFJet450': 710., 
#                  'AK8PFJet500': 760., 
#                  'AK8PFJet550': 820.}
# in 
xsdb= { 'QCD_Pt_170to300_TuneCP5_13TeV_pythia8' : 104000.0,
'QCD_Pt_300to470_TuneCP5_13TeV_pythia8' : 6806.0,
'QCD_Pt_470to600_TuneCP5_13TeV_pythia8' : 552.0,
'QCD_Pt_600to800_TuneCP5_13TeV_pythia8' : 154.6,
'QCD_Pt_800to1000_TuneCP5_13TeV_pythia8' : 26.15,
'QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8' : 0.03567,
'QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8' : 0.6419,
'QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8' : 0.0877,
'QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8' : 0.005241,
'QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8' : 0.0001346,
'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 11370.0,
'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 883.5,
'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 259.6,
'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 23.63,
'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':  4.943,
'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 0.8013,
'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 0.06815,
'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7': 0.01245,
 }

# in fb^-1 taken from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable
lumi = {'2018' : 59.74,
        '2017': 41.48,
        # '2016': 36.33 ####combined
        '2016APV':19.5,
        "2016":16.8
       }


#### Missing 2016APV XS's
num_gen_herwig = { "2016APV": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 3005498,
                               'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 1967563,
                               'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 1000485,
                               'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 496454,
                               'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':  488128,
                               'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 506141,
                               'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 492965,
                               'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7': 496469,
                              },
                  "2016": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 2942671,
                               'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 2006720,
                               'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 1006059,
                               'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 511149,
                               'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':  499379,
                               'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 493891,
                               'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 492975,
                               'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7':  488875,
                              },
                  "2017": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 2889515,
                               'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 1992969,
                               'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 1003410,
                               'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 516397,
                               'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':  501738,
                               'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 499775,
                               'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 507301,
                               'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7': 497985,
                              },
                  "2018": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 2934193,
                               'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 1990195,
                               'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 999343,
                               'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 497722,
                               'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7':  495400,
                               'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 506420,
                               'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 507791,
                               'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7': 492537,
                              },
}
num_gen = { '2016APV': {'QCD_Pt_170to300_TuneCP5_13TeV_pythia8' : 27885000,
'QCD_Pt_300to470_TuneCP5_13TeV_pythia8' : 54028000,
'QCD_Pt_470to600_TuneCP5_13TeV_pythia8' : 50782000,
'QCD_Pt_600to800_TuneCP5_13TeV_pythia8' : 61904000,
'QCD_Pt_800to1000_TuneCP5_13TeV_pythia8' : 35459000,
'QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8' : 19077000,
'QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8' : 11000000,
'QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8' : 5262000,
'QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8' : 2999000,
'QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8' : 1000000},
    '2016': {'QCD_Pt_170to300_TuneCP5_13TeV_pythia8' : 29758000,
'QCD_Pt_300to470_TuneCP5_13TeV_pythia8' : 55264000,
'QCD_Pt_470to600_TuneCP5_13TeV_pythia8' : 52408000,
'QCD_Pt_600to800_TuneCP5_13TeV_pythia8' : 64584000,
'QCD_Pt_800to1000_TuneCP5_13TeV_pythia8' : 37698000,
'QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8' : 19892000,
'QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8' : 10722000,
'QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8' : 5236000,
'QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8' : 2848000,
'QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8' : 996000},
           '2017': {'QCD_Pt_170to300_TuneCP5_13TeV_pythia8' : 29491000,
'QCD_Pt_300to470_TuneCP5_13TeV_pythia8' : 55358000,
'QCD_Pt_470to600_TuneCP5_13TeV_pythia8' : 50475000,
'QCD_Pt_600to800_TuneCP5_13TeV_pythia8' : 66419000,
'QCD_Pt_800to1000_TuneCP5_13TeV_pythia8' : 36890000,
'QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8' : 19461000,
'QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8' : 10994000,
'QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8' : 5168000,
'QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8' : 2997000,
'QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8' : 1000000},
           '2018': {'QCD_Pt_170to300_TuneCP5_13TeV_pythia8' : 29478000,
'QCD_Pt_300to470_TuneCP5_13TeV_pythia8' : 57868000,
'QCD_Pt_470to600_TuneCP5_13TeV_pythia8' : 52448000,
'QCD_Pt_600to800_TuneCP5_13TeV_pythia8' : 66914000,
'QCD_Pt_800to1000_TuneCP5_13TeV_pythia8' : 36830000,
'QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8' : 19664000,
'QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8' : 10982000,
'QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8' : 5491000,
'QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8' : 2931000,
'QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8' : 1000000}
}
        

def getRapidity(p4):
    return 0.5 * np.log(( p4.energy + p4.pz ) / ( p4.energy - p4.pz ))
    
def getXSweight(dataset, IOV):
    print("Dataset: ", dataset)
    for year in np.array(list(lumi.keys())):
        if year in IOV:
            lum = lumi[year]
            print("Lumi ", lum, " for year ", year)
            for process in np.array(list(xsdb.keys())):
                if process in dataset:
                    xs = xsdb[process]
                    if 'herwig' in process:
                        print("Number of gen events for ", year, " ", process, ": ", num_gen_herwig[year][process])
                        weight = xs * lum * 1000 / num_gen_herwig[year][process]
                    else:
                        print("Number of gen events for ", year, " ", process, ": ", num_gen[year][process])
                        weight = xs * lum * 1000 / num_gen[year][process]
                    return weight


def getLumiMask(year):

    files = { '2016APV': "correctionFiles/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
              '2016': "correctionFiles/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
              '2017': "correctionFiles/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
              '2018': "correctionFiles/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"
            }

    mask = LumiMask(files[year])

    return mask

def get_gen_sd_mass_jet( jet, subjets):
    combs = ak.cartesian( (jet, subjets), axis=1 )
    dr_jet_subjets = combs['0'].delta_r(combs['1'])
    combs = combs[dr_jet_subjets < 0.8]
    total = combs['1'].sum(axis=1)
    return total 

def get_dphi( jet0, jet1 ):
    '''
    Find dphi between two jets, returning none when the event does not have at least two jets
    '''
    combs = ak.cartesian( (jet0, jet1), axis=1 )
    dphi = np.abs(combs['0'].delta_phi(combs['1']))
    return ak.firsts(dphi)

def update(events, collections):
    # https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/hbbprocessor.py
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
#     logger.debug('update:%s:%s', time.time(), collections)
    
    for name, value in collections.items():
        out = ak.with_field(out, value, name)

    return out

### function to get apply prescale weights to 
def applyPrescales(events, year, trigger = "AK8PFJet", turnOnPts = turnOnPts_JetHT, data = True):
    print("Trigger year ", year)
    if year == '2016' or year == '2016APV':
        trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500]
        pseval = correctionlib.CorrectionSet.from_file("correctionFiles/ps_weight_JSON_2016.json")
    elif year == '2017':
        trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]  
        pseval = correctionlib.CorrectionSet.from_file("correctionFiles/ps_weight_JSON_"+year+".json")
    elif year == '2018':
        trigThresh = [15, 25, 40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
        pseval = correctionlib.CorrectionSet.from_file("correctionFiles/ps_weight_JSON_"+year+".json")
    turnOnPts = np.array(list(turnOnPts[year].values()))
    HLT_paths = [trigger + str(i) for i in trigThresh]
    events_mask = np.full(len(events), False)
    weights = np.ones(len(events))

    #### allRuns_AK8HLT.csv is the result csv of running 'brilcalc trg --prescale --hltpath "HLT_AK8PFJet*" --output-style                 csv' and is used to create the ps_weight_JSON files
    #### lumimask and requirement of one jet is already applied in jet processor

    for i in np.arange(len(HLT_paths))[::-1]:
        path = HLT_paths[i]
#             print("Index i: ", i, " for path: ", path)
        if path in events.HLT.fields:
            pt0 = events.FatJet[:,0].pt
            psweights = pseval['prescaleWeight'].evaluate(ak.to_numpy(events.run), path,
                                                          ak.to_numpy(ak.values_astype(events.luminosityBlock, np.float32)))
            #### here we will use correctionlib to assign weights
            if (i == (len(HLT_paths) - 1)):
                events_cut = events[((pt0 > turnOnPts[i]) & events.HLT[path])]
                events_mask = np.where(((pt0 > turnOnPts[i]) & events.HLT[path]), True, events_mask)
#                 print("Number of ", path, "'s trues: ", sum(((pt0 > turnOnPts[i]) & events.HLT[path])), " number of total trues ", sum(events_mask))
                weights = np.where(((pt0 > turnOnPts[i]) & events.HLT[path]), psweights, weights)
            else:
                events_cut = events[((pt0 > turnOnPts[i]) & (pt0 <= turnOnPts[i+1]) & events.HLT[path])]
                events_mask = np.where(((pt0 > turnOnPts[i]) & (pt0 <= turnOnPts[i+1]) & events.HLT[path]), True, events_mask)
#                 print("Number of ", path, "'s path's trues: ", sum(((pt0 > turnOnPts[i]) & (pt0 <= turnOnPts[i+1]) & events.HLT[path])), " number of total trues ", sum(events_mask))
                weights = np.where(((pt0 > turnOnPts[i]) & (pt0 <= turnOnPts[i+1])), psweights, weights)
    return events_mask, weights