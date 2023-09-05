from coffea.lumi_tools import LumiMask
import correctionlib
import awkward as ak
import numpy as np

turnOnPts_JetHT = {'2016': {'AK8PFJet40':  0.,
                            'AK8PFJet60':  0., 
                           'AK8PFJet80':  216.73271498356956, 
                           'AK8PFJet140': 288.77775439367343, 
                           'AK8PFJet200': 365.23692713436145, 
                           'AK8PFJet260': 447.2336225160111, 
                           'AK8PFJet320': 514.0097881035936, 
                           'AK8PFJet400': 632.9167920736127, 
                           'AK8PFJet450': 691.0929625368353, 
                           'AK8PFJet500': 750.4349697102186},
                  '2016APV': {'AK8PFJet40': 0., 
                            'AK8PFJet60': 0, 
                            'AK8PFJet80': 0, 
                            'AK8PFJet140': 292.4936932484294, 
                            'AK8PFJet200': 368.6063712240707, 
                            'AK8PFJet260': 449.1059028455934, 
                            'AK8PFJet320': 515.8785624883075, 
                            'AK8PFJet400': 628.9385653429788, 
                            'AK8PFJet450': 689.3079407632207, 
                            'AK8PFJet500': 749.695422049344},
                   '2017' : {'AK8PFJet40': 0.,
                            'AK8PFJet60': 0., 
                            'AK8PFJet80': 0., 
                            'AK8PFJet140': 280.13238125655073, 
                            'AK8PFJet200': 361.66204436543035, 
                            'AK8PFJet260': 448.6491079509249, 
                            'AK8PFJet320': 514.0894465691568, 
                            'AK8PFJet400': 630.6287526399205, 
                            'AK8PFJet450': 685.7915518743982, 
                            'AK8PFJet500': 741.8897640290605, 
                            'AK8PFJet550': 796.7956413028444},
                   '2018' : {'AK8PFJet15': 0.,
                            'AK8PFJet25': 0.,
                            'AK8PFJet40': 0.,
                            'AK8PFJet60': 0.,
                            'AK8PFJet80': 0., 
                            'AK8PFJet140': 281.81975103728195, 
                            'AK8PFJet200': 367.6531123322326, 
                            'AK8PFJet260': 458.9796186587356, 
                            'AK8PFJet320': 523.0660904166687, 
                            'AK8PFJet400': 638.7502963687824, 
                            'AK8PFJet450': 689.7766784490543, 
                            'AK8PFJet500': 745.1851326682686, 
                            'AK8PFJet550': 796.7749788024963}
            }


def getLumiMask(year):

    files = { '2016APV': "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
              '2016': "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
              '2017': "Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
              '2018': "Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"
            }

    mask = LumiMask(files[year])

    return mask

### function to get apply prescale weights to 
def applyPrescales(events, year, trigger = "AK8PFJet", turnOnPts = turnOnPts_JetHT, data = True):
    print("Trigger year ", year)
    if year == '2016' or year == '2016APV':
        trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500]
        pseval = correctionlib.CorrectionSet.from_file("ps_weight_JSON_2016.json")
    elif year == '2017':
        trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]  
        pseval = correctionlib.CorrectionSet.from_file("ps_weight_JSON_"+year+".json")
    elif year == '2018':
        trigThresh = [15, 25, 40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
        pseval = correctionlib.CorrectionSet.from_file("ps_weight_JSON_"+year+".json")
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