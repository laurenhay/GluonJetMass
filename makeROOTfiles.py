#### Convert coffea output into root TH2's to feed into TUnfold
#### LMH

import pickle
import uproot
import numpy as np

import argparse

fname = "coffeaOutput/trijetHistsTest_QCDsim_pt200.0_eta2.4_bbloose.pkl"
with open(fname, "rb") as f:
    result = pickle.load( f )

rootfile = uproot.recreate('trijetHistsQCDsim.root')
result=result[0]
# integrate and sum over axes
rootfile['jet_pt_mass_reco_u'] = result['jet_pt_mass_reco_u'][{'dataset':sum}]
rootfile['jet_pt_mass_reco_g'] = result['jet_pt_mass_reco_g'][{'dataset':sum}]
rootfile['jet_pt_mass_gen_u'] = result['jet_pt_mass_gen_u'][{'dataset':sum}]
rootfile['jet_pt_mass_gen_g'] = result['jet_pt_mass_gen_g'][{'dataset':sum}]
rootfile['mreco_mgen_u'] = result['response_matrix_u'].project("mreco", "mgen")
rootfile['mreco_mgen_g'] = result['response_matrix_g'].project("mreco","mgen")
rootfile['ptreco_ptgen_u'] = result['response_matrix_u'].project("ptreco", "ptgen")
rootfile['ptreco_ptgen_g'] = result['response_matrix_g'].project("ptreco","ptgen")
rootfile['fakes_ptreco_mreco'] = result['fakes'].project("ptreco","mreco")
rootfile['misses_ptgen_mgen'] = result['misses'].project("ptgen", "mgen")
response_matrix_u_values, ptreco_edges, mreco_edges, ptgen_edges, mgen_edges = result['response_matrix_u'].project("ptreco", "mreco", "ptgen", "mgen").to_numpy()
response_matrix_g_values = result['response_matrix_g'].project("ptreco", "mreco", "ptgen", "mgen").values()
print(response_matrix_g_values.shape)
ptreco_centers = (ptreco_edges[:-1]+ptreco_edges[1:])/2
print(ptreco_centers.shape)
mreco_centers = (mreco_edges[:-1]+mreco_edges[1:])/2
ptgen_centers = (ptgen_edges[:-1]+ptgen_edges[1:])/2
mgen_centers = (mgen_edges[:-1]+mgen_edges[1:])/2
rootfile['response'] = {'groomed':response_matrix_g_values[np.newaxis],
                       'ungroomed':response_matrix_u_values[np.newaxis]}
rootfile['centers'] = {'ptreco':ptreco_centers[np.newaxis], 'mreco':mreco_centers[np.newaxis], 
                      'ptgen':ptgen_centers[np.newaxis], 'mgen':mgen_centers[np.newaxis]}
rootfile.close()
