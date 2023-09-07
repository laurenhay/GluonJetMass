### This file houses plugins
import pandas as pd
import time
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

#if using LPC dask or running locally use 'root://cmsxrootd.fnal.gov/'
#is using coffea casa use 'root://xcache/'

import os
from distributed.diagnostics.plugin import UploadDirectory

def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


#reads in files and adds redirector, can specify year, default is all years
def handleData(jsonFile, redirector, year = '', testing = True, data = False):
    eras_data = {'2016':   '-UL2016',
                 '2016APV':'HIPM_UL2016', 
                 '2017':   'UL2017',
                 '2018':   'UL2018'
                    }
    eras_mc = {'2016':'UL16', 
               '2017':'UL17NanoAODv9', 
               '2018':'UL18NanoAODv9'
              }
    dict = {}
    qualifiers = []
    if data:
        inputs = 'JetHT_data'
        if year == '2016' or year == '2016APV' or year == '2018' or year == '2017':
            qualifiers.append(eras_data[year])
        else:
            for era in list(eras_data.values()):
                print("Era: ", era)
                qualifiers.append(era)
    else:
        if (redirector == '/mnt/data/cms'):
            jsonFile = "QCD_flat_files.json"
            inputs = 'QCD_flat'
        else:
            jsonFile = "fileset_QCD.json"
            inputs = 'QCD_binned'
        if year == '2016' or year == '2016APV' or year == '2017' or year == '2018':
            qualifier.add(eras_mc[year])
        else:
            for era in list(eras_mc.values()):
                print("Era: ", era)
                qualifiers.append(era)
    df = pd.read_json(jsonFile) 
    for key in df[inputs].keys():
        for qualifier in qualifiers:
            if qualifier in key:
                print("dataset = ", key)
                if testing:
                    dict[key] = [redirector +  df[inputs][key][0]]
                else:
                    dict[key] = [redirector + df[inputs][key][i] for i in range(len(df[inputs][key]))]
    return dict

#initiate dask client and run coffea job
from dask.distributed import Client

def runCoffeaJob(processor_inst, jsonFile, dask = False, casa = False, testing = False, year = '', data = False, winterfell = False):
    #default is to run locally
    tstart = time.time()
    executor = processor.futures_executor
    if casa:
        redirector = 'root://xcache/'
    elif winterfell:
        #### only data (not mc/sim) is on winterfell atm -- make sure using data json file and arguments
        redirector = '/mnt/data/cms'
    else:
        redirector = 'root://cmsxrootd.fnal.gov/'
    exe_args = {"schema": NanoAODSchema, 'skipbadfiles': True,}
    samples = handleData(jsonFile, redirector, year = year, testing = testing, data = data)
    client = None
    cluster = None
    if casa and dask:
        print("Running on coffea casa")
        from coffea_casa import CoffeaCasaCluster
        client = Client("tls://lauren-2emeryl-2ehay-40cern-2ech.dask.cmsaf-prod.flatiron.hollandhpc.org:8786")
        client.register_worker_plugin(UploadDirectory("/home/cms-jovyan/GluonJetMass", restart=True, update_path=True), nanny=True)
        print(client.run(os.listdir, "dask-worker-space") )
        # cluster = CoffeaCasaCluster(cores=11, memory="20 GiB", death_timeout = 60)
        # cluster.adapt(minimum=2, maximum=14)
        # client = Client(cluster)
        print(client)
        exe_args = {
            "client": client,
            'skipbadfiles':True,
            "schema": NanoAODSchema,
            "align_clusters": True,
        }
        executor = processor.dask_executor
    elif casa == False and dask:
        print("Running on LPC Condor")
        from lpcjobqueue import LPCCondorCluster
        #### figure out what replaces the tmp directory
        cluster = LPCCondorCluster(shared_temp_directory="/tmp")
        #### minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=1, maximum=10)
        client = Client(cluster)
        client.upload_file('plugins.py')
        client.upload_file('utils.py')
        client.upload_file('trijetProcessor.py') #upload additional files to the client
        client.upload_file('jetProcessors.py')
        exe_args = {
            "client": client,
            'skipbadfiles':True,
            "savemetrics": True,
            "schema": NanoAODSchema,
            "align_clusters": True,
        }
#         print("Waiting for at least one worker...")
        client.wait_for_workers(1)
        executor = processor.dask_executor
    else:
        print("Running locally")
    # samples = {'/JetHT/Run2018A-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD': ['root://xcache//store/data/Run2018A/JetHT/NANOAOD/UL2018_MiniAODv2_NanoAODv9-v2/100000/00AA9A90-57AA-D147-B4FA-54D6D8DA0D4A.root']}
    # samples = {'/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM': ['root://xcache//store/mc/RunIISummer20UL17NanoAODv9/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/270000/00DD1153-F006-3446-ABBC-7CA23A020566.root']}
    # print("Samples = ", samples, " executor = ", executor)
    result = processor.run_uproot_job(samples,
                                      "Events",
                                      processor_instance = processor_inst,
                                      executor = executor,
                                      executor_args = exe_args,
                                     )
    elapsed = time.time() - tstart
    print(result)
    print("Time taken to run over samples ", elapsed)
    return result
