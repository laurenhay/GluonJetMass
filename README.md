# GluonJetMass
Analysis repo for gluon jet mass measurement.


### This analysis is currently optimized to run on LPC DASK in a coffea 0.7 environment.

## Setup to run on LPC:
Connect to LPC and checkout this repository.
```
kinit $USER@FNAL.GOV
ssh -XY -L localhost:####:localhost:####  $USER@cmslpc-sl7.fnal.gov
git checkout https://github.com/laurenhay/GluonJetMass.git
```
Then get a voms ticket:
```
voms-proxy-init -voms cms
```
We will then run the following commands to set up the lpc_dask and Coffea environment
```
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
./shell coffeateam/coffea-dask:0.7.22-py3.9-g7f049
jupyter notebook --no-browser --port=#### --ip 127.0.0.1
```
Now you can access your notebooks and files by opening the webpage shown as output (usually http://127.0.0.1:####)
## Running the analysis

# Proccesing events

To produce histograms for intermediate analysis and response matrices that will be inputs into the unfolding script you must run the [dijet](https://github.com/laurenhay/GluonJetMass/blob/main/python/dijetProcessor.py) and [trijet](https://github.com/laurenhay/GluonJetMass/blob/main/python/trijetProcessor.py) processors.
These can be run from the commandline or in a jupyter notebook using [dijetSelection_studies](https://github.com/laurenhay/GluonJetMass/blob/main/dijetSelection_studies.py) and [trijetSelection_studies](https://github.com/laurenhay/GluonJetMass/blob/main/trijetSelection_studies.py).
For the full datasets, running the python files from the command line is recommended to avoid notebook connection problems. The notebooks are best for prompt feedback for tests.
A summary table of the full command line arguments follows, but examples of the current command line arguments are:
# Dijet 2018
```
python3 dijetSelection_studies.py --dask --allUncertaintySources --year=2018 ----mctype=MG &> dijetMCLog2018_MG_nominal.txt
```
# Trijet 2018
```
python3 trijetSelection_studies.py --dask --allUncertaintySources --year=2018 ----mctype=MG &> trijetMCLog2018_MG_nominal.txt
```
 
| command | function |
| ------ | ------ |
| --btag [btag choice]| Trijet processor only; chooses btagging wp for leading two jets. Choices=['bbloose', 'bloose', 'bbmed', 'bmed', None], default="None" |
| --year [dataset year choice] | Year to run on. If None, run over all years (not recommended due to XROOTD errors). Choices=['2016', '2017', '2018', '2016APV', None], default="None" |
| --mctype [dataset MC choice] | Chooses MC generator for fileset. Choices=['herwig', 'pythia', 'MG'], default="MG" |
| --data | Run on data. Without this flag run on MC by default. |
| --dask | Run on dask. Without this flag run locally. |
| --testing | Run over a subset of data. Without this run over all available files for specified era. |
| --verbose | Have processor output status -- set to false if making log files. |
| --allUncertaintySources | Run processor for all uncertainty sources. Otherwise use --jetSyst flag to specify which systematics. |
[ --jetSyst [Jet uncertainty sources]| Specificy jet uncertainties to run over as a list of strings. |
| --syst [Systematic uncertainty sources] | Specify systematic uncertainty sources to run over. |
| --datasetRange | Run on a subset of available datasets. |
| --jk | Run processor to produce jackknife samples. |

# Unfolding

### WIP


## Directory

### WIP

## Other helpful tools

# tmux

[tmux](https://github.com/tmux/tmux/wiki) lets you tile window panes in a command-line environment. This in turn allows you to run, or keep an eye on, multiple programs within one terminal. When used along with ssh, it also allows us to close the node but sign back in and reattach to the same tmux session without loss of continuity.
 
| command | function |
| ------ | ------ |
| tmux new -s [name of session] | start a new session with name = [name of session] |
| tmus ls | list tmux sessions |
| tmux a -t [name of session] | attach to the tmux session with name = [name of session] |
| tmux a # | attach to the last created session |
|tmux kill-session -t [name of session]| kill tmux session with name = [name of session] |
 
The following commands are used inside tmux.
 
| commands inside tmux | function |
| ------ | ------ |
| ctrl+b, d | detach/exit (without killing) from tmux session |
| ctrl+b, " | split pane horizontally |
| ctrl+b, % | split pane vertically |
|ctrl+b, [arrow key]| to move from one pane to the other|
|ctrl+b, x | kill the current pane|
| ctrl+b, [ | to scroll up linewise, press 'q' to exit this mode |


## Setup on CoffeaCasa


Details on setting up your gitenvironment on coffea casa here: https://github.com/b2g-nano/TTbarAllHadUproot/files/9181161/MovingToCoffeaCasa.pdf (slides by AC Williams)
If running this analysis locally on coffea casa, make sure `runCoffeaJob()` is given arguments `dask = False, casa = True`
If you're running on the CoffeaCasaCluster you should give runCoffeaJob() the arguments `dask = True, casa = True` and make sure to shut down the automatically generated cluster.


### FUTURE ADDITIONS: Instructions on accessing and running on winterfell

Testing push from winterfell