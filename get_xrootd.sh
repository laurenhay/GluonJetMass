#!/bin/bash

dataset=/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/RunIISummer20UL16NanoAOD-FlatPU0to70_106X_mcRun2_asymptotic_v13-v1/NANOAODSIM

dataset=$1

if [[ "$dataset" != *"/"* ]]
then
	echo "usage: ./get_xrood.sh [DAS dataset or file]"
	echo "must be in CMSSW environment"
	exit 1
fi

sites_file=dataset_sites_list.txt

export SSL_CERT_DIR='/etc/pki/tls/certs:/etc/grid-security/certificates'

if [[ "$dataset" == *".root"* ]]
then
	dasgoclient -query="site file=$dataset" > $sites_file
else
	:
	#dasgoclient -query="site dataset=$dataset" > $sites_file
fi

echo "" > xrootd_files_list.txt
echo "" > xrootd_list.txt

echo ""
cat $sites_file
echo ""

for ds in `cat $sites_file`
do

	echo $ds

	if [[ "$ds" == *"T1"* ]]
	then
		xrootd=root://cmsxrootd.fnal.gov/
	else 
		xrootd=`python3 GetSiteInfo.py $ds | grep XROOTD | awk '{print $2}'`
	fi


	echo $xrootd
	echo $xrootd >> xrootd_list.txt

		files=`dasgoclient -query="file site=$ds dataset=$dataset"`

	for file in $files
	do
		if grep -q "$file" <<< `cat xrootd_files_list.txt`
		then
			:
		else
		
			echo ""
			echo "$xrootd/$file"
			echo "$xrootd/$file" >> xrootd_files_list.txt
			echo ""
		fi
	done

done


