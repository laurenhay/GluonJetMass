#include <iostream>
#include <fstream>
#include <TFile.h>
#include <TH2D.h>
#include <TUnfoldBinning.h>
#include "TUnfoldDensity.h"
#include <TTreeReader.h>
#include <TTreeReaderValue.h>



// this file imports root histograms made with coffea and created a TUnfold binning object

using namespace std;

void testUnfold()
{//turn on errors in histograms?
  TH1::SetDefaultSumw2();

  //open MC root file from coffea/uproot and get hists
  TFile *f = TFile::Open("fakeHists_arr.root");
  TH2D *ptgen_mgen_g = f->Get<TH2D>("ptgen_mgen_g");
  TH2D *ptgen_mgen_u = f->Get<TH2D>("ptgen_mgen_u");
  TH2D *fakes_ptreco_mreco = f->Get<TH2D>("fakes_ptreco_mreco");
  TH2D *misses_ptgen_mgen = f->Get<TH2D>("misses_ptgen_mgen");
  
  //open data root file and get hists
  TFile *f_reco = TFile::Open("fakeHists_arr.root");
  TH2D *ptreco_mreco_u = f_reco->Get<TH2D>("jet_ptreco_mreco_u");
  TH2D *ptreco_mreco_g = f_reco->Get<TH2D>("jet_ptreco_mreco_g");
  TH2D *ptreco_mreco_up = f_reco->Get<TH2D>("jet_ptreco_mreco_up");
  TH2D *ptreco_mreco_dwn = f_reco->Get<TH2D>("jet_ptreco_mreco_down");

  
  // Create a TTreeReader for the tree by passing the TTree's name and the TDirectory / TFile it is in.
  TTreeReader matrixReader("response", f);
  TTreeReader binReader("centers", f);
  TTreeReader recoReader("reco", f_reco);
  TTreeReader genReader("gen", f);
  // The branch "reco_gen_groomed" contains doubles; access them as response_matrix_g
  TTreeReaderArray<Double_t> response_matrix_g(matrixReader, "groomed");
  // The branch "reco_gen_ungroomed" contains doubles; access them as response_matrix_u
  TTreeReaderArray<Double_t> response_matrix_u(matrixReader, "ungroomed");
  TTreeReaderArray<Double_t> response_matrix_up(matrixReader, "errorUp");
  TTreeReaderArray<Double_t> response_matrix_dwn(matrixReader, "errorDown");
  // Make bin arrays
  TTreeReaderArray<Double_t> ptreco_center(binReader, "ptreco");
  TTreeReaderArray<Double_t> mreco_center(binReader, "mreco");
  TTreeReaderArray<Double_t> ptgen_center(binReader, "ptgen");
  TTreeReaderArray<Double_t> mgen_center(binReader, "mgen");
  // make reco arrays
  TTreeReaderArray<Double_t> ptreco_mreco_g_val(recoReader, "groomed");
  TTreeReaderArray<Double_t> ptreco_mreco_u_val(recoReader, "ungroomed");
  TTreeReaderArray<Double_t> fakes_val(recoReader, "fakes");
  // make gen arrays
  TTreeReaderArray<Double_t> ptgen_mgen_g_val(genReader, "groomed");
  TTreeReaderArray<Double_t> ptgen_mgen_u_val(genReader, "ungroomed");
  TTreeReaderArray<Double_t> misses_val(genReader, "misses");
    //check content by drawing hists
  TCanvas *c1 = new TCanvas("c1","FAKES AND MISSES",1200,600);
  c1->Divide(2,1);
  c1->cd(1);
  misses_ptgen_mgen->Draw("colz");
  c1->cd(2);
  c1->SetLogz();
  fakes_ptreco_mreco->Draw("colz");

  //define mass binnings
  vector<double> binsMassCoarseVec{0.,1.,5.,10.,20.,40.,60.,80.,100.,150.,200.,250.,1000.}; //gen
  int nBinMassCoarse=binsMassCoarseVec.size()-1;
  vector<double> binsVector[2];
  for(int i=0;i<=nBinMassCoarse;i++) {
    double x0=binsMassCoarseVec[i];
    for(int k=0;k<2;k++) binsVector[k].push_back(x0);
    if(i<nBinMassCoarse) {
      // add extra bins
      double x1=binsMassCoarseVec[i+1];
      double x2=0.5*(x0+x1);
      binsVector[1].push_back(x2);
      cout << "Mass fine bins : " << x0 << " " << x2 << endl;
    }                                                                                                                                                       
  }
  vector<double> const &binsMassFine=binsVector[1]; //reco
  int nBinMassFine = binsMassFine.size()-1;
  vector<double> const &binsMassCoarse=binsVector[0]; //reco
  //define pt binning (same for gen and reco)
  vector<double> binsPt{200.,280.,360.,450.,520.,630.,690.,750.,800.,1300.,13000.};
  int nBinsPt = binsPt.size()-1;
  vector<double> const &binsPtRef=binsPt;
  
  //get reconstructed mass and pt bins
  int NBin_mass_fine = ptgen_mgen_u->GetYaxis()->GetNbins();
  int NBin_pt_fine = ptgen_mgen_u->GetXaxis()->GetNbins();
  cout << "Number of gen mass bins: "<< NBin_mass_fine << " and number of gen pt bins: " << NBin_pt_fine << endl;
  //get generated mass and pt bins
  int NBin_mass_coarse = ptreco_mreco_u->GetYaxis()->GetNbins();
  int NBin_pt_coarse = ptreco_mreco_u->GetXaxis()->GetNbins();
  cout << "Number of reco mass bins: "<< NBin_mass_coarse << " and number of reco pt bins: " << NBin_pt_coarse << endl;
  
  //make TUnfold binning axes
  TUnfoldBinning *detectorBinning = new TUnfoldBinning("detector");
  //  TUnfoldBinning *missBinning = detectorBinning->AddBinning("missesBin", 1);
  TUnfoldBinning *recoBinning = detectorBinning->AddBinning("reco");
  recoBinning->AddAxis("m_{RECO}", nBinMassFine, binsMassFine.data(), false, false); //mreco
  //  recoBinning->AddAxis(*ptreco_mreco_u->GetYaxis(), false, false); //mreco
  recoBinning->AddAxis("pt_{RECO}", nBinsPt, binsPt.data(), false, false); //ptreco

  
  TUnfoldBinning *generatorBinning = new TUnfoldBinning("generator");
  //  TUnfoldBinning *fakeBinning=generatorBinning->AddBinning("fakesBin", 1);
  TUnfoldBinning *genBinning = generatorBinning->AddBinning("gen");
  //  genBinning->AddAxis(*ptgen_mgen_u->GetYaxis(), false, false); //mgen
  //  genBinning->AddAxis(*ptgen_mgen_u->GetXaxis(), false, false); //ptgen
  genBinning->AddAxis("m_{GEN}", nBinMassCoarse, binsMassCoarse.data(), false, false); //mgen
  genBinning->AddAxis("pt_{GEN}", nBinsPt, binsPt.data(), false, false); //ptgen  
  cout << "Test binning for ptreco = 250, mreco = 500 " <<recoBinning->GetGlobalBinNumber(250,500) << endl;
  
  // create histogram of migrations and gen hists
  TH2D *histMCGenRec=TUnfoldBinning::CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec");
  TH2D *histMCGenRecUp=TUnfoldBinning::CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec+1#sigma");
  TH2D *histMCGenRecDwn=TUnfoldBinning::CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec-1#sigma");
  // create data histograms
  TH1 *histDataReco=recoBinning->CreateHistogram("histDataReco");  //data values in reco binnning --> input to unfolding

  //create MC hists for comparison
  TH1 *histMCReco=recoBinning->CreateHistogram("histMCReco"); //gen values in reco binning --> htruef in Sal's example
  TH1 *histMCTruth=genBinning->CreateHistogram("histMCTruth");  //gen values in gen binning
  
  // Loop through reco and gen bins of MC input and fill hist of migrations
  while(matrixReader.Next() && binReader.Next() && recoReader.Next() && genReader.Next()) {
    cout << "Type of groomed: " << typeid(response_matrix_g).name() << " and size: " << response_matrix_u.GetSize() << endl;
    cout << "Size of ptreco bins: " << ptreco_center.GetSize()<< endl;
    cout << "Size of mreco bins: " << mreco_center.GetSize()<< endl;
    cout << "Size of mgen bins: " << mgen_center.GetSize()<< endl;
    Int_t glob_recobin;
    Int_t recoBin;
    Int_t glob_genbin;
    Int_t genBin;
    // reco loop: i is ptreco, j is mreco
    for(int i=0; i<ptreco_center.GetSize(); i++){
      for(int j=0; j<mreco_center.GetSize(); j++){
	glob_recobin=i*mreco_center.GetSize()+j;
	/* cout<<"has centers mreco " << mreco_center[j] << " and ptreco " << ptreco_center[i] << endl; */
	recoBin=recoBinning->GetGlobalBinNumber(mreco_center[j],ptreco_center[i]);
	//only fill fakes in fake genBin
	//	    Double_t fake_weight = fakes_ptreco_mreco->GetBinContent(i,j);
	// cout << "Fake weight " << fake_weight << " for i == " << i << " and j == " << j << endl;
	//Int_t fakeBin=fakeBinning->GetStartBin();
	//histMCGenRec->SetBinContent(fakeBin,recoBin,fake_weight);
	//fill data hist
	Double_t data_weight_u=ptreco_mreco_u_val[glob_recobin];
	histDataReco->Fill(recoBin, data_weight_u);
	cout << "Reco weight " << data_weight_u << " for glob reco bin "<< glob_recobin << "and matrix reco bin " << recoBin << endl;
	// gen loop: k is ptgen, l is mgen
	for(int k=0; k<ptgen_center.GetSize(); k++){
	  for(int l=0; l<mgen_center.GetSize(); l++){
	    glob_genbin=k*mgen_center.GetSize()+l;
	    genBin=genBinning->GetGlobalBinNumber(mgen_center[l],ptgen_center[k]);
	    /* cout<< genBin << "has centers mgen " << mgen_center[l] << " and ptgen " << ptgen_center[k] << endl; */
	    //only fill misses in miss recoBin
	    //Double_t miss_weight = misses_ptgen_mgen->GetBinContent(k,l);
	    /* cout << "Miss weight " << miss_weight << " for k == " << k << " and k == " << k << endl; */
	    //Int_t missBin=missBinning->GetStartBin();
	    //histMCGenRec->SetBinContent(genBin,missBin,miss_weight);
	    //fill MC truth for closure test
	    //ONLY FILL ONCE INSIDE i,j LOOP
	    if(i==0 && j==0){
	      Double_t truth_weight_u = ptgen_mgen_u_val[glob_genbin];
	      histMCTruth->Fill(genBin, truth_weight_u);
	      cout << "Truth weight " << truth_weight_u << " for glob gen bin "<< glob_genbin << "and matrix gen bin " << genBin << endl;
	      //fill MC reco for comparison
	      Int_t recoBin_genObj=recoBinning->GetGlobalBinNumber(mgen_center[l],ptgen_center[k]);
	      cout << "Truth weight " << truth_weight_u << " for k == " << k << " and l == " << l << "and reco bin " << recoBin_genObj << endl;
	      histMCReco->Fill(recoBin_genObj, truth_weight_u);	    	    }
	    Int_t glob_bin = glob_recobin*(mgen_center.GetSize()*ptgen_center.GetSize())+glob_genbin;
	    /* cout<<"Global bin " << glob_bin << " for reco bin " << glob_recobin << " and gen bin " << glob_genbin << endl; */
	    //		    cout <<"TUnfold gen bin "<< genBin << " and reco bin " << recoBin<< "for value" << response_matrix_u[glob_bin] <<endl;
	    Double_t resp_weight_u = response_matrix_u[glob_bin];
	    histMCGenRec->Fill(genBin,recoBin,resp_weight_u);
	    //fill matrices for systematics
	    Double_t resp_weight_up = response_matrix_up[glob_bin];
	    histMCGenRecUp->Fill(genBin,recoBin,resp_weight_up);
	    Double_t resp_weight_dwn = response_matrix_dwn[glob_bin];
	    histMCGenRecDwn->Fill(genBin,recoBin,resp_weight_dwn);
	  }}}}
    cout << response_matrix_u[849] <<endl;              
  }
  
  histMCGenRec->Print("base");
  histMCReco->Print("base");
  histMCTruth->Print("base");
    
  // check that response matrix has been filled properly
  TH1 *histMCReco_M=histMCGenRec->ProjectionY("MCReco", 1, -2);
  TH1 *histMCTruth_M=histMCGenRec->ProjectionX("MCTruth", 1, -2);

  //check Truth contents
  int nBinsTruth = histMCTruth_M->GetXaxis()->GetNbins();
  for(int bin = 0; bin < nBinsTruth; bin++){
    cout << "Matrix truth value: " << histMCTruth_M->GetBinContent(bin) << " for matrix bin : "<< bin << endl;
  }
  int nBinsReco = histMCReco_M->GetXaxis()->GetNbins();
  for(int bin = 0; bin < nBinsReco; bin++){
    cout << "Matrix reco value: " << histMCReco_M->GetBinContent(bin) << " for matrix bin : "<< bin << endl;
  }
  
  TCanvas *c2 = new TCanvas("c2","Plot Data and MC input mass binned by pt",1200,400);
  c2->Divide(2,1);
  c2->cd(1);
  histMCTruth_M->SetMarkerStyle(24);
  histMCTruth_M->SetLineColor(kRed);
  histMCTruth_M->SetMarkerColor(kRed);
  histMCTruth_M->Draw();
  histMCTruth->SetMarkerStyle(21);
  histMCTruth->SetLineColor(kBlue);
  histMCTruth->SetMarkerColor(kBlue);
  histMCTruth->Draw("SAME");

  /* histMCTruth_M->SetMarkerStyle(21); */
  /* histMCTruth_M->SetLineColor(kRed); */
  /* histMCTruth_M->SetMarkerColor(kRed); */
  /* histMCTruth_M->Draw(); */
  /* histMCReco_M->SetMarkerStyle(24); */
  /* histMCReco_M->SetLineColor(kBlue); */
  /* histMCReco_M->SetMarkerColor(kBlue); */
  /* histMCReco_M->Draw("SAME"); */

  
  auto leg1 = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg1->AddEntry(histMCTruth_M, "MC Gen from M", "p");
  leg1->AddEntry(histMCTruth, "MC Grn", "p");
  leg1->Draw();
  c2->cd(2);
  histMCReco_M->SetMarkerStyle(24);
  histMCReco_M->SetLineColor(kRed);
  histMCReco_M->SetMarkerColor(kRed);
  histMCReco_M->Draw();
  histMCReco->SetMarkerStyle(21);
  histMCReco->SetLineColor(kBlue);
  histMCReco->SetMarkerColor(kBlue);
  histMCReco->Draw("SAME");

  /* histMCTruth->SetMarkerStyle(24); */
  /* histMCTruth->SetLineColor(kRed); */
  /* histMCTruth->SetMarkerColor(kRed); */
  /* histMCTruth->Draw(); */
  /* histMCReco->SetMarkerStyle(21); */
  /* histMCReco->SetLineColor(kBlue); */
  /* histMCReco->SetMarkerColor(kBlue); */
  /* histMCReco->Draw("SAME"); */

  auto leg2 = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg2->AddEntry(histMCReco_M, "MC Reco from M", "p");
  leg2->AddEntry(histMCReco, "MC Reco", "p");
  leg2->Draw();

  TCanvas *c3 = new TCanvas("c3","Plot full response",1200,800);
  c3->Divide(3,1);
  c3->cd(1);
  c3->SetLogz();
  histMCGenRec->Draw("colz");
  c3->cd(2);
  c3->SetLogz();
  histMCGenRecUp->Draw("colz");
  c3->cd(3);
  c3->SetLogz();
  histMCGenRecDwn->Draw("colz");
  
  //=======================================================
  // open file to save histograms and binning schemes
  
  TFile *outputFile=new TFile("testUnfold_histograms.root","recreate");
  
  detectorBinning->Write();
  generatorBinning->Write();

  histMCReco->Write();
  histMCGenRec->Write();
  histMCTruth->Write();
  
  detectorBinning->PrintStream(cout);
  generatorBinning->PrintStream(cout);
  
  //=======================================================
  // Do unfolding
  
  //Unfolding analysis plots
  TH1 *histDataUnfolded=0;
  TProfile *prof_pull_TikhonovLCurve=0;
  TH1 *hist_coverage_TikhonovLCurve=0;
  //Auxillary output for Tikhonov, minimum L-Curve curvature
  TGraph *graph_LCurve;
  TSpline *spline_logTauX;
  TSpline *spline_logTauY;
  double tauBest=-1.,DF_TikhonovLCurve=-1.;
  
  double biasScale=1.0;
  int NPOINT_TikhonovLCurve=50;
  
  // Ashley also used kHistMapOutputHoriz and kEConstraintNone
  // She also used kRegModeNone instead of keRegModeSize
  //Try Sal's setting: kRegModeCurvatur, kEConstraintArea, kDensityModeBinWidth
  TUnfoldDensity tunfold(histMCGenRec,TUnfoldDensity::kHistMapOutputHoriz,
				       TUnfoldDensity::kRegModeCurvature,TUnfoldDensity::kEConstraintArea,
			 TUnfoldDensity::kDensityModeBinWidth); //, generatorBinning, detectorBinning);
    // set up matrix of migrations
  /* TH1 *input; */
  /* input = histMCReco_M; */
  /* TH2D *inputEmatrix= */
  /*   detectorBinning->CreateErrorMatrixHistogram("input_covar",true); */
  /* for(int i=1;i<=inputEmatrix->GetNbinsX();i++) { */
  /*   Double_t e=input->GetBinError(i); */
  /*   cout << "Error in MC: " << e << endl; */
  /*   inputEmatrix->SetBinContent(i,i,e*e); */
    // test: non-zero covariance where variance is zero
    //if(e<=0.) inputEmatrix->SetBinContent(i,i,0.000000001);
  //}
  tunfold.SetInput(histDataReco);
  tunfold.AddSysError(histMCGenRecUp, "Up", TUnfoldDensity::kHistMapOutputHoriz, TUnfoldDensity::kSysErrModeMatrix);
  tunfold.AddSysError(histMCGenRecDwn, "Down", TUnfoldDensity::kHistMapOutputHoriz, TUnfoldDensity::kSysErrModeMatrix);
  tunfold.DoUnfold(0.0);

  // Get output -- DOES NOT CONTAIN SYSTEMATICS
  histDataUnfolded = tunfold.GetOutput("o");
  //tunfoldTikhonovLCurve.SetInput(input); //need?
  //Scan L curves for best tau
  //int iBest=tunfold.ScanLcurve(NPOINT_TikhonovLCurve, 0.0000000001, 1., &graph_LCurve, &spline_logTauX, &spline_logTauY);
  //Set tau value
  //  tauBest=tunfold.GetTau();
  //cout<<"Best tau: " << iBest << endl;
  /* //    DF_TikhonovLCurve=tunfoldTikhonovLCurve.GetDF(); */
  /* // all unfolded bins including fakes normalisation */
  /* histDataUnfolded=tunfoldTikhonovLCurve.GetOutput("hist_unfolded_TikhonovLCurve",";bin",0,0,false); */
  /* //histDataUnfolded->Write(); */
  /* // get matrix of probabilities */
  /* TH2 *histProbability=tunfoldTikhonovLCurve.GetProbabilityMatrix("histProbability"); */

  TCanvas *c4 = new TCanvas("c4","Plot unfolding outputs",600,600);
  c4->cd();
  histDataUnfolded->SetLineColor(kBlue);
  histDataUnfolded->Draw("E");
  histMCTruth_M->SetLineColor(kRed); 
  histMCTruth_M->Draw("SAME HIST");
  auto leg = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg->AddEntry(histDataUnfolded, "Unfolded, total unc","p");
  leg->AddEntry(histMCTruth_M, "MCTruth","p");
  leg->Draw();
    
}
