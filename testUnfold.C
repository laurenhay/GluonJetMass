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
  TFile *f = TFile::Open("dijetHistsQCDsim_jec_2016.root");
  TH2D *ptgen_mgen_g = f->Get<TH2D>("ptgen_mgen_g_nominal");
  TH2D *ptgen_mgen_u = f->Get<TH2D>("ptgen_mgen_u_nominal");
  TH2D *fakes_ptreco_mreco = f->Get<TH2D>("fakes_ptreco_mreco");
  TH2D *misses_ptgen_mgen = f->Get<TH2D>("misses_ptgen_mgen");

  ptgen_mgen_g->Print("base");
  
  //open data root file and get hists
  TFile *f_reco = TFile::Open("dijetHistsQCDsim_jec_2016.root");
  TH2D *ptreco_mreco_u = f_reco->Get<TH2D>("jet_ptreco_mreco_u_nominal");
  TH2D *ptreco_mreco_u_up = f_reco->Get<TH2D>("jet_ptreco_mreco_u_jerUp");
  TH2D *ptreco_mreco_u_dn = f_reco->Get<TH2D>("jet_ptreco_mreco_u_jerDown");
  TH2D *ptreco_mreco_g = f_reco->Get<TH2D>("jet_ptreco_mreco_g_nominal");
  TH2D *ptreco_mreco_g_up = f_reco->Get<TH2D>("jet_ptreco_mreco_g_jerUp");
  TH2D *ptreco_mreco_g_dn = f_reco->Get<TH2D>("jet_ptreco_mreco_g_jerDown");

  //ptreco_mreco_u->Print("all");
  
  // Create a TTreeReader for the tree by passing the TTree's name and the TDirectory / TFile it is in.
  TTreeReader matrixReader("response", f);
  TTreeReader binReader("centers", f);
  // The branch "reco_gen_groomed" contains doubles; access them as response_matrix_g
  TTreeReaderArray<Double_t> response_matrix_g(matrixReader, "groomed_nominal");
  TTreeReaderArray<Double_t> response_matrix_g_up(matrixReader, "groomed_jesUp");
  TTreeReaderArray<Double_t> response_matrix_g_dn(matrixReader, "groomed_jesDown");
  // The branch "reco_gen_ungroomed" contains doubles; access them as response_matrix_u
  TTreeReaderArray<Double_t> response_matrix_u(matrixReader, "ungroomed_nominal");
  TTreeReaderArray<Double_t> response_matrix_u_up(matrixReader, "ungroomed_jesUp");
  TTreeReaderArray<Double_t> response_matrix_u_dn(matrixReader, "ungroomed_jesDown");
  // Bin center arrays
  TTreeReaderArray<Double_t> ptreco_center(binReader, "ptreco");
  TTreeReaderArray<Double_t> mreco_center(binReader, "mreco");
  TTreeReaderArray<Double_t> ptgen_center(binReader, "ptgen");
  TTreeReaderArray<Double_t> mgen_center(binReader, "mgen");
  const TArrayD ptgen_low = *ptgen_mgen_u->GetXaxis()->GetXbins();
  const TArrayD mgen_low = *ptgen_mgen_u->GetYaxis()->GetXbins();
  const TArrayD ptreco_low = *ptreco_mreco_u->GetXaxis()->GetXbins();
  const TArrayD mreco_low = *ptreco_mreco_u->GetYaxis()->GetXbins();

  cout << "mass gen low edge from TH2 size " << mgen_low.GetSize()<<endl;
  
    //check content by drawing hists
  /* TCanvas *c1 = new TCanvas("c1","FAKES AND MISSES",1200,600); */
  /* c1->Divide(2,1); */
  /* c1->cd(1); */
  /* misses_ptgen_mgen->Draw("colz"); */
  /* c1->cd(2); */
  /* c1->SetLogz(); */
  /* fakes_ptreco_mreco->Draw("colz"); */

  //define mass binnings
  vector<double> binsMassCoarseVec{0.,1.,5.,10.,20.,40.,60.,80.,100.,150.,200.,250.,1000.}; //gen
  int nBinMassCoarse=binsMassCoarseVec.size()-1;
  vector<double> binsVector[2];
  for(int i=0;i<=nBinMassCoarse;i++) {
    double x0=binsMassCoarseVec[i];
    //    cout << "mass gen low edge from array " << mgen_center[i]<<endl;
    //cout << "mass gen low edge from TH2 " << mgen_low[i]<<endl;
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
  recoBinning->AddAxis(*ptreco_mreco_u->GetYaxis(), false, false); //mreco
  recoBinning->AddAxis(*ptgen_mgen_u->GetXaxis(), false, false); //ptreco
  //recoBinning->AddAxis("m_{RECO}", nBinMassFine, binsMassFine.data(), false, true); //mreco
  //recoBinning->AddAxis("pt_{RECO}", nBinsPt, binsPt.data(), false, false); //ptreco

  
  TUnfoldBinning *generatorBinning = new TUnfoldBinning("generator");
  //  TUnfoldBinning *fakeBinning=generatorBinning->AddBinning("fakesBin", 1);
  TUnfoldBinning *genBinning = generatorBinning->AddBinning("gen");
  genBinning->AddAxis(*ptgen_mgen_u->GetYaxis(), false, false); //mgen
  genBinning->AddAxis(*ptgen_mgen_u->GetXaxis(), false, false); //ptgen
  //genBinning->AddAxis("m_{GEN}", nBinMassCoarse, binsMassCoarse.data(), false, true); //mgen
  //genBinning->AddAxis("pt_{GEN}", nBinsPt, binsPt.data(), false, true); //ptgen  
  cout << "Test binning for ptreco = 250, mreco = 500 " <<recoBinning->GetGlobalBinNumber(250,500) << endl;
  
  // create histogram of migrations and gen hists
  TH2D *histMCGenRec_u=TUnfoldBinning::CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec Ungroomed");
  TH2D *histMCGenRecUp_u=TUnfoldBinning::CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec+jerUp Ungroomed");
  TH2D *histMCGenRecDn_u=TUnfoldBinning::CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec-jerDn Ungroomed");
  TH2D *histMCGenRec_g=TUnfoldBinning::CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec Groomed");
  TH2D *histMCGenRecUp_g=TUnfoldBinning::CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec+jerUp Groomed");
  TH2D *histMCGenRecDn_g=TUnfoldBinning::CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec-jerDn Groomed");
  // create data histograms
  TH1 *histDataReco_u=recoBinning->CreateHistogram("histDataReco Ungroomed");  //data values in reco binnning --> input to unfolding
  TH1 *histDataReco_g=recoBinning->CreateHistogram("histDataReco Groomed");  //data values in reco binnning --> input to unfolding

  //create MC hists for comparison
  TH1 *histMCReco_u=recoBinning->CreateHistogram("histMCReco Ungroomed"); //gen values in reco binning --> htruef in Sal's example
  TH1 *histMCTruth_u=genBinning->CreateHistogram("histMCTruth Ungroomed");  //gen values in gen binning
  TH1 *histMCReco_g=recoBinning->CreateHistogram("histMCReco Groomed"); //gen values in reco binning --> htruef in Sal's example
  TH1 *histMCTruth_g=genBinning->CreateHistogram("histMCTruth Groomed");  //gen values in gen binning
  
  // Loop through reco and gen bins of MC input and fill hist of migrations
  while(matrixReader.Next() && binReader.Next()) {
    cout << "Type of ungroomed: " << typeid(response_matrix_u).name() << " and size: " << response_matrix_u.GetSize() << endl;
    cout << "Size of ptreco bins: " << ptreco_center.GetSize()<< endl;
    cout << "Size of mreco bins: " << mreco_center.GetSize()<< endl;
    cout << "Size of mgen bins: " << mgen_center.GetSize()<< endl;
    cout << "Size of ptreco bins: " << ptreco_low.GetSize()<< endl;
    cout << "Size of mreco bins: " << mreco_low.GetSize()<< endl;
    cout << "Size of mgen bins: " << mgen_low.GetSize()<< endl;
    Int_t glob_recobin;
    Int_t recoBin;
    Int_t glob_genbin;
    Int_t genBin;
    // reco loop: i is ptreco, j is mreco
    for(int i=0; i<(ptreco_low.GetSize()-1); i++){
      for(int j=0; j<(mreco_low.GetSize()-1); j++){
    /* for(int i=0; i<NBin_pt_fine; i++){ */
    /*   for(int j=0; j<NBin_mass_fine; j++){	 */
	glob_recobin=(i)*(mreco_low.GetSize()-1)+j;
	cout<<"has centers mreco " << mreco_low[j] << " and ptreco " << ptreco_low[i] << endl;
	recoBin=recoBinning->GetGlobalBinNumber(mreco_low[j],ptreco_low[i]);
	//only fill fakes in fake genBin
	//	    Double_t fake_weight = fakes_ptreco_mreco->GetBinContent(i,j);
	// cout << "Fake weight " << fake_weight << " for i == " << i << " and j == " << j << endl;
	//Int_t fakeBin=fakeBinning->GetStartBin();
	//histMCGenRec->SetBinContent(fakeBin,recoBin,fake_weight);
	//fill data hist
	Double_t data_weight_u=ptreco_mreco_u->GetBinContent(i+1,j+1);
	histDataReco_u->Fill(recoBin, data_weight_u);
	cout << "Reco weight " << data_weight_u << " for glob reco bin "<< glob_recobin << "and matrix reco bin " << recoBin << endl;
	Double_t data_weight_g=ptreco_mreco_g->GetBinContent(i+1,j+1);
	histDataReco_g->Fill(recoBin, data_weight_g);
	// gen loop: k is ptgen, l is mgen
	for(int k=0; k<(mgen_low.GetSize()-1); k++){
	  for(int l=0; l<(mgen_low.GetSize()-1); l++){
	    glob_genbin=(k)*(mgen_low.GetSize()-1)+l;
	    genBin=genBinning->GetGlobalBinNumber(mgen_low[l],ptgen_low[k]);
	    //	    cout<< genBin << " has lower edges mgen " << mgen_low[l] << " and ptgen " << ptgen_low[k] << endl;
	    //only fill misses in miss recoBin
	    //Double_t miss_weight = misses_ptgen_mgen->GetBinContent(k,l);
	    /* cout << "Miss weight " << miss_weight << " for k == " << k << " and k == " << k << endl; */
	    //Int_t missBin=missBinning->GetStartBin();
	    //histMCGenRec->SetBinContent(genBin,missBin,miss_weight);
	    //fill MC truth for closure test
	    //ONLY FILL ONCE INSIDE i,j LOOP
	    if(i==0 && j==0){
	      Double_t truth_weight_u = ptgen_mgen_u->GetBinContent(k+1,l+1);
	      histMCTruth_u->Fill(genBin, truth_weight_u);
	      cout << "Truth weight " << truth_weight_u << " for glob gen bin "<< glob_genbin << "and matrix gen bin " << genBin << endl;
	      Double_t truth_weight_g = ptgen_mgen_g->GetBinContent(k+1,l+1);
	      histMCTruth_g->Fill(genBin, truth_weight_g);
	      //fill MC reco for comparison
	      // COULD BE GOING WRONG BC DIFF NUMBER OF K's AND L's VS I's AND J's
	      Int_t recoBin_genObj=recoBinning->GetGlobalBinNumber(mgen_low[l],ptgen_low[k]);
	      cout << "Truth weight " << truth_weight_u << " for k == " << k << " and l == " << l << "and reco bin " << recoBin_genObj << endl;
	      histMCReco_u->Fill(recoBin_genObj, truth_weight_u);
	      histMCReco_g->Fill(recoBin_genObj, truth_weight_g);
	  }
	    Int_t glob_bin = glob_recobin*(mgen_center.GetSize()*ptgen_center.GetSize())+glob_genbin;
	    /* cout<<"Global bin " << glob_bin << " for reco bin " << glob_recobin << " and gen bin " << glob_genbin << endl; */
	    //		    cout <<"TUnfold gen bin "<< genBin << " and reco bin " << recoBin<< "for value" << response_matrix_u[glob_bin] <<endl;
	    // fill ungroomed resp. matrices
	    Double_t resp_weight_u = response_matrix_u[glob_bin];
	    histMCGenRec_u->Fill(genBin,recoBin,resp_weight_u);
	    Double_t resp_weight_u_up = response_matrix_u_up[glob_bin];
	    histMCGenRecUp_u->Fill(genBin,recoBin,resp_weight_u_up);
	    Double_t resp_weight_u_dn = response_matrix_u_dn[glob_bin];
	    histMCGenRecDn_u->Fill(genBin,recoBin,resp_weight_u_dn);
	    /* // fill groomed resp. matrices */
	    Double_t resp_weight_g = response_matrix_g[glob_bin];
	    histMCGenRec_g->Fill(genBin,recoBin,resp_weight_g);
	    Double_t resp_weight_g_up = response_matrix_g_up[glob_bin];
	    histMCGenRecUp_g->Fill(genBin,recoBin,resp_weight_g_up);
	    Double_t resp_weight_g_dn = response_matrix_g_dn[glob_bin];
	    histMCGenRecDn_g->Fill(genBin,recoBin,resp_weight_g_dn);
	  }}}}
    cout << response_matrix_u[849] <<endl;              
  }
  
  //histMCGenRec->Print("base");
  cout<<"MC Reco (filled with MC gen but reco bins)" << endl;
  histMCReco_u->Print("base");
  cout<<"MC Truth (filled with MC gen and gen bins)" << endl;
  histMCTruth_u->Print("base");
  cout<<"Response matrix"<<endl;
  histMCGenRec_u->Print("base");
  // check that response matrix has been filled properly
  TH1 *histMCReco_u_M=histMCGenRec_u->ProjectionY("MCReco u", 1, -2);
  TH1 *histMCTruth_u_M=histMCGenRec_u->ProjectionX("MCTruth u", 1, -2);
  histMCReco_u_M->Print("all");
  // check that response matrix has been filled properly
  TH1 *histMCReco_g_M=histMCGenRec_g->ProjectionY("MCReco g", 1, -2); 
  TH1 *histMCTruth_g_M=histMCGenRec_g->ProjectionX("MCTruth g", 1, -2);
  
  //check Truth contents
  int nBinsTruth = histMCTruth_u_M->GetXaxis()->GetNbins();
  for(int bin = 0; bin < nBinsTruth; bin++){
    cout << "Matrix truth value: " << histMCTruth_u_M->GetBinContent(bin) << " for matrix bin : "<< bin << endl;
  }
  int nBinsReco = histMCReco_u_M->GetXaxis()->GetNbins();
  for(int bin = 0; bin < nBinsReco; bin++){
    cout << "Matrix reco value: " << histMCReco_u_M->GetBinContent(bin) << " for matrix bin : "<< bin << endl;
  }
  
  TCanvas *c2 = new TCanvas("c2","Plot MC input ungroomed binned by pt (outer) and mass (inner)",1200,400);
  c2->Divide(2,1);
  c2->cd(1);
  histMCTruth_u_M->SetMarkerStyle(21);
  histMCTruth_u_M->SetLineColor(kRed);
  histMCTruth_u_M->SetMarkerColor(kRed);
  histMCTruth_u_M->Draw();
  histMCTruth_u->SetMarkerStyle(24);
  histMCTruth_u->SetLineColor(kBlue);
  histMCTruth_u->SetMarkerColor(kBlue);
  histMCTruth_u->Draw("SAME");
  auto leg1 = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg1->AddEntry(histMCTruth_u_M, "MC Gen from M", "p");
  leg1->AddEntry(histMCTruth_u, "MC Grn", "p");
  leg1->Draw();
  
  c2->cd(2);
  histMCReco_u_M->SetMarkerStyle(21);
  histMCReco_u_M->SetLineColor(kRed);
  histMCReco_u_M->SetMarkerColor(kRed);
  histMCReco_u_M->Draw();
  histMCReco_u->SetMarkerStyle(24);
  histMCReco_u->SetLineColor(kBlue);
  histMCReco_u->SetMarkerColor(kBlue);
  histMCReco_u->Draw("SAME");
  auto leg2 = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg2->AddEntry(histMCReco_u_M, "MC Reco from M", "p");
  leg2->AddEntry(histMCReco_u, "MC Reco", "p");
  leg2->Draw();

  TCanvas *c2g = new TCanvas("c2g","Plot MC input groomed binned by pt (outer) and mass (inner)",1200,400);
  c2g->Divide(2,1);
  c2g->cd(1);
  histMCTruth_g_M->SetMarkerStyle(21);
  histMCTruth_g_M->SetLineColor(kRed);
  histMCTruth_g_M->SetMarkerColor(kRed);
  histMCTruth_g_M->Draw();
  histMCTruth_g->SetMarkerStyle(24);
  histMCTruth_g->SetLineColor(kBlue);
  histMCTruth_g->SetMarkerColor(kBlue);
  histMCTruth_g->Draw("SAME");
  auto leg1g = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg1g->AddEntry(histMCTruth_g_M, "MC Gen from M", "p");
  leg1g->AddEntry(histMCTruth_g, "MC Gen g", "p");
  leg1g->Draw();
  
  c2g->cd(2);
  histMCReco_g_M->SetMarkerStyle(21);
  histMCReco_g_M->SetLineColor(kRed);
  histMCReco_g_M->SetMarkerColor(kRed);
  histMCReco_g_M->Draw();
  histMCReco_g->SetMarkerStyle(24);
  histMCReco_g->SetLineColor(kBlue);
  histMCReco_g->SetMarkerColor(kBlue);
  histMCReco_g->Draw("SAME");
  auto leg2g = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg2g->AddEntry(histMCReco_g_M, "MC Reco from M", "p");
  leg2g->AddEntry(histMCReco_g, "MC Reco g", "p");
  leg2g->Draw();

  
  TCanvas *c3 = new TCanvas("c3","Plot full response",1200,800);
  c3->Divide(3,1);
  c3->cd(1);
  c3->SetLogz();
  histMCGenRec_u->Draw("colz");
  c3->cd(2);
  c3->SetLogz();
  histMCGenRecUp_u->Draw("colz");
  c3->cd(3);
  c3->SetLogz();
  histMCGenRecDn_u->Draw("colz");
  
  //=======================================================
  // open file to save histograms and binning schemes
  
  TFile *outputFile=new TFile("testUnfold_histograms.root","recreate");
  
  detectorBinning->Write();
  generatorBinning->Write();

  histMCReco_u->Write();
  histMCGenRec_u->Write();
  histMCTruth_u->Write();
  
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
  TUnfoldDensity tunfold(histMCGenRec_u,TUnfoldDensity::kHistMapOutputHoriz,
				       TUnfoldDensity::kRegModeCurvature,TUnfoldDensity::kEConstraintArea,
			 TUnfoldDensity::kDensityModeBinWidth); //, generatorBinning, detectorBinning);
    // set up matrix of migrations
  /* TH1 *input; */
  /* input = histMCReco_g_M; */
  /* TH2D *inputEmatrix= */
  /*   detectorBinning->CreateErrorMatrixHistogram("input_covar",true); */
  /* for(int i=1;i<=inputEmatrix->GetNbinsX();i++) { */
  /*   Double_t e=input->GetBinError(i); */
  /*   cout << "Error in MC: " << e << endl; */
  /*   inputEmatrix->SetBinContent(i,i,e*e); */
    // test: non-zero covariance where variance is zero
    //if(e<=0.) inputEmatrix->SetBinContent(i,i,0.000000001);
  //}
  tunfold.SetInput(histMCReco_u_M);
  tunfold.AddSysError(histMCGenRecUp_u, "Up", TUnfoldDensity::kHistMapOutputHoriz, TUnfoldDensity::kSysErrModeMatrix);
  tunfold.AddSysError(histMCGenRecDn_u, "Down", TUnfoldDensity::kHistMapOutputHoriz, TUnfoldDensity::kSysErrModeMatrix);
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
  histMCTruth_u_M->SetLineColor(kRed);
  histMCTruth_u_M->Draw("SAME HIST");
  auto leg = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg->AddEntry(histDataUnfolded, "Unfolded, total unc","p");
  leg->AddEntry(histMCTruth_u_M, "MCTruth","p");
  leg->Draw();
    
}
