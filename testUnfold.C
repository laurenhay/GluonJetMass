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

//TH2D* NormM( TH2D* InputMatrix, )

void testUnfold()
{
  //open MC root file from coffea/uproot and get hists
  /* TFile *f = TFile::Open("trijetHistsQCDsim_jec2016.root"); */
  TFile *f = TFile::Open("dijetHistsQCDsim_jec_2016.root");
  TH2D *ptgen_mgen_u = f->Get<TH2D>("ptgen_mgen_u_nominal");
  TH2D *ptgen_mgen_u_up = f->Get<TH2D>("ptgen_mgen_u_jerUp");
  TH2D *ptgen_mgen_u_dn = f->Get<TH2D>("ptgen_mgen_u_jerDown");
  TH2D *ptgen_mgen_g = f->Get<TH2D>("ptgen_mgen_g_nominal");
  TH2D *ptgen_mgen_g_up = f->Get<TH2D>("ptgen_mgen_g_jerUp");
  TH2D *ptgen_mgen_g_dn = f->Get<TH2D>("ptgen_mgen_g_jerDown");
  TH2D *fakes_ptreco_mreco = f->Get<TH2D>("fakes_ptreco_mreco_nominal");
  //  TH2D *misses_ptgen_mgen = f->Get<TH2D>("misses_ptgen_mgen_nominal");
  TH2D *ptreco_mreco_u = f->Get<TH2D>("ptreco_mreco_u_nominal");
  TH2D *ptreco_mreco_u_up = f->Get<TH2D>("ptreco_mreco_u_jerUp");
  TH2D *ptreco_mreco_u_dn = f->Get<TH2D>("ptreco_mreco_u_jerDown");
  TH2D *ptreco_mreco_g = f->Get<TH2D>("ptreco_mreco_g_nominal");
  TH2D *ptreco_mreco_g_up = f->Get<TH2D>("ptreco_mreco_g_jerUp");
  TH2D *ptreco_mreco_g_dn = f->Get<TH2D>("ptreco_mreco_g_jerDown");
  
  ptreco_mreco_u->Print("base");
  ptreco_mreco_g->Print("base");
  
  
  //open data root file and get hists
  /* TFile *f_data = TFile::Open("trijetHistsJetHT_jec_2016.root"); */
  TFile *f_data = TFile::Open("dijetHistsJetHT_jec_2016.root");
  TH2D *data_pt_m_u = f_data->Get<TH2D>("ptreco_mreco_u_nominal");
  /* TH2D *data_pt_m_u_up = f_data->Get<TH2D>("ptreco_mreco_u_jerUp"); */
  /* TH2D *data_pt_m_u_dn = f_data->Get<TH2D>("ptreco_mreco_u_jerDown"); */
  TH2D *data_pt_m_g = f_data->Get<TH2D>("ptreco_mreco_g_nominal");
  /* TH2D *data_pt_m_g_up = f_data->Get<TH2D>("ptreco_mreco_g_jerUp"); */
  /* TH2D *data_pt_m_g_dn = f_data->Get<TH2D>("ptreco_mreco_g_jerDown"); */
  
  // Create a TTreeReader for the tree by passing the TTree's name and the TDirectory / TFile it is in.
  TTreeReader matrixReader("response", f);
  // The branch "reco_gen_groomed" contains doubles; access them as response_matrix_g
  TTreeReaderArray<Double_t> response_matrix_g(matrixReader, "groomed_nominal");
  TTreeReaderArray<Double_t> response_matrix_g_up(matrixReader, "groomed_jesUp");
  TTreeReaderArray<Double_t> response_matrix_g_dn(matrixReader, "groomed_jesDown");
  // The branch "reco_gen_ungroomed" contains doubles; access them as response_matrix_u
  TTreeReaderArray<Double_t> response_matrix_u(matrixReader, "ungroomed_nominal");
  TTreeReaderArray<Double_t> response_matrix_u_up(matrixReader, "ungroomed_jesUp");
  TTreeReaderArray<Double_t> response_matrix_u_dn(matrixReader, "ungroomed_jesDown");
  // Bin edge arrays
  TArrayD ptgen_low = *ptgen_mgen_u->GetXaxis()->GetXbins();
  TArrayD mgen_low = *ptgen_mgen_u->GetYaxis()->GetXbins();
  TArrayD ptreco_low = *ptreco_mreco_u->GetXaxis()->GetXbins();
  TArrayD mreco_low = *ptreco_mreco_u->GetYaxis()->GetXbins();
  int n_ptgen = ptgen_low.GetSize();
  /* //cout << "Size of tarray ptgen: " << ptgen_low.GetSize() << endl; */
  /* ptgen_low.Set(ptgen_low.GetSize()+1); */
  /* //  cout << "Size of tarray ptgen: " << ptgen_low.GetSize() << endl; */
  
  //MAKE TUNFOLD binning axes
  TUnfoldBinning *detectorBinning = new TUnfoldBinning("detector");
  TUnfoldBinning *recoBinning = detectorBinning->AddBinning("reco");
  recoBinning->AddAxis(*ptreco_mreco_u->GetYaxis(), false, false); //mreco
  recoBinning->AddAxis(*ptreco_mreco_u->GetXaxis(), false, false); //ptreco
    
  
  TUnfoldBinning *generatorBinning = new TUnfoldBinning("generator");
  TUnfoldBinning *fakeBinning=generatorBinning->AddBinning("fakesBin", 1);
  TUnfoldBinning *genBinning = generatorBinning->AddBinning("gen");
  genBinning->AddAxis(*ptgen_mgen_u->GetYaxis(), false, false); //mgen
  genBinning->AddAxis(*ptgen_mgen_u->GetXaxis(), false, false); //ptgen
  
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
  /* TH1 *histDataReco_u_up=recoBinning->CreateHistogram("histDataReco Ungroomed Up");  //data values in reco binnning --> check unc. */
  /* TH1 *histDataReco_u_dn=recoBinning->CreateHistogram("histDataReco Ungroomed Down");  //data values in reco binnning --> check unc. */
  TH1 *histDataReco_g=recoBinning->CreateHistogram("histDataReco Groomed");  //data values in reco binnning --> input to unfolding
  /* TH1 *histDataReco_g_up=recoBinning->CreateHistogram("histDataReco Groomed Up");  //data values in reco binnning --> check unc. */
  /* TH1 *histDataReco_g_dn=recoBinning->CreateHistogram("histDataReco Groomed Down");  //data values in reco binnning --> check unc. */

  
  //create MC hists for comparison
  TH1 *histMCTruth_RecoBinned_u=recoBinning->CreateHistogram("histMCTruth Reco Binned, Ungroomed"); //gen values in reco binning --> htruef in Sal's example
  TH1 *histMCReco_u=recoBinning->CreateHistogram("histMCReco Ungroomed"); //mc reco values --> h in Sal's example
  TH1 *histMCReco_u_up=recoBinning->CreateHistogram("histMCReco Ungroomed Up"); //mc reco values --> unc. check
  TH1 *histMCReco_u_dn=recoBinning->CreateHistogram("histMCReco Ungroomed DOwn"); //mc reco values --> unc. check
  TH1 *histMCTruth_u=genBinning->CreateHistogram("histMCTruth Ungroomed");  //gen values in gen binning
  TH1 *histMCTruth_u_up=genBinning->CreateHistogram("histMCTruth Ungroomed Up");  //gen values in gen binning --> unc. check
  TH1 *histMCTruth_u_dn=genBinning->CreateHistogram("histMCTruth Ungroomed Down");  //gen values in gen binning --> unc. check
  TH1 *histMCTruth_RecoBinned_g=recoBinning->CreateHistogram("histMCTruth Reco Binned, Groomed"); //gen values in reco binning --> htruef in Sal's example
  TH1 *histMCReco_g=recoBinning->CreateHistogram("histMCReco Groomed"); //mc reco values --> h in Sal's example
  TH1 *histMCReco_g_up=recoBinning->CreateHistogram("histMCReco Groomed Up"); //mc reco values --> unc. check
  TH1 *histMCReco_g_dn=recoBinning->CreateHistogram("histMCRecoGroomed DOwn"); //mc reco values --> unc. check
  TH1 *histMCTruth_g=genBinning->CreateHistogram("histMCTruth Groomed");  //gen values in gen binning
  TH1 *histMCTruth_g_up=genBinning->CreateHistogram("histMCTruth Groomed Up");  //gen values in gen binning --> unc. check
  TH1 *histMCTruth_g_dn=genBinning->CreateHistogram("histMCTruth Groomed Down");  //gen values in gen binning --> unc. check
  
  // Loop through reco and gen bins of MC input and fill hist of migrations
  while(matrixReader.Next()) {
    cout << "Size of ptreco bins: " << ptreco_low.GetSize()<< endl;
    cout << "Size of mreco bins: " << mreco_low.GetSize()<< endl;
    cout << "Size of ptgen bins: " << ptgen_low.GetSize()<< endl;
    cout << "Size of mgen bins: " << mgen_low.GetSize()<< endl;
    cout << "Size of response matrix " << response_matrix_u.GetSize() << endl;
    Int_t glob_recobin;
    Int_t recoBin;
    Int_t glob_genbin;
    Int_t genBin;
    Int_t recoBin_genObj;
    // reco loop: i is ptreco, j is mreco
    for(int i=0; i<(ptreco_low.GetSize()); i++){
      for(int j=0; j<(mreco_low.GetSize()); j++){
	glob_recobin=(i)*(mreco_low.GetSize()+1)+j;
	cout<<"Bin j = " << j << " has edge mreco " << mreco_low[j] << "and  bin i " << i << " has ptreco edge " << ptreco_low[i] << endl;
	recoBin=recoBinning->GetGlobalBinNumber(mreco_low[j],ptreco_low[i]);
	//only fill fakes in fake genBin
	Double_t fake_weight = fakes_ptreco_mreco->GetBinContent(i,j);
	cout << "Fake weight " << fake_weight << " for i == " << i << " and j == " << j << endl;
	Int_t fakeBin=fakeBinning->GetStartBin();
	histMCGenRec_u->SetBinContent(fakeBin,recoBin,fake_weight);
	histMCGenRecUp_u->Fill(fakeBin,recoBin,fake_weight);
	histMCGenRecDn_u->Fill(fakeBin,recoBin,fake_weight);
	
	// fill data reco hist to be input to unfolding
	Double_t data_weight_u=data_pt_m_u->GetBinContent(i,j);
	histDataReco_u->Fill(recoBin, data_weight_u);
	//cout << "Data weight " << data_weight_u << " for glob reco bin "<< glob_recobin << "and matrix reco bin " << recoBin << endl;
	Double_t data_weight_g=data_pt_m_g->GetBinContent(i,j);
	cout << "Data weight " << data_weight_g << " for glob reco bin "<< glob_recobin << "and matrix reco bin " << recoBin << endl;
	histDataReco_g->Fill(recoBin, data_weight_g);
	// do same for up and down uncertainties for checks
	/* Double_t data_weight_u_up=data_pt_m_u_up->GetBinContent(i,j); */
	/* histDataReco_u_up->Fill(recoBin, data_weight_u_up); */
	/* Double_t data_weight_g_up=data_pt_m_g_up->GetBinContent(i,j); */
	/* histDataReco_g_up->Fill(recoBin, data_weight_g_up); */
	/* Double_t data_weight_u_dn=data_pt_m_u_dn->GetBinContent(i,j); */
	/* histDataReco_u_dn->Fill(recoBin, data_weight_u_dn); */
	/* Double_t data_weight_g_dn=data_pt_m_g_dn->GetBinContent(i,j); */
	/* histDataReco_g_dn->Fill(recoBin, data_weight_g_dn); */
	
	// fill MC reco hist for comparison of inputs
	Double_t reco_weight_u=ptreco_mreco_u->GetBinContent(i,j);
	histMCReco_u->Fill(recoBin, reco_weight_u);
	cout << "Reco weight ungroomed " << reco_weight_u << " for glob reco bin "<< glob_recobin << "and matrix reco bin " << recoBin << endl;
	Double_t reco_weight_g=ptreco_mreco_g->GetBinContent(i,j);
	histMCReco_g->Fill(recoBin, reco_weight_g);
	cout << "Reco weight groomed " << reco_weight_g << " for glob reco bin "<< glob_recobin << "and matrix reco bin " << recoBin << endl;
	// do same for up and down uncertainties for checks
	Double_t reco_weight_u_up=ptreco_mreco_u_up->GetBinContent(i,j);
	histMCReco_u_up->Fill(recoBin, reco_weight_u);
	Double_t reco_weight_g_up=ptreco_mreco_g_up->GetBinContent(i,j);
	histMCReco_g_up->Fill(recoBin, reco_weight_g_up);
	Double_t reco_weight_u_dn=ptreco_mreco_u_dn->GetBinContent(i,j);
	histMCReco_u_dn->Fill(recoBin, reco_weight_u_dn);
	Double_t reco_weight_g_dn=ptreco_mreco_g_dn->GetBinContent(i,j);
	histMCReco_g_dn->Fill(recoBin, reco_weight_g_dn);
	
	// gen loop: k is ptgen, l is mgen
	for(int k=0; k<(ptgen_low.GetSize()); k++){
	  for(int l=0; l<(mgen_low.GetSize()); l++){
	    glob_genbin=(k)*(mgen_low.GetSize()+1)+l;
	    genBin=genBinning->GetGlobalBinNumber(mgen_low[l],ptgen_low[k]);
	    //cout<< genBin << " has lower edges mgen " << mgen_low[l] << " and ptgen " << ptgen_low[k] << endl;
	    //fill MC truth for closure test
	    //ONLY FILL ONCE INSIDE i,j LOOP
	    if(i==0 && j==0){
	      // fill truth inputs for comparison
	      Double_t truth_weight_u = ptgen_mgen_u->GetBinContent(k,l);
	      histMCTruth_u->Fill(genBin, truth_weight_u);
	      //cout << "Truth weight " << truth_weight_u << " for glob gen bin "<< glob_genbin << "and matrix gen bin " << genBin << endl;
	      Double_t truth_weight_g = ptgen_mgen_g->GetBinContent(k,l);
	      histMCTruth_g->Fill(genBin, truth_weight_g);
	      // do same for uncertainties
	      Double_t truth_weight_u_up = ptgen_mgen_u_up->GetBinContent(k,l);
	      histMCTruth_u_up->Fill(genBin, truth_weight_u_up);
	      Double_t truth_weight_g_up = ptgen_mgen_g_up->GetBinContent(k,l);
	      histMCTruth_g_up->Fill(genBin, truth_weight_g_up);
	      Double_t truth_weight_u_dn = ptgen_mgen_u_dn->GetBinContent(k,l);
	      histMCTruth_u_dn->Fill(genBin, truth_weight_u_dn);
	      Double_t truth_weight_g_dn = ptgen_mgen_g_dn->GetBinContent(k,l);
	      histMCTruth_g_dn->Fill(genBin, truth_weight_g_dn);
	      
	      //Fill truth but binned in reco for comparison
	      //recoBin_genObj=recoBinning->GetGlobalBinNumber(mgen_low[l],ptgen_low[k]);
	      //cout << "With pt edge " << ptgen_low[k] << " for k == " << k << " and m edge " << mgen_low[l]  <<" for l ==  " << l << "and reco bin " << recoBin_genObj << endl;
	      histMCTruth_RecoBinned_u->Fill(recoBin_genObj, truth_weight_u);
	      histMCTruth_RecoBinned_g->Fill(recoBin_genObj, truth_weight_g);
	  }
	    Int_t glob_bin = glob_recobin*((mgen_low.GetSize()+1)*(ptgen_low.GetSize()+1))+glob_genbin;
	    /* cout<<"Global bin " << glob_bin << " for reco bin " << glob_recobin << " and gen bin " << glob_genbin << endl; */
	    //cout << "Response weight groomed " << response_matrix_g[glob_bin] <<" for gen bin "<< genBin << " and reco bin " << recoBin<< " and global bin "<<glob_bin<<endl;
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
  TH1 *histMCReco_u_M=histMCGenRec_u->ProjectionY("MCReco u");
  TH1 *histMCTruth_u_M=histMCGenRec_u->ProjectionX("MCTruth u");
  histMCReco_u_M->Print("all");
  //do same for uncertainties
  TH1 *histMCReco_u_M_up=histMCGenRecUp_u->ProjectionY("MCReco ungroomed Up");
  TH1 *histMCReco_u_M_dn=histMCGenRecDn_u->ProjectionY("MCReco ungroomed Down");
  TH1 *histMCTruth_u_M_up=histMCGenRecUp_u->ProjectionX("MCTruth ungroomed Up");
  TH1 *histMCTruth_u_M_dn=histMCGenRecDn_u->ProjectionX("MCTruth ungroomed Down");
  
  // check that response matrix has been filled properly
  TH1 *histMCReco_g_M=histMCGenRec_g->ProjectionY("MCReco g"); 
  TH1 *histMCTruth_g_M=histMCGenRec_g->ProjectionX("MCTruth g");
  
  //check Truth contents
  int nBinsTruth = histMCTruth_u_M->GetXaxis()->GetNbins();
  for(int bin = 0; bin < nBinsTruth; bin++){
    cout << "Matrix truth value: " << histMCTruth_u_M->GetBinContent(bin) << " for matrix bin : "<< bin << endl;
  }
  int nBinsReco = histMCReco_u_M->GetXaxis()->GetNbins();
  for(int bin = 0; bin < nBinsReco; bin++){
    cout << "Matrix reco value: " << histMCReco_u_M->GetBinContent(bin) << " for matrix bin : "<< bin << endl;
  }

  TCanvas *c1 = new TCanvas("c1","Plot input uncertainties",400,400);
  c1->cd();
  histMCReco_u->SetMarkerStyle(20);
  histMCReco_u->SetLineColor(kBlack);
  histMCReco_u->SetMarkerColor(kBlack);
  histMCReco_u->Draw("HIST P");
  histMCReco_u_up->SetMarkerStyle(25);
  histMCReco_u_up->SetLineColor(kAzure+5); //kblue = 600
  histMCReco_u_up->SetMarkerColor(860+5);
  histMCReco_u_up->Draw("HIST P SAME");
  histMCReco_u_dn->SetMarkerStyle(23);
  histMCReco_u_dn->SetLineColor(kAzure-5);
  histMCReco_u_dn->SetMarkerColor(kAzure-5);
  histMCReco_u_dn->Draw("HIST P SAME");
  histMCReco_u_M_up->SetLineStyle(2);
  histMCReco_u_M_up->SetLineColor(860+6);
  histMCReco_u_M_up->Draw("HIST SAME");
  histMCReco_u_M_dn->SetLineStyle(3);
  histMCReco_u_M_dn->SetLineColor(kAzure-6);
  histMCReco_u_M_dn->Draw("HIST SAME");

  auto leg1 = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg1->AddEntry(histMCReco_u, "Reco", "p");
  leg1->AddEntry(histMCReco_u_up, "+1#sigma", "p");
  leg1->AddEntry(histMCReco_u_dn, "-1#sigma", "p");
  leg1->AddEntry(histMCReco_u_M_up, "Up from M", "l");
  leg1->AddEntry(histMCReco_u_M_dn, "Down from M", "l");  
  leg1->Draw();

  TCanvas *c15 = new TCanvas("c15","Plot MC input ungroomed ratio plot",600,400);
  c15->cd();
  auto rp = new TRatioPlot(histMCTruth_u, histMCTruth_u_M);
  rp->Draw();
  /* rp->GetLowerRefGraph()->SetMinimum(-1.5); */
  /* rp->GetLowerRefGraph()->SetMaximum(1.5); */
  rp->GetLowerRefYaxis()->SetTitle("ratio");
  TPad *p15 = rp->GetUpperPad();
  TLegend *leg15 = p15->BuildLegend();
  leg15->Draw();
  c15->Update();

  
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
  auto leg2a = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg2a->AddEntry(histMCTruth_u_M, "MC Gen from M", "p");
  leg2a->AddEntry(histMCTruth_u, "MC Gen", "p");
  leg2a->Draw();


  
  c2->cd(2);
  histMCReco_u_M->SetMarkerStyle(21);
  histMCReco_u_M->SetLineColor(kRed);
  histMCReco_u_M->SetMarkerColor(kRed);
  histMCReco_u_M->Draw();
  histMCReco_u->SetMarkerStyle(24);
  histMCReco_u->SetLineColor(kBlue);
  histMCReco_u->SetMarkerColor(kBlue);
  histMCReco_u->Draw("SAME");
  histDataReco_u->SetMarkerStyle(23);
  histDataReco_u->SetLineColor(kBlack);
  histDataReco_u->SetMarkerColor(kBlack);
  histDataReco_u->Draw("SAME");

  auto leg2b = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg2b->AddEntry(histMCReco_u_M, "MC Reco from M", "p");
  leg2b->AddEntry(histMCReco_u, "MC Reco", "p");
  leg2b->AddEntry(histDataReco_u, "Data Reco", "p");
  leg2b->Draw();

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
  histDataReco_g->SetMarkerStyle(23);
  histDataReco_g->SetLineColor(kBlack);
  histDataReco_g->SetMarkerColor(kBlack);
  histDataReco_g->Draw("SAME");
  auto leg2g = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg2g->AddEntry(histMCReco_g_M, "MC Reco from M", "p");
  leg2g->AddEntry(histMCReco_g, "MC Reco", "p");
  leg2g->AddEntry(histDataReco_g, "Data Reco g", "p");
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
  TH1 *histTruthUnfolded=0;
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
  tunfold.SetInput(histDataReco_u);
  tunfold.AddSysError(histMCGenRecUp_u, "Up", TUnfoldDensity::kHistMapOutputHoriz, TUnfoldDensity::kSysErrModeMatrix);
  tunfold.AddSysError(histMCGenRecDn_u, "Down", TUnfoldDensity::kHistMapOutputHoriz, TUnfoldDensity::kSysErrModeMatrix);
  tunfold.DoUnfold(0.0);

  // Get output -- DOES NOT CONTAIN SYSTEMATICS
  histDataUnfolded = tunfold.GetOutput("Unfolded ungroomed data");

  // Do closure test
  tunfold.SetInput(histMCReco_u);
   tunfold.DoUnfold(0.0);
  histTruthUnfolded = tunfold.GetOutput("Unfolded ungroomed mc (closure)");
  //Scan L curves for best tau
  //int iBest=tunfold.ScanLcurve(NPOINT_TikhonovLCurve, 0.0000000001, 1., &graph_LCurve, &spline_logTauX, &spline_logTauY);
  //Set tau value
  //  tauBest=tunfold.GetTau();
  //cout<<"Best tau: " << iBest << endl;
  /* //    DF_TikhonovLCurve=tunfoldTikhonovLCurve.GetDF(); */
  /* // all unfolded bins including fakes normalisation */
  /* histDataUnfolded=tunfoldTikhonovLCurve.GetOutput("hist_unfolded_TikhonovLCurve",";bin",0,0,false); */
  /* //histDataUnfolded->Write(); */
  // get matrix of probabilities
  TH2 *histProbability=tunfold.GetProbabilityMatrix("histProbability");

  TCanvas *c4 = new TCanvas("c4","Plot unfolding outputs",600,600);
  c4->cd();
  histDataUnfolded->SetLineColor(kBlue);
  histDataUnfolded->Draw("E");
  histMCTruth_u_M->SetLineColor(kRed);
  histMCTruth_u_M->Draw("SAME HIST");
  auto leg4 = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg4->AddEntry(histDataUnfolded, "Unfolded, total unc","p");
  leg4->AddEntry(histMCTruth_u_M, "MCTruth","p");
  leg4->Draw();

  TCanvas *c5 = new TCanvas("c5","Plot closure test outputs",600,600);
  c5->cd();
  histTruthUnfolded->SetLineColor(kBlue);
  histTruthUnfolded->Draw("E");
  histMCTruth_u_M->SetLineColor(kRed);
  histMCTruth_u_M->Draw("SAME HIST");
  auto leg5 = new TLegend(0.7, 0.7, 0.86, 0.86);
  leg5->AddEntry(histTruthUnfolded, "Unfolded, total unc","p");
  leg5->AddEntry(histMCTruth_u_M, "MCTruth","p");
  leg5->Draw();

  TCanvas *c6 = new TCanvas("c6","Probability Matrix",600,600);
  c6->cd();
  histProbability->Draw("BOX");
  // (3) unfolded data, data truth, MC truth

    
}
