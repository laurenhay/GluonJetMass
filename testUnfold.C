#include <iostream>
#include <fstream>
#include <TFile.h>
#include <TH2D.h>
#include <TUnfoldBinning.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>



// this file imports root histograms made with coffea and created a TUnfold binning object

using namespace std;

void testUnfold()
{
    
    //open root file from coffea/uproot and get hists
    TFile *f = TFile::Open("trijetHistsQCDsim.root");
    TH2D *mreco_mgen_g = f->Get<TH2D>("mreco_mgen_g");
    TH2D *ptreco_ptgen_g = f->Get<TH2D>("ptreco_ptgen_g");
    TH2D *mreco_mgen_u = f->Get<TH2D>("mreco_mgen_u");
    TH2D *ptreco_ptgen_u = f->Get<TH2D>("ptreco_ptgen_u");
    TH2D *fakes_ptreco_mreco = f->Get<TH2D>("fakes_ptreco_mreco");
    TH2D *misses_ptgen_mgen = f->Get<TH2D>("misses_ptgen_mgen");
    //make arrays to store bin centers: 
    //    Double_t ptreco_center_d[10];
//     Double_t resp_u[10][11][10][21];
//     TTree *response = (TTree*)f->Get("response");
//     response->SetBranchAddress("ungroomed", &resp_u);
//    TTree *centers = (TTree*)f->Get("centers");
/*     centers->SetBranchAddress("ptreco", &ptreco_center_d); */
/*     centers->GetEntry(0); */
/*     for(int i=0; i<10; i++){ */
/*       cout<< ptreco_center_d[i]<<endl; */
/*     } */
    // Create a TTreeReader for the tree by passing the TTree's name and the TDirectory / TFile it is in.
    TTreeReader matrixReader("response", f);
    TTreeReader binReader("centers", f);
    // The branch "reco_gen_groomed" contains doubles; access them as response_matrix_g
    TTreeReaderArray<Double_t> response_matrix_g(matrixReader, "groomed");
    // The branch "reco_gen_ungroomed" contains doubles; access them as response_matrix_u
    TTreeReaderArray<Double_t> response_matrix_u(matrixReader, "ungroomed");
    // Make bin arrays
    TTreeReaderArray<Double_t> ptreco_center(binReader, "ptreco");
    TTreeReaderArray<Double_t> mreco_center(binReader, "mreco");
    TTreeReaderArray<Double_t> ptgen_center(binReader, "ptgen");
    TTreeReaderArray<Double_t> mgen_center(binReader, "mgen");

    //check content by drawing hists
    TCanvas *c1 = new TCanvas("c1","Trying to plot things",1200,600);
    c1->Divide(2,1);
    c1->SetLogz();
    c1->cd(1);
    c1->SetLogz();
    mreco_mgen_u->Draw("colz");
    c1->cd(2);
    c1->SetLogz();
    ptreco_ptgen_u->Draw("colz");

    
    //reconstructed mass and pt
    int NBin_mass_fine = mreco_mgen_u->GetXaxis()->GetNbins();
    int NBin_pt_fine = ptreco_ptgen_u->GetXaxis()->GetNbins();
    cout << "Number of reco mass bins: "<< NBin_mass_fine << " and number of reco pt bins: " << NBin_pt_fine << endl;
    //generated mass and pt
    int NBin_mass_coarse = mreco_mgen_u->GetYaxis()->GetNbins();
    int NBin_pt_coarse = ptreco_ptgen_u->GetYaxis()->GetNbins();
    cout << "Number of gen mass bins: "<< NBin_mass_coarse << " and number of gen pt bins: " << NBin_pt_coarse << endl;
    
//     double_t massBins_fine[NBin_mass_fine];
    
//     for(Int_t i=2; i<= NBin_mass_fine+1; i++){
//         cout << "Bin number " << i << endl;
//         massBins_fine[i-1] = mreco_mgen_u->GetXaxis()->GetBinLowEdge(i);
//         cout << "Bin edge " << massBins_fine[i-1] << endl;;
//     }
//     for(Int_t i=2; i<= NBin_mass_fine+1; i++){
//         cout << "Bin number " << i << endl;
//         massBins_fine[i-1] = mreco_mgen_u->GetXaxis()->GetBinLowEdge(i);
//         cout << "Bin edge " << massBins_fine[i-1] << endl;;
//     }
    TUnfoldBinning *detectorBinning = new TUnfoldBinning("detector");
    TUnfoldBinning *missBinning = detectorBinning->AddBinning("missesBin", 1);
    TUnfoldBinning *recoBinning = detectorBinning->AddBinning("reco");
    recoBinning->AddAxis(*mreco_mgen_u->GetXaxis(), false, true);
    recoBinning->AddAxis(*ptreco_ptgen_u->GetXaxis(), false, true);
    cout << detectorBinning << endl;
    
    TUnfoldBinning *generatorBinning = new TUnfoldBinning("generator");
    TUnfoldBinning *fakeBinning=generatorBinning->AddBinning("fakesBin", 1);
    TUnfoldBinning *genBinning = generatorBinning->AddBinning("gen");
    genBinning->AddAxis(*mreco_mgen_u->GetYaxis(), false, true);
    genBinning->AddAxis(*ptreco_ptgen_u->GetYaxis(), false, true);
    cout << &generatorBinning << endl;
    
    cout << "Test binning for ptreco = 250, mreco = 500 " <<recoBinning->GetGlobalBinNumber(250,500) << endl;
    
    TH2D *histMCGenRec=TUnfoldBinning::CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec");
    TH1 *histMCGen =   generatorBinning->CreateHistogram("histMCGen");
    TH1 *histMCReco =   detectorBinning->CreateHistogram("histMCReco");
    int global_bin_ind = 0;



    // Loop through reco and gen bins and fill Response matrix    
while(matrixReader.Next() && binReader.Next()) {
        cout << "Type of groomed: " << typeid(response_matrix_g).name() << " and size: " << response_matrix_u.GetSize() << endl;
        cout << "Size of ptreco bins: " << ptreco_center.GetSize()<< endl;
        cout << "Size of mreco bins: " << mreco_center.GetSize()<< endl;
        cout << "Size of mgen bins: " << mgen_center.GetSize()<< endl;
        // i is ptreco, j is mreco, k is ptgen, l is mgen
        for(int i=0; i<ptreco_center.GetSize(); i++){
             for(int j=0; j<mreco_center.GetSize(); j++){
                for(int k=0; k<ptgen_center.GetSize(); k++){
		  for(int l=0; l<mgen_center.GetSize(); l++){
     		    Int_t glob_recobin = i*mreco_center.GetSize()+j;
		    Int_t glob_genbin = k*mgen_center.GetSize()+l;
		    Int_t glob_bin = glob_recobin*(mgen_center.GetSize()*ptgen_center.GetSize())+glob_genbin;
		    cout<<"Global bin " << glob_bin << " for reco bin " << glob_recobin << " and gen bin " << glob_genbin << endl;
		    cout<<"has centers mreco " << mreco_center[j] << " and ptreco " << ptreco_center[i] << endl;
		    cout<<"has centers mgen " << mgen_center[l] << " and ptreco " << ptgen_center[k] << endl;
		    Int_t genBin=genBinning->GetGlobalBinNumber(mgen_center[l],ptgen_center[k]);
		    Int_t recoBin=recoBinning->GetGlobalBinNumber(mreco_center[j],ptreco_center[i]);
		    cout <<"TUnfold gen bin "<< genBin << " and reco bin " << recoBin<< "for value" << response_matrix_u[glob_bin] <<endl;
		    Double_t resp_weight_u = response_matrix_u[glob_bin];
		    histMCGenRec->Fill(genBin,recoBin,resp_weight_u);
          	    //only fill fakes in fake genBin
		    Double_t fake_weight = fakes_ptreco_mreco->GetBinContent(i,j);
		    cout << "Fake weight " << fake_weight << " for i == " << i << " and j == " << j << endl;
		    Int_t fakeBin=fakeBinning->GetStartBin();
		    histMCGenRec->SetBinContent(fakeBin,recoBin,fake_weight);
                    //only fill misses in miss recoBin                                                                                                                                                             
                    Double_t miss_weight = misses_ptgen_mgen->GetBinContent(k,l);
		    cout << "Miss weight " << miss_weight << " for k == " << k << " and k == " << k << endl;
		    Int_t missBin=missBinning->GetStartBin();
		    histMCGenRec->SetBinContent(genBin,missBin,miss_weight);
		    //fill rest of truth and 
                    histMCGen->Fill(genBin, resp_weight_u);
		    histMCReco->Fill(recoBin, resp_weight_u);
		    global_bin_ind++;
		  }}}}
        cout << response_matrix_u[849] <<endl;              
    }
    
    // check that response matrix has been filled properly
    //TH1 *histMCReco=histMCGenRec->ProjectionY("histMCReco",0,-1);
    TH1 *histMCTruth=histMCGenRec->ProjectionX("histMCTruth",0,-1);
    TCanvas *c2 = new TCanvas("c2","Plot full responses",1200,400);
    c2->Divide(3,1);
    c2->cd(1);
    histMCReco->SetLineColor(kBlue);
    histMCReco->Draw("E");
    c2->cd(2);
    //    gPad->SetLogy();
    histMCGen->SetLineColor(kBlue);
    histMCGen->Draw("E");
    /* histMCTruth->SetLineColor(kRed); */
    /* histMCTruth->Draw("E"); */
    c2->cd(3);
    c2->SetLogz();
    histMCGenRec->Draw("colz");
    
    cout << "Reco_bins" << global_bin_ind << endl;
    
    
     //=======================================================
    // open file to save histograms and binning schemes

    TFile *outputFile=new TFile("testUnfold_histograms.root","recreate");
    
    detectorBinning->Write();
    generatorBinning->Write();
    
    detectorBinning->PrintStream(cout);
    generatorBinning->PrintStream(cout);
    
     //=======================================================
  // Step 4: book and fill histogram of migrations
  //         it receives events from both signal MC and background MC
    
    
}


