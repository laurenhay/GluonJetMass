#include <iostream>
#include <fstream>
#include <TFile.h>
#include <TH2D.h>
#include <TUnfoldBinning.h>



// this file imports root histograms made with coffea and created a TUnfold binning object

using namespace std;

void testUnfold()
{
    
    //open root file from coffea/uproot and get hists
    TFile *uprootFile = TFile::Open("trijetHistsQCDsim.root");
    TH2D *mreco_mgen_g = uprootFile->Get<TH2D>("mreco_mgen_g");
    TH2D *ptreco_ptgen_g = uprootFile->Get<TH2D>("ptreco_ptgen_g");
    TH2D *mreco_mgen_u = uprootFile->Get<TH2D>("mreco_mgen_u");
    TH2D *ptreco_ptgen_u = uprootFile->Get<TH2D>("ptreco_ptgen_u");
    
    //check content by drawing hists
    TCanvas *c1 = new TCanvas("c1","Trying to plot things",1200,600);
    c1->Divide(2,1);
    c1->cd(1);
    c1->SetLogy();
    c1->SetLogx();
    mreco_mgen_u->Draw("colz");
    c1->cd(2);
    c1->SetLogy();
    c1->SetLogx();
    ptreco_ptgen_u->Draw("colz");
    
//     //reconstructed mass and pt
//     int NBin_mass_fine = mreco_mgen_u->GetXaxis()->GetNbins();
//     int NBin_pt_fine = ptreco_ptgen_u->GetXaxis()->GetNbins();
//     cout << "Number of reco mass bins: "<< NBin_mass_fine << " and number of reco pt bins: " << NBin_pt_fine << endl;
//     //generated mass and pt
//     int NBin_mass_coarse = mreco_mgen_u->GetYaxis()->GetNbins();
//     int NBin_pt_coarse = ptreco_ptgen_u->GetYaxis()->GetNbins();
//     cout << "Number of gen mass bins: "<< NBin_mass_coarse << " and number of gen pt bins: " << NBin_pt_coarse << endl;
    
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
    TUnfoldBinning *recoBins = new TUnfoldBinning(*mreco_mgen_u->GetXaxis(), false, false);
    recoBins->AddAxis(*ptreco_ptgen_u->GetXaxis(), false, false);
    cout << recoBins << endl;
    
    TUnfoldBinning *genBins = new TUnfoldBinning(*mreco_mgen_u->GetYaxis(), false, false);
    genBins->AddAxis(*mreco_mgen_u->GetYaxis(), false, false);
    cout << &genBins << endl;
    
     //=======================================================
    // open file to save histograms and binning schemes

    TFile *outputFile=new TFile("testUnfold5_histograms.root","recreate");
    
    recoBins->Write();
    genBins->Write();
    
    recoBins->PrintStream(cout);
    genBin->PrintStream(cout);
    
     //=======================================================
  // Step 4: book and fill histogram of migrations
  //         it receives events from both signal MC and background MC
    
    
}


