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
    //make arrays to store bin centers: 
//     Double_t ptreco_center_d[10];
//     Double_t resp_u[10][11][10][21];
//     TTree *response = (TTree*)f->Get("response");
//     response->SetBranchAddress("ungroomed", &resp_u);
//     TTree *centers = (TTree*)f->Get("centers");
//     centers->SetBranchAddress("ptreco", &ptreco_center_d);
//     cout << "Tarray success?? "<< resp_u[0][4][0][9] << endl;
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
    vector<Double_t> ptreco_center_vec;
    //Double_t mreco_center_d[mreco_center.GetSize()], ptgen_center_d[ptgen_center.GetSize()], mgen_center_d[mgen_center.GetSize()];
    cout << "Seg fault?" << endl;
    while(binReader.Next()) {
        for(int i=0;i<ptreco_center.GetSize();i++){
            ptreco_center_vec.push_back(ptreco_center[i]);
            cout<<"PTreco entry "<< i<< " " << ptreco_center_vec[i] << endl;
        }
//         for(int i=0;i<mreco_center.GetSize();i++){
//             mreco_center_d[i]=mreco_center[i];
//             cout<<"Mreco entry "<< i<< " " << mreco_center_d[i] << endl;
//         }
//         for(int i=0;i<ptgen_center.GetSize();i++){
//             ptgen_center_d[i]=ptgen_center[i];
//             cout<<"PTgen entry "<< i<< " " << ptgen_center_d[i] << endl;
//         }
//         for(int i=0;i<mgen_center.GetSize();i++){
//             mgen_center_d[i]=mgen_center[i];
//             cout<<"Mgebn entry "<< i<< " " << mgen_center_d[i] << endl;
//         }
    }
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
    TUnfoldBinning *detectorBinning = new TUnfoldBinning(*mreco_mgen_u->GetXaxis(), true, true);
    detectorBinning->AddAxis(*ptreco_ptgen_u->GetXaxis(), true, true);
    cout << detectorBinning << endl;
    
    TUnfoldBinning *generatorBinning = new TUnfoldBinning(*mreco_mgen_u->GetYaxis(), true, true);
    generatorBinning->AddAxis(*ptreco_ptgen_u->GetYaxis(), true, true);
    cout << &generatorBinning << endl;
    
    cout << "Test binning for ptreco = 250, mreco = 500 " <<detectorBinning->GetGlobalBinNumber(250,500) << endl;
    
    TH2D *histMCGenRec=TUnfoldBinning::CreateHistogramOfMigrations(generatorBinning,detectorBinning,"histMCGenRec");
    
    int global_bin_ind = 0;
    
    while(matrixReader.Next() & binReader.Next()) {
        cout << "Type of groomed: " << typeid(response_matrix_g).name() << " and size: " << response_matrix_u.GetSize() << endl;
        cout << "Size of ptreco bins: " << ptreco_center.GetSize()<< endl;
        cout << "Size of mreco bins: " << mreco_center.GetSize()<< endl;
        cout << "Size of mgen bins: " << mgen_center.GetSize()<< endl;
        // i is ptreco, j is mreco, k is ptgen, l is mgen
        for(int i=0; i<ptreco_center.GetSize(); i++){
//             for(int j=0; j<mreco_center.GetSize(); j++){
            for(int j=0; j<1; j++){            
                for(int k=0; k<ptgen_center.GetSize(); k++){
//                     for(int l=0; l<mgen_center.GetSize(); l++){
                    for(int l=0; l<1; l++){
                        Int_t glob_recobin = i*mreco_center.GetSize()+j;
                        Int_t glob_genbin = l*mgen_center.GetSize()+k;
                        Int_t glob_bin = glob_recobin*(mgen_center.GetSize()*ptgen_center.GetSize())+glob_genbin;
                        cout<<"Global bin " << glob_bin << " for reco bin " << glob_recobin << " and gen bin " << glob_genbin << endl;
                        cout<<"has centers mreco " << mreco_center[j] << " and ptreco " << ptreco_center[i] << glob_genbin << endl;
                        cout<<"has centers mgen " << mgen_center[l] << " and ptreco " << ptgen_center[k] << glob_genbin << endl;
                        Int_t genBin=generatorBinning->GetGlobalBinNumber(mgen_center[l],ptgen_center[k]);
                        Int_t recoBin=detectorBinning->GetGlobalBinNumber(mreco_center[j],ptreco_center[i]);
                        cout <<"TUnfold gen bin "<< genBin << " and reco bin " << recoBin<<endl;
                        Double_t resp_weight_u = response_matrix_u[glob_bin];
                        histMCGenRec->Fill(genBin,recoBin,resp_weight_u);
                        global_bin_ind++;
                    }}}}
        cout << response_matrix_u[849] <<endl;              
    }
    
    // check that response matrix has been filled properly
    TH1 *histMCReco=histMCGenRec->ProjectionY("histMCReco",0,-1);
    TH1 *histMCTruth=histMCGenRec->ProjectionX("histMCTruth",0,-1);
    TCanvas *c2 = new TCanvas("c2","Plot full responses",1200,400);
    c2->Divide(3,1);
    c2->cd(1);
    histMCReco->SetLineColor(kBlue);
    histMCReco->Draw("E");
    c2->cd(2);
    gPad->SetLogy();
    histMCTruth->SetLineColor(kRed);
    histMCTruth->Draw("E");
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


