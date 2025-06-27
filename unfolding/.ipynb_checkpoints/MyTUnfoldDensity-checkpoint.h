// Stolen from Robin : https://github.com/raggleton/QGAnalysisPlotting/blob/9df0da71887093d4d6d29ea433f14c0b48a9efc5/MyTUnfoldDensity.cpp
// 
// Hacky code to access protected methods
// This converts protected methods/vars into public ones, so that they can be
// used in PyROOT (which can't see protected things)
// 
// IMPORTANT: don't start with a block comment...wont work
#include <TH1.h>
#include <TH2.h>
#include <TUnfoldDensity.h>

class MyTUnfoldDensity : public TUnfoldDensity {
public:
    MyTUnfoldDensity(const TH2 *hist_A,
                     EHistMap histmap,
                     ERegMode regmode = kRegModeCurvature,
                     EConstraint constraint=kEConstraintArea,
                     EDensityMode densityMode=kDensityModeBinWidthAndUser,
                     const TUnfoldBinning *outputBins=0,
                     const TUnfoldBinning *inputBins=0,
                     const char *regularisationDistribution=0,
                     const char *regularisationAxisSteering="*[UOB]"
		     ) :
        
        TUnfoldDensity(hist_A,
                       histmap,
                       regmode,
                       constraint,
                       densityMode,
                       outputBins,
                       inputBins,
                       regularisationDistribution,
                       regularisationAxisSteering) {
       std::cout << "Setup (My)TUnfoldDensity" << std::endl;
    }

    // Here are methods that were protected in TUnfoldDensity, that are now public for us to use
 
    // multiply sparse and non-sparse matrix
    TMatrixDSparse *MultiplyMSparseM(const TMatrixDSparse *a,const TMatrixD *b) const { return TUnfold::MultiplyMSparseM(a, b); } 
    // multiply sparse and sparse matrix
    TMatrixDSparse *MultiplyMSparseMSparse(const TMatrixDSparse *a,const TMatrixDSparse *b) const { return TUnfold::MultiplyMSparseMSparse(a, b); } 
    // multiply transposed sparse and sparse matrix
    TMatrixDSparse *MultiplyMSparseTranspMSparse(const TMatrixDSparse *a,const TMatrixDSparse *b) const { return TUnfold::MultiplyMSparseTranspMSparse(a, b); } 
    // calculate M_ij = sum_k [m1_ik*m2_jk*v[k] ]. the pointer v may be zero (means no scaling).
    TMatrixDSparse *MultiplyMSparseMSparseTranspVector (const TMatrixDSparse *m1,const TMatrixDSparse *m2, const TMatrixTBase<Double_t> *v) const { return TUnfold::MultiplyMSparseMSparseTranspVector(m1, m2, v); } 
    // invert symmetric (semi-)positive sparse matrix
    TMatrixDSparse *InvertMSparseSymmPos(const TMatrixDSparse *A, bool doPseudoInverse) const { 
        if (doPseudoInverse) { 
            Int_t rank = 1; 
            return TUnfold::InvertMSparseSymmPos(A, &rank); 
        } else { 
            return TUnfold::InvertMSparseSymmPos(A, nullptr); 
        } 
    }
    // replacement for dest += f*src
    void AddMSparse(TMatrixDSparse *dest,Double_t f,const TMatrixDSparse *src) const { return TUnfold::AddMSparse(dest, f, src); } 
    // create a TMatrixDSparse from an array
    TMatrixDSparse *CreateSparseMatrix(Int_t nrow,Int_t ncol,Int_t nele,Int_t *row,Int_t *col,Double_t *data) const { return TUnfold::CreateSparseMatrix(nrow, ncol, nele, row, col, data); } 

    /// vector of the unfolding result
    inline const TMatrixD *GetX(void) const { return TUnfoldDensity::GetX(); }
    /// covariance matrix of the result
    inline const TMatrixDSparse *GetVxx(void) const { return TUnfoldDensity::GetVxx(); }
    /// inverse of covariance matrix of the result
    inline const TMatrixDSparse *GetVxxInv(void) const { return TUnfoldDensity::GetVxxInv(); }
    /// vector of folded-back result
    inline const TMatrixDSparse *GetAx(void) const { return TUnfoldDensity::GetAx(); }
    /// matrix of derivatives dx/dy
    inline const TMatrixDSparse *GetDXDY(void) const { return TUnfoldDensity::GetDXDY(); }
    /// matrix contributions of the derivative dx/dA
    inline const TMatrixDSparse *GetDXDAM(int i) const { return TUnfoldDensity::GetDXDAM(i); }
    /// vector contributions of the derivative dx/dA
    inline const TMatrixDSparse *GetDXDAZ(int i) const { return TUnfoldDensity::GetDXDAZ(i); }
    /// matrix E<sup>-1</sup>, using internal bin counting
    inline const TMatrixDSparse *GetEinv(void) const { return TUnfoldDensity::GetEinv(); }
    /// matrix E, using internal bin counting
    inline const TMatrixDSparse *GetE(void) const { return TUnfoldDensity::GetE(); }
    /// vector of the input data y
    inline const TMatrixD *GetY(void) const { return fY; }
    /// covariance matrix of the data y
    inline const TMatrixDSparse *GetVyy(void) const { return fVyy; }
    /// inverse of covariance matrix of the data y
    inline const TMatrixDSparse *GetVyyInv(void) const { return TUnfoldDensity::GetVyyInv(); }
    /// response matrix A
    inline const TMatrixDSparse *GetA(void) const { return fA; }
    /// mapping array to hist
    inline const TArrayI & GetXToHist(void) const { return fXToHist; }
    /// mapping hist to array
    inline const TArrayI & GetHistToX(void) const { return fHistToX; }
    
   // return an error matrix as histogram
    void ErrorMatrixToHist(TH2 *ematrix, const TMatrixDSparse *emat) const { TUnfold::ErrorMatrixToHist(ematrix, emat, nullptr, true); }


    /// determine total error matrix on the vector Ax
    TMatrixDSparse *GetSummedErrorMatrixYY(void) { return TUnfoldSys::GetSummedErrorMatrixYY(); }
    
    /// determine total error matrix on the vector x
    TMatrixDSparse *GetSummedErrorMatrixXX(void) { return TUnfoldSys::GetSummedErrorMatrixXX(); }

    Bool_t AddRegularisationCondition(Int_t i0, Double_t f0, Int_t i1=-1, Double_t f1=0., Int_t i2=-1, Double_t f2=0.) { return TUnfold::AddRegularisationCondition(i0,f0,i1,f1,i2,f2); } // add regularisation condition for a triplet of output bins
    Bool_t AddRegularisationCondition(Int_t nEle, const Int_t *indices, const Double_t *rowData) { return TUnfold::AddRegularisationCondition(nEle, indices, rowData); } // add a regularisation condition

};

void makeJacobian( TH1 const & v, TMatrixD & Jac){
  auto N = v.Integral(0, v.GetNbinsX()+1);
  for ( unsigned int i = 0; i <= v.GetNbinsX()+1; ++i ) {
    for ( unsigned int j = 0; j <= v.GetNbinsX()+1; ++j ) {
      if ( i == j )
        Jac[i][j] = (N - v.GetBinContent(i)) / N / N;
      else
        Jac[i][j] = -v.GetBinContent(i) / N / N;
    }
  }
}


void makeJacobian2D( TH2 const & v, TMatrixD & Jac){  
  for ( unsigned int i = 0; i <=  v.GetNbinsX() + 1; ++i ){
      for ( unsigned int j = 0; j <= v.GetNbinsY() + 1; ++j ) {          
          // flattened index for v rows
          auto index_1 = j * (v.GetNbinsX()+2) + i;
          auto N = v.Integral(0, v.GetNbinsX()+1, j, j);
          if ( N > 0.0 ){
              for ( unsigned int k = 0; k <= v.GetNbinsX() + 1; ++k ){
                  for ( unsigned int l = 0; l <= v.GetNbinsY() + 1; ++l ){
                      // flattened index for v columns 
                      auto index_2 = l * (v.GetNbinsX() + 2) + k;

                      // The entire matrix is block diagonal. 

                      // for diagonal elements within the blocks. 
                      if ( index_1 == index_2 )
                        Jac[index_1][index_2] = (N - v.GetBinContent(i,j)) / N / N;
                      // for off-diagonal elements within the blocks. 
                      else if (l==j)
                        Jac[index_1][index_2] = -v.GetBinContent(i,j) / N / N;
                      //for off-diagonal elements outside of the blocks.
                      else
                           Jac[index_1][index_2] = 0;
                  }
              }
          }
    }
  }
}


void makeJacobian2Dfrom1D( int const n_x, int const n_y, TH1 const & v, TMatrixD & Jac){  
  for ( unsigned int i = 0; i <=  n_x +1; ++i ){
      for ( unsigned int j = 0; j <= n_y +1; ++j ) {          
          // flattened index for v rows
          auto counter = j * (n_x+2);
          auto index_1 = counter + i;
          auto N = v.Integral(counter, n_x+counter+1);
          if ( N > 0.0 ){
              for ( unsigned int k = 0; k <= n_x + 1; ++k ){
                  for ( unsigned int l = 0; l <= n_y + 1; ++l ){
                      // flattened index for v columns
                      auto index_2 = l * (n_x + 2) + k; 

                      // The entire matrix is block diagonal. 

                      // for diagonal elements within the blocks. 
                      if ( index_1 == index_2 )
                        Jac[index_1][index_2] = (N - v.GetBinContent(i+counter)) / N / N;
                      // for off-diagonal elements within the blocks. 
                      else if (l==j)
                          Jac[index_1][index_2] = -v.GetBinContent(i+counter) / N / N;
                      //for off-diagonals outside of the blocks.
                      else
                        Jac[index_1][index_2] = 0;
                  }
              }
          }
    }
  }
}