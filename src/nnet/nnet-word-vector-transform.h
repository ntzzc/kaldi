	// nnet/nnet-word-vector-transform.h

// Copyright 2011-2014  Brno University of Technology (author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#ifndef KALDI_NNET_WORD_VECTOR_TRANSFORM_H_
#define KALDI_NNET_WORD_VECTOR_TRANSFORM_H_


#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"


namespace kaldi {
namespace nnet1 {

class WordVectorTransform : public UpdatableComponent {

	friend class NnetModelSync;

 public:
	WordVectorTransform(int32 dim_in, int32 dim_out)
    : UpdatableComponent(dim_in, dim_out), 
	  learn_rate_coef_(1.0)
  { }
  ~WordVectorTransform()
  { }

  Component* Copy() const { return new WordVectorTransform(*this); }
  ComponentType GetType() const { return kWordVectorTransform; }
  
  void InitData(std::istream &is) {
    // define options
    float param_stddev = 0.1, param_range = 0.0;
    float learn_rate_coef = 1.0;
    int32 vocab_size = 0;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<ParamRange>")   ReadBasicType(is, false, &param_range);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
      else if (token == "<VocabSize>") ReadBasicType(is, false, &vocab_size);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|LearnRateCoef|BiasLearnRateCoef)";
      is >> std::ws; // eat-up whitespace
    }

    KALDI_ASSERT(vocab_size > 0);

    //
    learn_rate_coef_ = learn_rate_coef;
    vocab_size_ = vocab_size; // wordvector length: output_dim_
    //

    //
    // initialize
    //
    Matrix<BaseFloat> mat(vocab_size_, output_dim_);
    for (int32 r=0; r<vocab_size_; r++) {
      for (int32 c=0; c<output_dim_; c++) {
        if (param_range == 0.0)
        	mat(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
        else
        	mat(r,c) = param_range * (RandUniform() - 0.5) * 2;
      }
    }
    wordvector_ = mat;

  }

  void ReadData(std::istream &is, bool binary) {
    // optional learning-rate coefs
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<VocabSize>");
      ReadBasicType(is, binary, &vocab_size_);
      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &learn_rate_coef_);
    }
    // weights
    wordvector_.Read(is, binary);

    KALDI_ASSERT(wordvector_.NumRows() == vocab_size_);
    KALDI_ASSERT(wordvector_.NumCols() == output_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
	WriteToken(os, binary, "<VocabSize>");
	WriteBasicType(os, binary, vocab_size_);
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    // weights
    wordvector_.Write(os, binary);
  }

  int32 NumParams() const { return wordvector_.NumRows()*wordvector_.NumCols(); }
  
  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());
    int32 wordvector_num_elem = wordvector_.NumRows() * wordvector_.NumCols();
    wei_copy->Range(0,wordvector_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(wordvector_));
  }
  
  std::string Info() const {
    return std::string("\n  wordvector") + MomentStatistics(wordvector_);
  }

  std::string InfoGradient() const {
    return std::string("\n  wordvector_grad") + MomentStatistics(wordvector_corr_) +
           ", lr-coef " + ToString(learn_rate_coef_);
  }


  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
	wordid_.Resize(in.NumRows(), kUndefined);
	CuMatrix<BaseFloat> tmp(1, in.NumRows());
	tmp.CopyFromMat(in, kTrans);
	tmp.CopyRowToVecId(wordid_);
	out->CopyRows(wordvector_, wordid_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // multiply error derivative by weights
    //in_diff->AddMatMat(1.0, out_diff, kNoTrans, linearity_, kNoTrans, 0.0);
  }

  void Gradient(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff)
  {
	    // we use following hyperparameters from the option class
	    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
	    //const BaseFloat mmt = opts_.momentum;
	    //const BaseFloat l2 = opts_.l2_penalty;
	    //const BaseFloat l1 = opts_.l1_penalty;
	    // we will also need the number of frames in the mini-batch
	    //const int32 num_frames = input.NumRows();
		local_lrate = -lr;

	    // compute gradient (incl. momentum)
		wordvector_corr_ = diff;
  }

  void UpdateGradient()
  {
	    // update
	  	wordvector_.AddMatToRows(local_lrate, wordvector_corr_, wordid_);
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    const BaseFloat mmt = opts_.momentum;
    const BaseFloat l2 = opts_.l2_penalty;
    const BaseFloat l1 = opts_.l1_penalty;
    // we will also need the number of frames in the mini-batch
    const int32 num_frames = input.NumRows();
    // compute gradient (incl. momentum)

    wordvector_corr_ = diff;
    wordvector_.AddMatToRows(lr, wordvector_corr_, wordid_);
  }

  const CuMatrixBase<BaseFloat>& GetLinearity() const {
    return wordvector_;
  }

  void SetLinearity(const CuMatrixBase<BaseFloat>& wordvector) {
    KALDI_ASSERT(wordvector_.NumRows() == wordvector.NumRows());
    KALDI_ASSERT(wordvector_.NumCols() == wordvector.NumCols());
    wordvector_.CopyFromMat(wordvector);
  }

  const CuMatrixBase<BaseFloat>& GetLinearityCorr() const {
    return wordvector_corr_;
  }

protected:

  CuMatrix<BaseFloat> wordvector_;

  CuMatrix<BaseFloat> wordvector_corr_;

  CuArray<MatrixIndexT> wordid_;

  BaseFloat learn_rate_coef_;

  BaseFloat local_lrate;

  int32 vocab_size_;

};

} // namespace nnet1
} // namespace kaldi

#endif
