// nnet/nnet-parallel-component-multitask.h

// Copyright 2014  Brno University of Technology (Author: Karel Vesely)
// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

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


#ifndef KALDI_NNET_NNET_PARALLEL_COMPONENT_MULTITASK_H_
#define KALDI_NNET_NNET_PARALLEL_COMPONENT_MULTITASK_H_


#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

#include <sstream>

namespace kaldi {
namespace nnet1 {

class ParallelComponentMultiTask : public UpdatableComponent {
 public:
	ParallelComponentMultiTask(int32 dim_in, int32 dim_out)
    : UpdatableComponent(dim_in, dim_out)
  { }
  ~ParallelComponentMultiTask()
  { }

  Component* Copy() const { return new ParallelComponentMultiTask(*this); }
  ComponentType GetType() const { return kParallelComponentMultiTask; }

  void InitData(std::istream &is) {
    // define options
    std::vector<std::string> nested_nnet_proto;
    std::vector<std::string> nested_nnet_filename;
    // parse config
    std::string token; 
	int32 offset, len = 0;
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<NestedNnet>" || token == "<NestedNnetFilename>") {
        while(!is.eof()) {
		  ExpectToken(is, false, "<InputOffset>");
		  ReadBasicType(is, false, &offset);
		  input_offset.push_back(std::pair<int32, int32>(offset, len));

		  ExpectToken(is, false, "<OutputOffset>");
		  ReadBasicType(is, false, &offset);
		  output_offset.push_back(std::pair<int32, int32>(offset, len));

          std::string file_or_end;
          ReadToken(is, false, &file_or_end);
          if (file_or_end == "</NestedNnet>" || file_or_end == "</NestedNnetFilename>") break;
          nested_nnet_filename.push_back(file_or_end);
        }
      } else if (token == "<NestedNnetProto>") {
        while(!is.eof()) {
      	  ExpectToken(is, false, "<InputOffset>");
      	  ReadBasicType(is, false, &offset);
      	  input_offset.push_back(std::pair<int32, int32>(offset, len));

      	  ExpectToken(is, false, "<OutputOffset>");
      	  ReadBasicType(is, false, &offset);
      	  output_offset.push_back(std::pair<int32, int32>(offset, len));

          std::string file_or_end;
          ReadToken(is, false, &file_or_end);
          if (file_or_end == "</NestedNnetProto>") break;
          nested_nnet_proto.push_back(file_or_end);
        }
      } else KALDI_ERR << "Unknown token " << token << ", typo in config?"
                       << " (NestedNnet|NestedNnetFilename|NestedNnetProto)";
      is >> std::ws; // eat-up whitespace
    }
    // initialize
    KALDI_ASSERT((nested_nnet_proto.size() > 0) ^ (nested_nnet_filename.size() > 0)); //xor
    // read nnets from files
    if (nested_nnet_filename.size() > 0) {
      for (int32 i=0; i<nested_nnet_filename.size(); i++) {
        Nnet nnet;
        nnet.Read(nested_nnet_filename[i]);
        nnet_.push_back(nnet);
        input_offset[i].second = nnet.InputDim();
        output_offset[i].second = nnet.OutputDim();
        KALDI_LOG << "Loaded nested <Nnet> from file : " << nested_nnet_filename[i];
      }
    }
    // initialize nnets from prototypes
    if (nested_nnet_proto.size() > 0) {
      for (int32 i=0; i<nested_nnet_proto.size(); i++) {
        Nnet nnet;
        nnet.Init(nested_nnet_proto[i]);
        nnet_.push_back(nnet);
        input_offset[i].second = nnet.InputDim();
        output_offset[i].second = nnet.OutputDim();
        KALDI_LOG << "Initialized nested <Nnet> from prototype : " << nested_nnet_proto[i];
      }
    }

    // check dim-sum of nested nnets
    check();
  }

  void ReadData(std::istream &is, bool binary) {
    // read
    ExpectToken(is, binary, "<NestedNnetCount>");
    std::pair<int32, int32> offset;
    int32 nnet_count;
    ReadBasicType(is, binary, &nnet_count);
    for (int32 i=0; i<nnet_count; i++) {
      ExpectToken(is, binary, "<NestedNnet>");
      int32 dummy;
      ReadBasicType(is, binary, &dummy);

      ExpectToken(is, binary, "<InputOffset>");
      ReadBasicType(is, binary, &offset.first);
      input_offset.push_back(offset);

      ExpectToken(is, binary, "<OutputOffset>");
      ReadBasicType(is, binary, &offset.first);
      output_offset.push_back(offset);

      Nnet nnet;
      nnet.Read(is, binary);
      nnet_.push_back(nnet);
      input_offset[i].second = nnet.InputDim();
      output_offset[i].second = nnet.OutputDim();
    }
    ExpectToken(is, binary, "</ParallelComponentMultiTask>");

    // check dim-sum of nested nnets
    check();
  }

  void WriteData(std::ostream &os, bool binary) const {
    // useful dims
    int32 nnet_count = nnet_.size();
    //
    WriteToken(os, binary, "<NestedNnetCount>");
    WriteBasicType(os, binary, nnet_count);
    for (int32 i=0; i<nnet_count; i++) {
      WriteToken(os, binary, "<NestedNnet>");
      WriteBasicType(os, binary, i+1);

      WriteToken(os, binary, "<InputOffset>");
      WriteBasicType(os, binary, input_offset[i].first);

      WriteToken(os, binary, "<OutputOffset>");
      WriteBasicType(os, binary, output_offset[i].first);

      nnet_[i].Write(os, binary);
    }
    WriteToken(os, binary, "</ParallelComponentMultiTask>");
  }

  int32 NumParams() const { 
    int32 num_params_sum = 0;
    for (int32 i=0; i<nnet_.size(); i++) 
      num_params_sum += nnet_[i].NumParams();
    return num_params_sum;
  }

  void GetParams(Vector<BaseFloat>* wei_copy) const { 
    wei_copy->Resize(NumParams());
    int32 offset = 0;
    for (int32 i=0; i<nnet_.size(); i++) {
      Vector<BaseFloat> wei_aux;
      nnet_[i].GetParams(&wei_aux);
      wei_copy->Range(offset, wei_aux.Dim()).CopyFromVec(wei_aux);
      offset += wei_aux.Dim();
    }
    KALDI_ASSERT(offset == NumParams());
  }
    
  std::string Info() const { 
    std::ostringstream os;
    for (int32 i=0; i<nnet_.size(); i++) {
      os << "nested_network #" << i+1 << "{\n" << nnet_[i].Info() << "}\n";
    }
    std::string s(os.str());
    s.erase(s.end() -1); // removing last '\n'
    return s;
  }
                       
  std::string InfoGradient() const {
    std::ostringstream os;
    for (int32 i=0; i<nnet_.size(); i++) {
      os << "nested_gradient #" << i+1 << "{\n" << nnet_[i].InfoGradient() << "}\n";
    }
    std::string s(os.str());
    s.erase(s.end() -1); // removing last '\n'
    return s;
  }

  std::string InfoPropagate() const {
    std::ostringstream os;
    for (int32 i=0; i<nnet_.size(); i++) {
      os << "nested_propagate #" << i+1 << "{\n" << nnet_[i].InfoPropagate() << "}\n";
    }
    return os.str();
  }

  std::string InfoBackPropagate() const {
    std::ostringstream os;
    for (int32 i=0; i<nnet_.size(); i++) {
      os << "nested_backpropagate #" << i+1 << "{\n" << nnet_[i].InfoBackPropagate() << "}\n";
    }
    return os.str();
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {

    for (int32 i=0; i<nnet_.size(); i++) {
      CuSubMatrix<BaseFloat> src(in.ColRange(input_offset[i].first, input_offset[i].second));
      CuSubMatrix<BaseFloat> tgt(out->ColRange(output_offset[i].first, output_offset[i].second));
      //
      nnet_[i].Propagate(src, &tgt);
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
	in_diff->SetZero();
    for (int32 i=0; i<nnet_.size(); i++) {
      CuSubMatrix<BaseFloat> src(out_diff.ColRange(output_offset[i].first, output_offset[i].second));
      CuSubMatrix<BaseFloat> tgt(in_diff->ColRange(input_offset[i].first, input_offset[i].second));
      // 
      CuMatrix<BaseFloat> tgt_aux;
      nnet_[i].Backpropagate(src, &tgt_aux);
      tgt.AddMat(1.0, tgt_aux);
    }
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    ; // do nothing
  }
 
  void SetTrainOptions(const NnetTrainOptions &opts) {
    for (int32 i=0; i<nnet_.size(); i++) {
      nnet_[i].SetTrainOptions(opts);
    }
  }

  int32 GetDim() const
  {
	  int32 dim = 0;
	  for (int i = 0; i < nnet_.size(); i++)
		  dim += nnet_[i].GetDim();
	  return dim;
  }

  int WeightCopy(void *host, int direction, int copykind)
  {
	  int pos = 0;
	  for (int i = 0; i < nnet_.size(); i++)
		  pos += nnet_[i].WeightCopy((void*)((char *)host+pos), direction, copykind);
	  return pos;
  }

  Component* GetComponent(Component::ComponentType type)
  {
	  Component *com = NULL;
	  for (int i = 0; i < nnet_.size(); i++)
	  {
		  for (int32 c = 0; c < nnet_[i].NumComponents(); c++)
		  {

			  if (nnet_[i].GetComponent(c).GetType() == type)
				  return com;
			  else if (nnet_[i].GetComponent(c).GetType() == Component::kParallelComponentMultiTask)
			  {
				  com = (dynamic_cast<ParallelComponentMultiTask&>(nnet_[i].GetComponent(c))).GetComponent(type);
				  if (com != NULL) return com;
			  }
			  else
				  return com;
		  }
	  }
	  return com;
  }

  std::vector<std::pair<int32, int32> >	GetOutputOffset()
  {
	  return output_offset;
  }

 private:
  void check()
  {
	    // check dim-sum of nested nnets
	    int32 nnet_input_max = 0, nnet_output_max = 0, dim = 0;
	    for (int32 i=0; i<nnet_.size(); i++) {
	    	dim = input_offset[i].first + input_offset[i].second;
	    	if (nnet_input_max < dim) nnet_input_max = dim;

	    	dim = output_offset[i].first + output_offset[i].second;
	    	if (nnet_output_max < dim) nnet_output_max = dim;
	    }
	    KALDI_ASSERT(InputDim() == nnet_input_max);
	    KALDI_ASSERT(OutputDim() == nnet_output_max);
  }

 private:
  std::vector<Nnet> nnet_;
  std::vector<std::pair<int32, int32> > input_offset;  // <offset, length>
  std::vector<std::pair<int32, int32> > output_offset;
};

} // namespace nnet1
} // namespace kaldi

#endif
