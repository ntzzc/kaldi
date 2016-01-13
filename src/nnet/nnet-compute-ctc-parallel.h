// nnet/nnet-compute-ctc-parallel.h

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

#ifndef KALDI_NNET_NNET_COMPUTE_LSTM_H_
#define KALDI_NNET_NNET_COMPUTE_LSTM_H_

#include "nnet2/am-nnet.h"
#include "hmm/transition-model.h"

#include <string>
#include <iomanip>
#include <mpi.h>

#include "nnet-trnopts.h"
#include "nnet/nnet-randomizer.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-model-sync.h"

#include "cudamatrix/cu-device.h"

#include "nnet/nnet-compute-parallel.h"

namespace kaldi {
namespace nnet1 {

struct NnetCtcUpdateOptions : public NnetUpdateOptions {


    int32 num_stream;
    int32 max_frames;
    int32 targets_delay;


    NnetCtcUpdateOptions(const NnetTrainOptions *trn_opts, const NnetDataRandomizerOptions *rnd_opts, const NnetParallelOptions *parallel_opts)
    	: NnetUpdateOptions(trn_opts, rnd_opts, parallel_opts), num_stream(4), max_frames(25000), targets_delay(0) { }

  	  void Register(OptionsItf *po)
  	  {
  		  NnetUpdateOptions::Register(po);

	      po->Register("num-stream", &num_stream, "---CTC--- BPTT multi-stream training");

	      po->Register("max-frames", &max_frames, "Max number of frames to be processed");
		
          po->Register("targets-delay", &targets_delay, "---LSTM--- BPTT targets delay");

  	  }
};


struct NnetCtcStats: NnetStats {

    Ctc ctc;

    NnetCtcStats() { std::memset(this, 0, sizeof(*this)); }

    void MergeStats(NnetCtcUpdateOptions *opts, int root)
    {
    	NnetStats::MergeStats(opts, root);
    	int myid = opts->parallel_opts->myid;

    	if (opts->objective_function == "ctc") {
        		ctc.Merge(myid, 0);
        } else {
        		KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }

    }

    void Print(NnetCtcUpdateOptions *opts, double time_now)
    {
    	NnetStats::Print(opts, time_now);

        if (opts->objective_function == "ctc") {
        	KALDI_LOG << ctc.Report();
        } else {
        	KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }
    }
};


void NnetCtcUpdateParallel(const NnetCtcUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		NnetCtcStats *stats);

void NnetCEUpdateParallel(const NnetCtcUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		NnetStats *stats);

} // namespace nnet1
} // namespace kaldi

#endif // KALDI_NNET_NNET_COMPUTE_LSTM_H_
