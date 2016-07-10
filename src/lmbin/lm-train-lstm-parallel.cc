// lm/lm-train-lstm-parallel.cc

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

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "lm/lm-compute-lstm-parallel.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  using namespace kaldi::lm;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of Neural Network training by mini-batch Stochastic Gradient Descent.\n"
        "This version use for lstm language model training.\n"
        "Usage:  nnet-train-lstm-lm-parallel [options] <feature-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " lm-train-lstm-parallel scp:feature.scp nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);

    NnetParallelOptions parallel_opts;
    parallel_opts.Register(&po);

    LstmlmUpdateOptions opts(&trn_opts, &rnd_opts, &parallel_opts);
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3-(opts.crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      model_filename = po.GetArg(2);
        
    std::string target_model_filename;
    if (!opts.crossvalidate) {
      target_model_filename = po.GetArg(3);
    }

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().Initialize();
    //CuDevice::Instantiate().DisableCaching();
#endif


    Nnet nnet;
    LmStats stats;

    Timer time;
    double time_now = 0;
    KALDI_LOG << "TRAINING STARTED";


    LstmlmUpdateParallel(&opts,
					model_filename,
					feature_rspecifier,
								&nnet,
								&stats);


    if (!opts.crossvalidate) {
      nnet.Write(target_model_filename, opts.binary);
    }

    KALDI_LOG << "TRAINING FINISHED; ";
    time_now = time.Elapsed();

    stats.Print(&opts, time_now);

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
