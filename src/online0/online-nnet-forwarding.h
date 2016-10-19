// online0/online-nnet-decoding.h

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

#ifndef ONLINE0_ONLINE_NNET_DECODING_H_
#define ONLINE0_ONLINE_NNET_DECODING_H_

#include "thread/kaldi-message-queue.h"
#include "online0/online-message.h"
#include "nnet0/nnet-trnopts.h"
#include "nnet0/nnet-pdf-prior.h"

namespace kaldi {

struct OnlineNnetForwardingOptions {
    typedef nnet0::PdfPriorOptions PdfPriorOptions;
    std::string feature_transform;
    bool no_softmax;
    bool apply_log;
    bool copy_posterior;
    std::string use_gpu;
    int32 num_threads;

    int32 time_shift;
    int32 batch_size;
    int32 num_stream;
    int32 dump_interval;
    int32 skip_frames;
    int32 sweep_time;
    std::string sweep_frames_str;

    const PdfPriorOptions *prior_opts;

    OnlineNnetForwardingOptions(const PdfPriorOptions *prior_opts)
    	:feature_transform(""),no_softmax(true),apply_log(false),copy_posterior(true),use_gpu("no"),num_threads(1),
		 	 	 	 	 	 	 time_shift(0),batch_size(16),num_stream(0),dump_interval(0), skip_frames(1), sweep_time(1), sweep_frames_str("0"), prior_opts(prior_opts)
    {

    }

    void Register(OptionsItf *po)
    {
    	po->Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");
    	po->Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    	po->Register("apply-log", &apply_log, "Transform MLP output to logscale");
    	po->Register("copy-posterior", &copy_posterior, "Copy posterior for skip frames output");
    	po->Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");


    	po->Register("num-threads", &num_threads, "Number of threads(GPUs) to use");


        //<jiayu>
    	po->Register("time-shift", &time_shift, "LSTM : repeat last input frame N-times, discrad N initial output frames.");
        po->Register("batch-size", &batch_size, "---LSTM--- BPTT batch size");
        po->Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training");
        po->Register("dump-interval", &dump_interval, "---LSTM--- num utts between model dumping [ 0 == disabled ]");
        po->Register("skip-frames", &skip_frames, "LSTM model skip frames for next input");
        //</jiayu>

        sweep_time = skip_frames;
        po->Register("sweep-time", &sweep_time, "Sweep times for each utterance in skip frames training(Deprecated, use --sweep-frames instead)");
        po->Register("sweep-frames", &sweep_frames_str, "Sweep frames index for each utterance in skip frames decoding, e.g. 0");
    }

};

}// namespace kaldi

#endif /* ONLINE0_ONLINE_NNET_DECODING_H_ */
