// online0bin/online-nnet-forward.cc

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

#include <limits>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"

#include "online0/online-nnet-forwarding.h"
#include "thread/kaldi-message-queue.h"
#include "nnet0/nnet-nnet.h"

int main(int argc, char *argv[]) {
	  using namespace kaldi;
	  using namespace kaldi::nnet0;
	  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Perform forward pass through Neural Network in online decoding.\n"
        "\n"
        "Usage:  online-nnet-forward [options] <model-in> <mqueue-rspecifier>  \n"
        "e.g.: \n"
        " online-nnet-forward final.nnet forward.mq\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    OnlineNnetForwardingOptions opts(&prior_opts);
    opts.Register(&po);


    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
    		mqueue_rspecifier = po.GetArg(2);

    MessageQueue mq_forward;
    int oflag = O_RDWR | O_CREAT | O_EXCL | O_NONBLOCK;
    struct mq_attr sample_attr;
    sample_attr.mq_maxmsg = MAX_SAMPLE_MQ_MQXMSG;
    sample_attr.mq_msgsize = MAX_SAMPLE_MQ_MSGSIZE;
    mq_forward.Create(mqueue_rspecifier, &sample_attr, oflag);

    //Select the GPU
#if HAVE_CUDA==1
    if (opts.use_gpu == "yes")
    {
        CuDevice::Instantiate().Initialize();
        CuDevice::Instantiate().SelectGpu();
    }
    //CuDevice::Instantiate().DisableCaching();
#endif

	bool no_softmax = opts.no_softmax;
	std::string feature_transform = opts.feature_transform;
	bool apply_log = opts.apply_log;
	int32 num_stream = opts.num_stream;
	int32 batch_size = opts.batch_size;
	int32 skip_frames = opts.skip_frames;


	Nnet nnet_transf;

    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);

    // optionally remove softmax,
    Component::ComponentType last_type = nnet.GetComponent(nnet.NumComponents()-1).GetType();
    if (no_softmax) {
      if (last_type == Component::kSoftmax || last_type == Component::kBlockSoftmax) {
        KALDI_LOG << "Removing " << Component::TypeToMarker(last_type) << " from the nnet " << model_filename;
        nnet.RemoveComponent(nnet.NumComponents()-1);
      } else {
        KALDI_WARN << "Cannot remove softmax using --no-softmax=true, as the last component is " << Component::TypeToMarker(last_type);
      }
    }

    // avoid some bad option combinations,
    if (apply_log && no_softmax) {
      KALDI_ERR << "Cannot use both --apply-log=true --no-softmax=true, use only one of the two!";
    }

    // we will subtract log-priors later,
    PdfPrior pdf_prior(prior_opts);

	std::vector<int32> sweep_frames;
	if (!kaldi::SplitStringToIntegers(opts.sweep_frames_str, ":", false, &sweep_frames))
		KALDI_ERR << "Invalid sweep-frames string " << opts.sweep_frames_str;

	if (sweep_frames[0] > skip_frames || sweep_frames.size() > 1)
		KALDI_ERR << "invalid sweep frame index";

    Timer time;
    double time_now = 0;

    KALDI_LOG << "Nnet Forward STARTED";

    CuMatrix<BaseFloat>  cufeat, feats_transf, nnet_out;
    std::vector<MessageQueue> mq_output(num_stream);

    MQSample mq_sample;
    MQDecodable mq_decodable;
    std::vector<std::queue<MQDecodable*> > decodable_list(num_stream);

    Matrix<BaseFloat> sample;
    std::vector<Matrix<BaseFloat> > feats(num_stream);
    std::vector<int> curt(num_stream, 0);
    std::vector<int> lent(num_stream, 0);
    std::vector<int> frame_num_utt(num_stream, 0);
    std::vector<int> utt_curt(num_stream, 0);
    std::vector<int> new_utt_flags(num_stream, 1);
    std::vector<bool> is_end(num_stream, false);
    std::vector<pid_t> key_pid(num_stream, -1);
    std::unordered_map<pid_t, int> pid_key;
    Matrix<BaseFloat> feat, nnet_out_host;
    int step = 1024, t, s , k, dim;

    while (true)
    {
    	while (mq_forward.Receive((char*)&mq_sample, sample_attr.mq_msgsize, NULL) > 0)
    	{
    		pid_t pid = mq_sample.pid;
    		s = 0, dim = mq_sample.dim;
    		if (pid_key.find(pid) == pid_key.end())
    		{
    			for (s = 0; s < num_stream; s++)
    			{
    				if (key_pid[s] == -1)
    				{
    					key_pid[s] = pid;
    					break;
    				}
    			}
    			if (s == num_stream)
    			{
    				KALDI_WARN << "decoding thread out of forward stream range, discard message";
    				continue;
    			}
    			pid_key[pid] = s;
    			// attach corresponding decodable message queue
    			oflag = O_RDWR | O_NONBLOCK;
    			mq_output[s].Open(mq_sample.mq_callback_name, oflag);
    			KALDI_LOG << "decoder " << s << " attached " << std::string(mq_sample.mq_callback_name);
    		}

    		s = pid_key[pid];
    		if (feats[s].NumRows() == 0)
    			feats[s].Resize(step, dim, kUndefined, kStrideEqualNumCols);
    		if (feats[s].NumRows() < lent[s]+mq_sample.num_sample)
    		{
    			Matrix<BaseFloat> tmp(feats[s].NumRows()+step, dim, kUndefined, kStrideEqualNumCols);
    			tmp.RowRange(0, lent[s]).CopyFromMat(feats[s].RowRange(0, lent[s]));
    			feats[s].Swap(&tmp);
    		}
			memcpy((char*)feats[s].RowData(lent[s]), (char*)mq_sample.sample, sizeof(float)*dim*mq_sample.num_sample);
			lent[s] += mq_sample.num_sample;
			is_end[s] = mq_sample.is_end;

			frame_num_utt[s] = lent[s]/skip_frames;
			frame_num_utt[s] += lent[s]%skip_frames > sweep_frames[0] ? 1 : 0;
			//lent[s] = lent[s] > frame_num_utt[s]*skip_frames ? frame_num_utt[s]*skip_frames : lent[s];
    	}

    	for (s = 0; s < num_stream; s++)
    	{
    		while (!decodable_list[s].empty())
    		{
    			MQDecodable* mq_decodable = decodable_list[s].front();
    			int ret = mq_output[s].Send((char*)mq_decodable, sizeof(MQDecodable), 0);
    			if (ret == -1) break;
    			// send successful
    			decodable_list[s].pop();
    			delete mq_decodable;

    			// initialization for new utterance
    			if(mq_decodable->is_end)
    			{
    				lent[s] = 0;
    				curt[s] = 0;
    				utt_curt[s] = 0;
    				new_utt_flags[s] = 1;
    				feats[s].Resize(step, dim, kUndefined, kStrideEqualNumCols);
    			}
    		}
    	}

        // we are done if all streams are exhausted
        bool done = true;
        for (s = 0; s < num_stream; s++) {
            if (curt[s] < lent[s]) done = false;  // this stream still contains valid data, not exhausted
        }

        if (done) {
        	usleep(0.05*1000000);
        	continue;
        }

    	if (feat.NumCols() != dim) {
    		feat.Resize(batch_size * num_stream, dim, kSetZero, kStrideEqualNumCols);
    	}
    	 // fill a multi-stream bptt batch
    	for (t = 0; t < batch_size; t++) {
    		for (s = 0; s < num_stream; s++) {
				// feat shifting & padding
				if (curt[s] < lent[s]) {
					feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(curt[s]));
				} else {
					int last = (frame_num_utt[s]-1)*skip_frames; // lent[s]-1
					if (last >= 0)
					feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(last));
				}
				curt[s] += skip_frames;
    		}
    	}

    	// apply optional feature transform
    	nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &feats_transf);

		// for streams with new utterance, history states need to be reset
		nnet.ResetLstmStreams(new_utt_flags);
		nnet.SetSeqLengths(new_utt_flags);

		// forward pass
		nnet.Propagate(CuMatrix<BaseFloat>(feat), &nnet_out);

    	// convert posteriors to log-posteriors,
    	if (apply_log) {
    	  nnet_out.Add(1e-20); // avoid log(0),
    	  nnet_out.ApplyLog();
    	}

    	// subtract log-priors from log-posteriors or pre-softmax,
    	if (prior_opts.class_frame_counts != "") {
    	  pdf_prior.SubtractOnLogpost(&nnet_out);
    	}

		nnet_out.CopyToMat(&nnet_out_host);


		for (s = 0; s < num_stream; s++) {

			MQDecodable *mq_decodable = new MQDecodable;
			dim = nnet_out_host.NumCols();
			float *dest = mq_decodable->sample;

			for (t = 0; t < batch_size; t++) {
				// feat shifting & padding
				if (opts.copy_posterior) {
				   for (k = 0; k < skip_frames; k++){
						if (utt_curt[s] < lent[s]) {
							memcpy((char*)dest, (char*)nnet_out_host.RowData(t * num_stream + s), dim*sizeof(float));
							dest += dim;
							utt_curt[s]++;
							mq_decodable->num_sample++;
						}
				   }
				}
				else {
				   if (utt_curt[s] < frame_num_utt[s]) {
					   memcpy((char*)dest, (char*)nnet_out_host.RowData(t * num_stream + s), dim*sizeof(float));
					   dest += dim;
					   utt_curt[s]++;
					   mq_decodable->num_sample++;
				   }
				}
			}
			if (mq_decodable->num_sample == 0) {
				delete mq_decodable;
				continue;
			}
			mq_decodable->dim = dim;
			mq_decodable->is_end = is_end[s] && ((opts.copy_posterior && utt_curt[s] == lent[s]) ||
									(!opts.copy_posterior && utt_curt[s] == frame_num_utt[s]));
			decodable_list[s].push(mq_decodable);
			new_utt_flags[s] = 0;
		} // rearrangement


    }

    KALDI_LOG << "Nnet Forward FINISHED; ";

    time_now = time.Elapsed();


#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
