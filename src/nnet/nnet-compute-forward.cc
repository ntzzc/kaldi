// nnet/nnet-compute-forward.cc

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

#include "lat/lattice-functions.h"
#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-mutex.h"
#include "thread/kaldi-thread.h"

#include "nnet/nnet-example.h"
#include "nnet/nnet-compute-forward.h"

namespace kaldi {
namespace nnet1 {

class NnetForwardParallelClass: public MultiThreadable {
private:
	const NnetForwardOptions *opts;
	std::string model_filename;
	ExamplesRepository *repository;
	BaseFloatMatrixWriter *feature_writer;
	Mutex *examples_mutex;
	NnetForwardStats *stats;

public:
	NnetForwardParallelClass(const NnetForwardOptions *opts,
							std::string model_filename,
							ExamplesRepository *repository,
							BaseFloatMatrixWriter *feature_writer,
							Mutex *examples_mutex,
							NnetForwardStats *stats):
								opts(opts),model_filename(model_filename),
								repository(repository),feature_writer(feature_writer),
								examples_mutex(examples_mutex), stats(stats)
	{

	}

	  // This does the main function of the class.
	void operator ()()
	{
		int gpuid;

		examples_mutex->Lock();
		// Select the GPU
		#if HAVE_CUDA == 1
			if (opts->use_gpu == "yes")
		    	gpuid = CuDevice::Instantiate().SelectGpu();
		    //CuDevice::Instantiate().DisableCaching();
		#endif

		examples_mutex->Unlock();

		bool no_softmax = opts->no_softmax;
		std::string feature_transform = opts->feature_transform;
		bool apply_log = opts->apply_log;
		int32 time_shift = opts->time_shift;
		const PdfPriorOptions *prior_opts = opts->prior_opts;
		int32 num_stream = opts->num_stream;
		int32 batch_size = opts->batch_size;
		int32 skip_frames = opts->skip_frames;


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
	    PdfPrior pdf_prior(*opts->prior_opts);

	    // disable dropout,
	    nnet_transf.SetDropoutRetention(1.0);
	    nnet.SetDropoutRetention(1.0);

	    CuMatrix<BaseFloat>  cufeat, feats_transf, nnet_out;

	    std::vector<std::string> keys(num_stream);
	    std::vector<Matrix<BaseFloat> > feats(num_stream);
	    std::vector<Posterior> targets(num_stream);
	    std::vector<int> curt(num_stream, 0);
	    std::vector<int> lent(num_stream, 0);
	    std::vector<int> frame_num_utt(num_stream, 0);
	    std::vector<int> new_utt_flags(num_stream, 0);

	    std::vector<Matrix<BaseFloat> > utt_feats(num_stream);
	    std::vector<int> utt_curt(num_stream, 0);
	    std::vector<bool> utt_copied(num_stream, 0);

	    // bptt batch buffer
	    int32 feat_dim = nnet.InputDim();
	    int32 out_dim = nnet.OutputDim();
	    Vector<BaseFloat> frame_mask(batch_size * num_stream, kSetZero);
	    Matrix<BaseFloat> feat, nnet_out_host;
	    if (batch_size * num_stream > 0)
	    {
	    	feat.Resize(batch_size * num_stream, feat_dim, kSetZero);
	    	nnet_out_host.Resize(batch_size * num_stream, out_dim, kSetZero);
	    }


	    kaldi::int64 total_frames = 0;
	    int32 num_done = 0, num_frames;
	    Timer time;
	    double time_now = 0;


	    FeatureExample *example;

	    //num_stream=1 for lstm debug
	    if (num_stream >= 1)
	    while (1)
	    {
	    	 // loop over all streams, check if any stream reaches the end of its utterance,
	    	 // if any, feed the exhausted stream with a new utterance, update book-keeping infos
	    	for (int s = 0; s < num_stream; s++)
	    	{
	    		// this stream still has valid frames
	    		if (curt[s] < lent[s]) {
	    			new_utt_flags[s] = 0;
	    		    continue;
	    		}

	    		if (utt_curt[s] > 0 && !utt_copied[s])
	    		{
	    			examples_mutex->Lock();
	    			feature_writer->Write(keys[s], utt_feats[s]);
	    			examples_mutex->Unlock();
	    			utt_copied[s] = true;
	    		}

	    		while ((example = dynamic_cast<FeatureExample*>(repository->ProvideExample())) != NULL)
	    		{
	    	    	std::string key = example->utt;
	    	    	Matrix<BaseFloat> &mat = example->input_frames;
			// forward the features through a feature-transform,
                        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

	    	    	num_done++;

	                // checks ok, put the data in the buffers,
	                keys[s] = key;
	                feats[s].Resize(feats_transf.NumRows(), feats_transf.NumCols());
	                feats_transf.CopyToMat(&feats[s]);
	                //feats[s] = mat;
	                curt[s] = 0;
	                lent[s] = feats[s].NumRows();
	                new_utt_flags[s] = 1;  // a new utterance feeded to this stream

	                frame_num_utt[s] = (lent[s]+skip_frames-1)/skip_frames;
	                int32 utt_frames = opts->copy_posterior ? lent[s]:frame_num_utt[s];
	                utt_feats[s].Resize(utt_frames, out_dim, kUndefined);
	                utt_copied[s] = false;
	                utt_curt[s] = 0;

	                delete example;
	                break;
	    		}
	    	}

		        // we are done if all streams are exhausted
		        int done = 1;
		        for (int s = 0; s < num_stream; s++) {
		            if (curt[s] < lent[s]) done = 0;  // this stream still contains valid data, not exhausted
		        }

		        if (done) break;

		        // fill a multi-stream bptt batch
		        // * frame_mask: 0 indicates padded frames, 1 indicates valid frames
		        // * target: padded to batch_size
		        // * feat: first shifted to achieve targets delay; then padded to batch_size
		        for (int t = 0; t < batch_size; t++) {
		           for (int s = 0; s < num_stream; s++) {
		               // feat shifting & padding
		               if (curt[s] + time_shift*skip_frames < lent[s]) {
		                   feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(curt[s]+time_shift*skip_frames));
		               } else {
		            	   int last = (frame_num_utt[s]-1)*skip_frames; // lent[s]-1
				   if (last >= 0)
		                   feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(last));
		               }

		               curt[s]+=skip_frames;
		           }
		       }

		        num_frames = feat.NumRows();
			    // report the speed
			    if (num_done % 5000 == 0) {
			      time_now = time.Elapsed();
			      KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
			                    << time_now/60 << " min; processed " << total_frames/time_now
								<< " frames per second.";
			    }

			    // apply optional feature transform
			   //nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &feats_transf);

			   // for streams with new utterance, history states need to be reset
			   nnet.ResetLstmStreams(new_utt_flags);
			   nnet.SetSeqLengths(new_utt_flags);

			   // forward pass
			   //nnet.Propagate(feats_transf, &nnet_out);
			   nnet.Propagate(CuMatrix<BaseFloat>(feat), &nnet_out);

		    	// convert posteriors to log-posteriors,
		    	if (apply_log) {
		    	  nnet_out.Add(1e-20); // avoid log(0),
		    	  nnet_out.ApplyLog();
		    	}
        
		    	// subtract log-priors from log-posteriors or pre-softmax,
		    	if (prior_opts->class_frame_counts != "") {
		    	  pdf_prior.SubtractOnLogpost(&nnet_out);
		    	}

			   nnet_out.CopyToMat(&nnet_out_host);

		       for (int t = 0; t < batch_size; t++) {
		           for (int s = 0; s < num_stream; s++) {
		               // feat shifting & padding
		        	   if (opts->copy_posterior)
		        	   {
		        		   for (int k = 0; k < skip_frames; k++){
		        		   		if (utt_curt[s] < lent[s]) {
		        		   			utt_feats[s].Row(utt_curt[s]).CopyFromVec(nnet_out_host.Row(t * num_stream + s));
		        		   			utt_curt[s]++;
		        		   		}
		        		   }
		        	   }
		        	   else
		        	   {
		        		   if (utt_curt[s] < frame_num_utt[s]) {
		        			   utt_feats[s].Row(utt_curt[s]).CopyFromVec(nnet_out_host.Row(t * num_stream + s));
		        			   utt_curt[s]++;
		        		   }
		        	   }
		           }
		       }

		       total_frames += num_frames;
	    }

	    // for dnn cnn
	    if (num_stream < 1)
	    while ((example = dynamic_cast<FeatureExample*>(repository->ProvideExample())) != NULL)
	    {
	    	std::string utt = example->utt;
	    	Matrix<BaseFloat> &mat = example->input_frames;

	    	/*
	        if (!KALDI_ISFINITE(mat.Sum())) { // check there's no nan/inf,
	          KALDI_ERR << "NaN or inf found in features for " << utt;
	        }
			*/

	    	// time-shift, copy the last frame of LSTM input N-times,
	    	if (time_shift > 0) {
	    	  int32 last_row = mat.NumRows() - 1; // last row,
	    	  mat.Resize(mat.NumRows() + time_shift, mat.NumCols(), kCopyData);
	    	  for (int32 r = last_row+1; r<mat.NumRows(); r++) {
	    	    mat.CopyRowFromVec(mat.Row(last_row), r); // copy last row,
	    	  }
	    	}

	    	// push it to gpu,
	    	cufeat = mat;
	        // fwd-pass, feature transform,
	        nnet_transf.Feedforward(cufeat, &feats_transf);

	        // fwd-pass, nnet,
	        nnet.Feedforward(feats_transf, &nnet_out);


	    	// convert posteriors to log-posteriors,
	    	if (apply_log) {
	    	  if (!(nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0)) {
	    	    KALDI_WARN << utt << " "
	    	               << "Applying 'log' to data which don't seem to be probabilities "
	    	               << "(is there a softmax somwhere?)";
	    	  }
	    	  nnet_out.Add(1e-20); // avoid log(0),
	    	  nnet_out.ApplyLog();
	    	}

	    	// subtract log-priors from log-posteriors or pre-softmax,
	    	if (prior_opts->class_frame_counts != "") {
	    	  if (nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0) {
	    	    KALDI_WARN << utt << " "
	    	               << "Subtracting log-prior on 'probability-like' data in range [0..1] "
	    	               << "(Did you forget --no-softmax=true or --apply-log=true ?)";
	    	  }
	    	  pdf_prior.SubtractOnLogpost(&nnet_out);
	    	}

	    	// download from GPU,
	    	nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
	    	nnet_out.CopyToMat(&nnet_out_host);

	    	// time-shift, remove N first frames of LSTM output,
	    	if (time_shift > 0) {
	    	  Matrix<BaseFloat> tmp(nnet_out_host);
	    	  nnet_out_host = tmp.RowRange(time_shift, tmp.NumRows() - time_shift);
	    	}

	    	// write,
	    	if (!KALDI_ISFINITE(nnet_out_host.Sum())) { // check there's no nan/inf,
	    	  KALDI_ERR << "NaN or inf found in final output nn-output for " << utt;
	    	}

	    	examples_mutex->Lock();
	    	feature_writer->Write(utt, nnet_out_host);
	    	examples_mutex->Unlock();

	    	// progress log
	    	if (num_done % 100 == 0) {
	    	  time_now = time.Elapsed();
	    	  KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
	    	                << time_now/60 << " min; processed " << total_frames/time_now
	    	                << " frames per second.";
	    	}
	    	num_done++;
	    	total_frames += example->input_frames.NumRows();

	        // release the buffers we don't need anymore
	       	delete example;
	    }

	    examples_mutex->Lock();
	    stats->num_done += num_done;
	    stats->total_frames += total_frames;
	    examples_mutex->Unlock();

	}


};


void NnetForwardParallel(const NnetForwardOptions *opts,
						std::string	model_filename,
						std::string feature_rspecifier,
						std::string feature_wspecifier,
						NnetForwardStats *stats)
{
    ExamplesRepository repository;
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);
    Mutex examples_mutex;

    NnetForwardParallelClass c(opts, model_filename, &repository, &feature_writer, &examples_mutex, stats);

    		// The initialization of the following class spawns the threads that
    	    // process the examples.  They get re-joined in its destructor.
    	    MultiThreader<NnetForwardParallelClass> m(opts->num_threads, c);

    	    // iterate over all feature files
    	    NnetExample *example;
    	    std::vector<NnetExample*> examples;
    	    for (; !feature_reader.Done(); feature_reader.Next()) {
    	    	example = new FeatureExample(&feature_reader);
    	    	if (example->PrepareData(examples))
    	    	{
    	    		for (int i = 0; i < examples.size(); i++)
    	    			repository.AcceptExample(examples[i]);
    	    		if (examples[0] != example)
    	    		    			delete example;
    	    	}
    	    	else
    	    		delete example;
    	    }
    	    repository.ExamplesDone();

}

} // namespace nnet
} // namespace kaldi
