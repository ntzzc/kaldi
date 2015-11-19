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
		PdfPriorOptions *prior_opts	= opts->prior_opts;
		int32 num_stream = opts->num_stream;
		int32 batch_size = opts->batch_size;


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
	    PdfPrior pdf_prior(opts->prior_opts);

	    // disable dropout,
	    nnet_transf.SetDropoutRetention(1.0);
	    nnet.SetDropoutRetention(1.0);

	    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
	    Matrix<BaseFloat> nnet_out_host;

	    std::vector<std::string> keys(num_stream);
	    std::vector<Matrix<BaseFloat> > utt_feats(num_stream);
	    std::vector<Posterior> targets(num_stream);
	    std::vector<int> curt(num_stream, 0);
	    std::vector<int> lent(num_stream, 0);
	    std::vector<int> new_utt_flags(num_stream, 0);

	    // bptt batch buffer
	    int32 feat_dim = nnet.InputDim();
	    Vector<BaseFloat> frame_mask(batch_size * num_stream, kSetZero);
	    Matrix<BaseFloat> feat(batch_size * num_stream, feat_dim, kSetZero);


	    kaldi::int64 tot_t = 0;
	    int32 num_done = 0, num_frames;
	    Timer time;
	    double time_now = 0;


	    FeatureExample *example;

	    while (1)
	    {
	    	 // loop over all streams, check if any stream reaches the end of its utterance,
	    	        // if any, feed the exhausted stream with a new utterance, update book-keeping infos
	    	for (int s = 0; s < num_stream; s++) {
	    		// this stream still has valid frames
	    		if (curt[s] < lent[s]) {
	    			new_utt_flags[s] = 0;
	    		    continue;
	    		}

	    	}
	    }


	    while ((example = dynamic_cast<FeatureExample*>(repository->ProvideExample())) != NULL)
	    {
	    	std::string utt = example->utt;
	    	const Matrix<BaseFloat> &mat = example->feat;

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
	    	feats = mat;
	        // fwd-pass, feature transform,
	        nnet_transf.Feedforward(feats, &feats_transf);

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
	    	                << time_now/60 << " min; processed " << tot_t/time_now
	    	                << " frames per second.";
	    	}
	    	num_done++;
	    	tot_t += example->feat.NumRows();

	        // release the buffers we don't need anymore
	       	delete example;
	    }

	    examples_mutex->Lock();
	    stats->num_done += num_done;
	    stats->total_frames += tot_t;
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
    	    for (; !feature_reader.Done(); feature_reader.Next()) {
    	    	example = new FeatureExample(&feature_reader);
    	    	if (example->PrepareData())
    	    		repository.AcceptExample(example);
    	    	else
    	    		delete example;
    	    }
    	    repository.ExamplesDone();

}

} // namespace nnet
} // namespace kaldi
