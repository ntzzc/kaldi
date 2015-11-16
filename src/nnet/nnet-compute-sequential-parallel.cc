// nnet/nnet-compute-sequential-parallel.cc

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

#include <deque>
#include "hmm/posterior.h"
#include "lat/lattice-functions.h"
#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-mutex.h"
#include "thread/kaldi-thread.h"

#include "lat/kaldi-lattice.h"

#include "cudamatrix/cu-device.h"
#include "base/kaldi-types.h"


#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-affine-preconditioned-transform.h"
#include "nnet/nnet-model-merge-function.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-example.h"

#include "nnet/nnet-compute-sequential-parallel.h"

namespace kaldi {
namespace nnet1 {

class SeqTrainParallelClass: public MultiThreadable {

private:
    const NnetSequentialUpdateOptions *opts;
    NnetModelSync *model_sync;

	std::string feature_transform,
				model_filename,
				si_model_filename,
				transition_model_filename;

	std::string den_lat_rspecifier,
				num_ali_rspecifier,
				feature_rspecifier;

	ExamplesRepository *repository_;
    NnetSequentialStats *stats_;

    const NnetTrainOptions *trn_opts;
    const PdfPriorOptions *prior_opts;
    const NnetParallelOptions *parallel_opts;

    BaseFloat 	acoustic_scale,
    			lm_scale,
		old_acoustic_scale,
    			kld_scale,
				frame_smooth;
    kaldi::int32 max_frames;
    bool drop_frames;
    std::string use_gpu;
    int32 num_threads;


 public:
  // This constructor is only called for a temporary object
  // that we pass to the RunMultiThreaded function.
	SeqTrainParallelClass(const NnetSequentialUpdateOptions *opts,
			NnetModelSync *model_sync,
			std::string feature_transform,
			std::string	model_filename,
			std::string transition_model_filename,
			std::string den_lat_rspecifier,
			std::string num_ali_rspecifier,
			ExamplesRepository *repository,
			Nnet *nnet,
			NnetSequentialStats *stats):
				opts(opts),
				model_sync(model_sync),
				feature_transform(feature_transform),
				model_filename(model_filename),
				transition_model_filename(transition_model_filename),
				den_lat_rspecifier(den_lat_rspecifier),
				num_ali_rspecifier(num_ali_rspecifier),
				repository_(repository),
				stats_(stats)
 	 		{
				trn_opts = opts->trn_opts;
				prior_opts = opts->prior_opts;
				parallel_opts = opts->parallel_opts;

				acoustic_scale = opts->acoustic_scale;
				lm_scale = opts->lm_scale;
				old_acoustic_scale = opts->old_acoustic_scale;
				kld_scale = opts->kld_scale;
				frame_smooth = opts->frame_smooth;
				max_frames = opts->max_frames;
				drop_frames = opts->drop_frames;
				use_gpu = opts->use_gpu;
				si_model_filename = opts->si_model_filename;

				num_threads = parallel_opts->num_threads;
 	 		}


	void LatticeAcousticRescore(const Matrix<BaseFloat> &log_like,
	                            const TransitionModel &trans_model,
	                            const std::vector<int32> &state_times,
	                            Lattice *lat) {
	  kaldi::uint64 props = lat->Properties(fst::kFstProperties, false);
	  if (!(props & fst::kTopSorted))
	    KALDI_ERR << "Input lattice must be topologically sorted.";

	  KALDI_ASSERT(!state_times.empty());
	  std::vector<std::vector<int32> > time_to_state(log_like.NumRows());
	  for (size_t i = 0; i < state_times.size(); i++) {
	    KALDI_ASSERT(state_times[i] >= 0);
	    if (state_times[i] < log_like.NumRows())  // end state may be past this..
	      time_to_state[state_times[i]].push_back(i);
	    else
	      KALDI_ASSERT(state_times[i] == log_like.NumRows()
	                   && "There appears to be lattice/feature mismatch.");
	  }

	  for (int32 t = 0; t < log_like.NumRows(); t++) {
	    for (size_t i = 0; i < time_to_state[t].size(); i++) {
	      int32 state = time_to_state[t][i];
	      for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
	           aiter.Next()) {
	        LatticeArc arc = aiter.Value();
	        int32 trans_id = arc.ilabel;
	        if (trans_id != 0) {  // Non-epsilon input label on arc
	          int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
	          arc.weight.SetValue2(-log_like(t, pdf_id) + arc.weight.Value2());
	          aiter.SetValue(arc);
	        }
	      }
	    }
	  }
	}

	void inline MMIObj(CuMatrix<BaseFloat> &nnet_out, CuMatrix<BaseFloat> &nnet_diff,
				TransitionModel &trans_model, SequentialNnetExample *example,
				double &total_mmi_obj, double &total_post_on_ali, int32 &num_frm_drop, int32 num_done,
				CuMatrix<BaseFloat> *soft_nnet_out, CuMatrix<BaseFloat> *si_nnet_out=NULL)
	{
		std::string utt = example->utt;
		const Matrix<BaseFloat> &mat = example->input_frames;
		const std::vector<int32> &num_ali = example->num_ali;
		Lattice &den_lat = example->den_lat;
		std::vector<int32> &state_times = example->state_times;

		Matrix<BaseFloat> nnet_out_h,  nnet_diff_h, si_nnet_out_h, soft_nnet_out_h;
		int num_frames, num_pdfs;
	    double lat_like; // total likelihood of the lattice
	    double lat_ac_like; // acoustic likelihood weighted by posterior.
	    double mmi_obj = 0.0, post_on_ali = 0.0;


		num_frames = nnet_out.NumRows();
		num_pdfs = nnet_out.NumCols();

	    // transfer it back to the host
		nnet_out_h.Resize(num_frames,num_pdfs, kUndefined);
		nnet_out.CopyToMat(&nnet_out_h);


		if (this->kld_scale > 0 || this->frame_smooth > 0)
		{
			soft_nnet_out_h.Resize(num_frames,num_pdfs, kUndefined);
			soft_nnet_out->CopyToMat(&soft_nnet_out_h);
		}

		if (this->kld_scale > 0)
		{
			si_nnet_out_h.Resize(num_frames,num_pdfs, kUndefined);
			si_nnet_out->CopyToMat(&si_nnet_out_h);

			//soft_nnet_out_h.AddMat(-1.0, si_nnet_out_h);
			si_nnet_out_h.AddMat(-1.0, soft_nnet_out_h);
		}


		// 4) rescore the latice
		LatticeAcousticRescore(nnet_out_h, trans_model, state_times, &den_lat);
		if (acoustic_scale != 1.0 || lm_scale != 1.0)
			fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &den_lat);

	       // 5) get the posteriors
	       kaldi::Posterior post;
	       lat_like = kaldi::LatticeForwardBackward(den_lat, &post, &lat_ac_like);

		double tmp = 0.0;
	       // 6) convert the Posterior to a matrix
	       nnet_diff_h.Resize(num_frames, num_pdfs, kSetZero);
	       for (int32 t = 0; t < post.size(); t++) {
	         for (int32 arc = 0; arc < post[t].size(); arc++) {
	           int32 pdf = trans_model.TransitionIdToPdf(post[t][arc].first);
	           nnet_diff_h(t, pdf) += post[t][arc].second;
			tmp = post[t][arc].second;
	         }
	       }


	       // 7) Calculate the MMI-objective function
	       // Calculate the likelihood of correct path from acoustic score,
	       // the denominator likelihood is the total likelihood of the lattice.
	       double path_ac_like = 0.0;
	       for(int32 t=0; t<num_frames; t++) {
	         int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
	         path_ac_like += nnet_out_h(t,pdf);
	       }
	       path_ac_like *= acoustic_scale;
	       mmi_obj = path_ac_like - lat_like;
	       //
	       // Note: numerator likelihood does not include graph score,
	       // while denominator likelihood contains graph scores.
	       // The result is offset at the MMI-objective.
	       // However the offset is constant for given alignment,
	       // so it is not harmful.

	       // Sum the den-posteriors under the correct path:
	       post_on_ali = 0.0;
	       for(int32 t=0; t<num_frames; t++) {
	         int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
	         double posterior = nnet_diff_h(t, pdf);
	         post_on_ali += posterior;
	       }

	       // Report
	       KALDI_VLOG(1) << "Lattice #" << num_done + 1 << " processed"
	                     << " (" << utt << "): found " << den_lat.NumStates()
	                     << " states and " << fst::NumArcs(den_lat) << " arcs.";

	       KALDI_VLOG(1) << "Utterance " << utt << ": Average MMI obj. value = "
	                     << (mmi_obj/num_frames) << " over " << num_frames
	                     << " frames."
	                     << " (Avg. den-posterior on ali " << post_on_ali/num_frames << ")";


	       // 7a) Search for the frames with num/den mismatch
	       int32 frm_drop = 0;
	       std::vector<int32> frm_drop_vec;
	       for(int32 t=0; t<num_frames; t++) {
	         int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
	         double posterior = nnet_diff_h(t, pdf);
	         if(posterior < 1e-20) {
	           frm_drop++;
	           frm_drop_vec.push_back(t);
	         }
	       }

	       // 8) subtract the pdf-Viterbi-path
	       for(int32 t=0; t<nnet_diff_h.NumRows(); t++) {
	         int32 pdf = trans_model.TransitionIdToPdf(num_ali[t]);
	         nnet_diff_h(t, pdf) -= 1.0;

	         //frame-smoothing
	         if (this->frame_smooth > 0)
	        	 soft_nnet_out_h(t, pdf) -= 1.0;
	       }

	       if (this->frame_smooth > 0)
	       {
	    	   nnet_diff_h.Scale(1-frame_smooth);
	    	   nnet_diff_h.AddMat(frame_smooth*0.1, soft_nnet_out_h);
	       }

			if (this->kld_scale > 0)
	        {
	        	nnet_diff_h.Scale(1.0-kld_scale);
	        	//-kld_scale means gradient descent direction.
				nnet_diff_h.AddMat(-kld_scale, si_nnet_out_h);

				KALDI_VLOG(1) << "likelihood of correct path = "  << path_ac_like
								<< " total likelihood of the lattice = " << lat_like;
	        }

	       // 9) Drop mismatched frames from the training by zeroing the derivative
	       if(drop_frames) {
	         for(int32 i=0; i<frm_drop_vec.size(); i++) {
	           nnet_diff_h.Row(frm_drop_vec[i]).Set(0.0);
	         }
	         num_frm_drop += frm_drop;
	       }
	       // Report the frame dropping
	       if (frm_drop > 0) {
	         std::stringstream ss;
	         ss << (drop_frames?"Dropped":"[dropping disabled] Would drop")
	            << " frames in " << utt << " " << frm_drop << "/" << num_frames << ",";
	         //get frame intervals from vec frm_drop_vec
	         ss << " intervals :";
	         //search for streaks of consecutive numbers:
	         int32 beg_streak=frm_drop_vec[0];
	         int32 len_streak=0;
	         int32 i;
	         for(i=0; i<frm_drop_vec.size(); i++,len_streak++) {
	           if(beg_streak + len_streak != frm_drop_vec[i]) {
	             ss << " " << beg_streak << ".." << frm_drop_vec[i-1] << "frm";
	             beg_streak = frm_drop_vec[i];
	             len_streak = 0;
	           }
	         }
	         ss << " " << beg_streak << ".." << frm_drop_vec[i-1] << "frm";
	         //print
	         KALDI_WARN << ss.str();
	       }

	       // 10) backpropagate through the nnet
	       nnet_diff.Resize(num_frames, num_pdfs, kUndefined);
	       nnet_diff.CopyFromMat(nnet_diff_h);

			       total_mmi_obj += mmi_obj;
	       total_post_on_ali += post_on_ali;

	}
	  // This does the main function of the class.
	void operator ()()
	{

		int gpuid;

		model_sync->LockModel();

	    // Select the GPU
	#if HAVE_CUDA == 1
	    if (parallel_opts->num_procs > 1)
	    {
	    	int32 thread_idx = model_sync->GetThreadIdx();
	    	KALDI_LOG << "MyId: " << parallel_opts->myid << "  ThreadId: " << thread_idx;
	    	gpuid = CuDevice::Instantiate().MPISelectGpu(model_sync->gpuinfo_, model_sync->win, thread_idx, this->num_threads);
	    	for (int i = 0; i< this->num_threads*parallel_opts->num_procs; i++)
	    	{
	    		KALDI_LOG << model_sync->gpuinfo_[i].hostname << "  myid: " << model_sync->gpuinfo_[i].myid
	    					<< "  gpuid: " << model_sync->gpuinfo_[i].gpuid;
	    	}
	    }
	    else
	    	gpuid = CuDevice::Instantiate().SelectGpu();

	    //CuDevice::Instantiate().DisableCaching();
	#endif

	    model_sync->UnlockModel();

	    // Read the class-frame-counts, compute priors
	    PdfPrior log_prior(*prior_opts);

	    // Read transition model
	    TransitionModel trans_model;
	    ReadKaldiObject(transition_model_filename, &trans_model);


	    int32 num_done = 0, num_no_num_ali = 0, num_no_den_lat = 0,
	          num_other_error = 0, num_frm_drop = 0;

	    int32 rank_in = 20, rank_out = 80, update_period = 4;
	    BaseFloat num_samples_history = 2000.0;
	    BaseFloat alpha = 4.0;

	    kaldi::int64 total_frames = 0;
	    double total_mmi_obj = 0.0, mmi_obj = 0.0;
	    double total_post_on_ali = 0.0, post_on_ali = 0.0;

		Nnet nnet_transf;
	    if (feature_transform != "") {
	      nnet_transf.Read(feature_transform);
	    }

	    Nnet nnet;
	    nnet.Read(model_filename);
	    // using activations directly: remove softmax, if present
	    if (nnet.GetComponent(nnet.NumComponents()-1).GetType() ==
	        kaldi::nnet1::Component::kSoftmax) {
	      KALDI_LOG << "Removing softmax from the nnet " << model_filename;
	      nnet.RemoveComponent(nnet.NumComponents()-1);
	    } else {
	      KALDI_LOG << "The nnet was without softmax " << model_filename;
	    }
	    //if (opts->num_procs > 1 || opts->use_psgd)
	    if (opts->use_psgd)
	    	nnet.SwitchToOnlinePreconditioning(rank_in, rank_out, update_period, num_samples_history, alpha);

	    nnet.SetTrainOptions(*trn_opts);

	    Nnet si_nnet, softmax;
	    if (this->kld_scale > 0)
	    {
	    	si_nnet.Read(si_model_filename);
	    }

	    if (this->kld_scale > 0 || frame_smooth > 0)
	    {
	    	KALDI_LOG << "KLD model Appending the softmax ...";
	    	softmax.AppendComponent(new Softmax(nnet.OutputDim(),nnet.OutputDim()));
        }

	    model_sync->Initialize(&nnet);

	    Timer time;
	    double time_now = 0;

		CuMatrix<BaseFloat> feats, feats_transf, nnet_out, nnet_diff,
							si_nnet_out, soft_nnet_out, *p_si_nnet_out=NULL, *p_soft_nnet_out;
		Matrix<BaseFloat> nnet_out_h, nnet_diff_h;

		SequentialNnetExample *example;

		ModelMergeFunction *p_merge_func = model_sync->GetModelMergeFunction();

		//double t1, t2, t3, t4;
		int32 update_frames = 0;

		while ((example = dynamic_cast<SequentialNnetExample*>(repository_->ProvideExample())) != NULL)
		{
			//time.Reset();
			std::string utt = example->utt;
			const Matrix<BaseFloat> &mat = example->input_frames;
			//t1 = time.Elapsed();
			//time.Reset();

		      // get actual dims for this utt and nnet
		      int32	num_frames = mat.NumRows(),
		    		  num_fea = mat.NumCols(),
					  num_pdfs = nnet.OutputDim();

		      // 3) propagate the feature to get the log-posteriors (nnet w/o sofrmax)
		      // push features to GPU
		      feats.Resize(num_frames, num_fea, kUndefined);
		      feats.CopyFromMat(mat);
		      // possibly apply transform
		      nnet_transf.Feedforward(feats, &feats_transf);
		      // propagate through the nnet (assuming w/o softmax)
		      nnet.Propagate(feats_transf, &nnet_out);

		      if (this->kld_scale > 0)
		      {	
		      	si_nnet.Propagate(feats_transf, &si_nnet_out);
		      	p_si_nnet_out = &si_nnet_out;
		      }

		      if (this->kld_scale > 0 || frame_smooth > 0)
		      {
		    	  softmax.Propagate(nnet_out, &soft_nnet_out);
		    	  p_soft_nnet_out = &soft_nnet_out;
		      }

		      // subtract the log_prior
		      if(prior_opts->class_frame_counts != "") {
		        log_prior.SubtractOnLogpost(&nnet_out);
		      }



		      MMIObj(nnet_out, nnet_diff,
		      				trans_model, example,
		      				total_mmi_obj, total_post_on_ali, num_frm_drop, num_done,
							p_soft_nnet_out, p_si_nnet_out);



		       if (parallel_opts->num_threads > 1 && update_frames >= opts->update_frames)
		       {
		    	   nnet.Backpropagate(nnet_diff, NULL, false);
		    	   nnet.Gradient();

		    	   //t2 = time.Elapsed();
		    	   //time.Reset();

		    	   if (parallel_opts->asgd_lock)
		    		   model_sync->LockModel();

		    	   model_sync->SetWeight(&nnet);
		    	   nnet.UpdateGradient();
		    	   model_sync->GetWeight(&nnet);

		    	   if (parallel_opts->asgd_lock)
		    		   model_sync->UnlockModel();

		    	   update_frames = 0;
			
		    	   //t3 = time.Elapsed();
		       }
		       else
		       {
		    	   nnet.Backpropagate(nnet_diff, NULL, true);

		    	   //t2 = time.Elapsed();
		    	   //time.Reset();
		       }

		       //KALDI_WARN << "prepare data: "<< t1 <<" forward & backward: "<< t2 <<" update: "<< t3;
		       // relase the buffer, we don't need anymore

		       //multi-machine
		       if (parallel_opts->num_procs > 1)
		       {
		    	   model_sync->LockModel();

		    	   if (p_merge_func->CurrentMergeCache()+num_frames > parallel_opts->merge_size && p_merge_func->leftMerge() > 1)
		    	   {
		    		   model_sync->GetWeight(&nnet);

		    		   p_merge_func->Merge(0);
		    		   KALDI_VLOG(1) << "Model merge NO." << parallel_opts->num_merge-p_merge_func->leftMerge()
		    						   << " Current mergesize = " << p_merge_func->CurrentMergeCache() <<" frames.";
		    		   p_merge_func->MergeCacheReset();

		    		   model_sync->SetWeight(&nnet);
		    	   }

		    	   p_merge_func->AddMergeCache((int)num_frames);

		    	   model_sync->UnlockModel();

		       }

			   // release the buffers we don't need anymore
			   feats.Resize(0,0);
			   feats_transf.Resize(0,0);
			   nnet_out.Resize(0,0);
			   si_nnet_out.Resize(0,0);
		       nnet_diff.Resize(0,0);
		       delete example;

		       // increase time counter
		       total_frames += num_frames;
		       num_done++;
		       update_frames += num_frames;

		       if (num_done % 100 == 0)
		       {
		         time_now = time.Elapsed();
		         KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
		                       << time_now/60 << " min; processed " << total_frames/time_now
		                       << " frames per second.";

		 	 	 #if HAVE_CUDA==1
		         	 	 // check the GPU is not overheated
		         	 	 CuDevice::Instantiate().CheckGpuHealth();
		 	 	 #endif

		       }
		       fflush(stderr); 
                       fsync(fileno(stderr));
	  }

		model_sync->LockStates();
		stats_->total_mmi_obj += total_mmi_obj;
		stats_->total_post_on_ali += total_post_on_ali;

		stats_->total_frames += total_frames;
		stats_->num_frm_drop += num_frm_drop;
		stats_->num_done += num_done;
		model_sync->UnlockStates();

		model_sync->LockModel();

		//last merge
		if (parallel_opts->num_procs > 1)
		{
			if (p_merge_func->leftMerge() == 1)
			{
				model_sync->GetWeight(&nnet);

				p_merge_func->Merge(0);
	    		KALDI_VLOG(1) << "Model merge NO." << parallel_opts->num_merge-p_merge_func->leftMerge()
	    						   << " Current mergesize = " << p_merge_func->CurrentMergeCache();
	    		model_sync->SetWeight(&nnet);
			}

		}
		model_sync->CopyToHost(&nnet);

		model_sync->UnlockModel();
	}

};


void NnetSequentialUpdateParallel(const NnetSequentialUpdateOptions *opts,
		std::string feature_transform,
		std::string	model_filename,
		std::string transition_model_filename,
		std::string feature_rspecifier,
		std::string den_lat_rspecifier,
		std::string num_ali_rspecifier,
		Nnet *nnet,
		NnetSequentialStats *stats)
{
		ExamplesRepository repository;
		NnetModelSync model_sync(nnet, opts->parallel_opts);

		SeqTrainParallelClass c(opts, &model_sync,
								feature_transform, model_filename, transition_model_filename, den_lat_rspecifier, num_ali_rspecifier,
								&repository, nnet, stats);


	  {

	    	SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
	    	RandomAccessLatticeReader den_lat_reader(den_lat_rspecifier);
	    	RandomAccessInt32VectorReader num_ali_reader(num_ali_rspecifier);

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<SeqTrainParallelClass> m(opts->parallel_opts->num_threads, c);
	    NnetExample *example;
	    for (; !feature_reader.Done(); feature_reader.Next()) {
	    	example = new SequentialNnetExample(&feature_reader, &den_lat_reader, &num_ali_reader, &model_sync, stats, opts);
	    	if (example->PrepareData())
	    		repository.AcceptExample(example);
	    	else
	    		delete example;
	    }
	    repository.ExamplesDone();
	  }

}


} // namespace nnet
} // namespace kaldi


