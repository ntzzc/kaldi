// nnet/nnet-example.h

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
#include "hmm/posterior.h"
#include "lat/lattice-functions.h"
#include "nnet/nnet-example.h"

namespace kaldi {
namespace nnet1 {

bool DNNNnetExample::PrepareData()
{

	        utt = feature_reader->Key();
	        KALDI_VLOG(3) << "Reading " << utt;
	        // check that we have targets
	        if (!targets_reader->HasKey(utt)) {
	          KALDI_WARN << utt << ", missing targets";
	          model_sync->LockStates();
	          stats->num_no_tgt_mat++;
	          model_sync->UnlockStates();
	          return false;
	        }
	        // check we have per-frame weights
	        if (opts->frame_weights != "" && !weights_reader->HasKey(utt)) {
	          KALDI_WARN << utt << ", missing per-frame weights";
	          model_sync->LockStates();
	          stats->num_other_error++;
	          model_sync->UnlockStates();
	          return false;
	        }

	        // get feature / target pair
	        input_frames = feature_reader->Value();
	        targets = targets_reader->Value(utt);
	        // get per-frame weights
	        if (opts->frame_weights != "") {
	        	frames_weights = weights_reader->Value(utt);
	        } else { // all per-frame weights are 1.0
	        	frames_weights.Resize(input_frames.NumRows());
	        	frames_weights.Set(1.0);
	        }


	        // correct small length mismatch ... or drop sentence
	        {
	          // add lengths to vector
	          std::vector<int32> lenght;
	          lenght.push_back(input_frames.NumRows());
	          lenght.push_back(targets.size());
	          lenght.push_back(frames_weights.Dim());
	          // find min, max
	          int32 min = *std::min_element(lenght.begin(),lenght.end());
	          int32 max = *std::max_element(lenght.begin(),lenght.end());
	          // fix or drop ?
	          if (max - min < opts->length_tolerance) {
	            if(input_frames.NumRows() != min) input_frames.Resize(min, input_frames.NumCols(), kCopyData);
	            if(targets.size() != min) targets.resize(min);
	            if(frames_weights.Dim() != min) frames_weights.Resize(min, kCopyData);
	          } else {
	            KALDI_WARN << utt << ", length mismatch of targets " << targets.size()
	                       << " and features " << input_frames.NumRows();
	            model_sync->LockStates();
	            stats->num_other_error++;
	            model_sync->UnlockStates();
	            return false;
	          }
	        }


	return true;
}

bool CTCNnetExample::PrepareData()
{
    utt = feature_reader->Key();
    KALDI_VLOG(3) << "Reading " << utt;
    // check that we have targets
    if (!targets_reader->HasKey(utt)) {
      KALDI_WARN << utt << ", missing targets";
      model_sync->LockStates();
      stats->num_no_tgt_mat++;
      model_sync->UnlockStates();
      return false;
    }

    // get feature / target pair
    input_frames = feature_reader->Value();
    targets = targets_reader->Value(utt);

    return true;
}


bool SequentialNnetExample::PrepareData()
{
			  utt = feature_reader->Key();
		      if (!den_lat_reader->HasKey(utt)) {
		        KALDI_WARN << "Utterance " << utt << ": found no lattice.";
		        model_sync->LockStates();
		        stats->num_no_den_lat++;
		        model_sync->UnlockStates();
		        return false;
		      }
		      if (!num_ali_reader->HasKey(utt)) {
		        KALDI_WARN << "Utterance " << utt << ": found no reference alignment.";
		        model_sync->LockStates();
		        stats->num_no_num_ali++;
		        model_sync->UnlockStates();

		        return false;
		      }

		      // 1) get the features, numerator alignment
		      input_frames = feature_reader->Value();
		      num_ali = num_ali_reader->Value(utt);
		      int32 skip_frames = opts->skip_frames;
		      int32 utt_frames = (input_frames.NumRows()+skip_frames-1)/skip_frames; // utt_frames=input_frames.NumRows()

		      // check for temporal length of numerator alignments
		      if ((int32)num_ali.size() != utt_frames){
		        KALDI_WARN << "Numerator alignment has wrong length "
		                   << num_ali.size() << " vs. "<< utt_frames;
		        model_sync->LockStates();
		        stats->num_other_error++;
		        model_sync->UnlockStates();
		        return false;
		      }
		      if (input_frames.NumRows() > opts->max_frames) {
		    	  KALDI_WARN << "Utterance " << utt << ": Skipped because it has " << input_frames.NumRows() <<
		    			  	  " frames, which is more than " << opts->max_frames << ".";
		    	  model_sync->LockStates();
		    	  stats->num_other_error++;
		    	  model_sync->UnlockStates();
		    	  return false;
		      }

		      // 2) get the denominator lattice, preprocess
		      den_lat = den_lat_reader->Value(utt);
		      if (den_lat.Start() == -1) {
		        KALDI_WARN << "Empty lattice for utt " << utt;
		        model_sync->LockStates();
		        stats->num_other_error++;
		        model_sync->UnlockStates();
		        return false;
		      }
		      if (opts->old_acoustic_scale != 1.0) {
		        fst::ScaleLattice(fst::AcousticLatticeScale(opts->old_acoustic_scale), &den_lat);
		      }
		      // optional sort it topologically
		      kaldi::uint64 props = den_lat.Properties(fst::kFstProperties, false);
		      if (!(props & fst::kTopSorted)) {
		        if (fst::TopSort(&den_lat) == false)
		          KALDI_ERR << "Cycles detected in lattice.";
		      }
		      // get the lattice length and times of states
		      int32 max_time = kaldi::LatticeStateTimes(den_lat, &state_times);
		      // check for temporal length of denominator lattices
		      if (max_time != utt_frames) {
		        KALDI_WARN << "Denominator lattice has wrong length "
		                   << max_time << " vs. " << utt_frames;
		        model_sync->LockStates();
		        stats->num_other_error++;
		        model_sync->UnlockStates();
		        return false;
		      }

		      return true;
}

bool LstmNnetExample::PrepareData()
{

	return true;
}

void ExamplesRepository::AcceptExample(
		NnetExample *example) {
  empty_semaphore_.Wait();
  examples_mutex_.Lock();
  examples_.push_back(example);
  examples_mutex_.Unlock();
  full_semaphore_.Signal();
}

void ExamplesRepository::ExamplesDone() {
  for (int32 i = 0; i < buffer_size_; i++)
    empty_semaphore_.Wait();
  examples_mutex_.Lock();
  KALDI_ASSERT(examples_.empty());
  examples_mutex_.Unlock();
  done_ = true;
  full_semaphore_.Signal();
}

NnetExample*
ExamplesRepository::ProvideExample() {
  full_semaphore_.Wait();
  if (done_) {
    KALDI_ASSERT(examples_.empty());
    full_semaphore_.Signal(); // Increment the semaphore so
    // the call by the next thread will not block.
    return NULL; // no examples to return-- all finished.
  } else {
    examples_mutex_.Lock();
    KALDI_ASSERT(!examples_.empty());
    NnetExample *ans = examples_.front();
    examples_.pop_front();
    examples_mutex_.Unlock();
    empty_semaphore_.Signal();
    return ans;
  }
}


} // namespace nnet
} // namespace kaldi

