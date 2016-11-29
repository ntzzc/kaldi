// online0bin/online-nnet-wav-decoder.cc

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



#include "base/timer.h"
#include "feat/wave-reader.h"

#include "online/onlinebin-util.h"
#include "online0/online-nnet-decoding.h"

int main(int argc, char *argv[])
{
	try {

	    using namespace kaldi;
	    using namespace fst;

	    typedef kaldi::int32 int32;

	    const char *usage =
	        "Reads in wav file(s) and simulates online decoding with neural nets "
	        "(nnet0 or nnet1 setup), Note: some configuration values and inputs are\n"
	    	"set via config files whose filenames are passed as options\n"
	    	"\n"
	        "Usage: online-nnet-decoder [config option]\n"
	    	"e.g.: \n"
	        "	online-nnet-decoder --config=conf/online_decoder.conf --wavscp=wav.scp\n";

	    ParseOptions po(usage);

	    OnlineNnetFasterDecoderOptions decoder_opts;
	    OnlineNnetDecodingOptions decoding_opts(decoder_opts);
	    decoding_opts.Register(&po);

	    std::string wav_rspecifier;
		po.Register("wavscp", &wav_rspecifier, "wav list for decode");

	    po.Read(argc, argv);

	    if (argc < 2) {
	        po.PrintUsage();
	        exit(1);
	    }


	    OnlineNnetForwardOptions forward_opts;
	    OnlineNnetFeaturePipelineOptions feature_opts(decoding_opts.feature_cfg);
		ReadConfigFromFile(decoding_opts.decoder_cfg, &decoder_opts);
		ReadConfigFromFile(decoding_opts.forward_cfg, &forward_opts);

	    Int32VectorWriter words_writer(decoding_opts.words_wspecifier);
	    Int32VectorWriter alignment_writer(decoding_opts.alignment_wspecifier);

	    TransitionModel trans_model;
		bool binary;
		Input ki(decoding_opts.model_rspecifier, &binary);
		trans_model.Read(ki.Stream(), binary);

	    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldi(decoding_opts.fst_rspecifier);
	    fst::SymbolTable *word_syms = NULL;
	    if (!(word_syms = fst::SymbolTable::ReadText(decoding_opts.word_syms_filename)))
	        KALDI_ERR << "Could not read symbol table from file " << decoding_opts.word_syms_filename;

	    DecoderSync decoder_sync;

	    OnlineNnetFasterDecoder decoder(*decode_fst, decoder_opts);
	    OnlineDecodableMatrixMapped decodable(trans_model, decoding_opts.acoustic_scale);

	    OnlineNnetDecodingClass decoding(decoding_opts,
	    								&decoder, &decodable, &decoder_sync,
										*word_syms, words_writer, alignment_writer);
		// The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<OnlineNnetDecodingClass> m(1, decoding);

	    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);

        // client feature
        Timer timer;

        OnlineNnetFeaturePipeline feature_pipeline(feature_opts);
        OnlineNnetForward forward(forward_opts);
        BaseFloat chunk_length_secs = 0.05;
        int feat_dim = feature_pipeline.Dim();
        int batch_size = forward_opts.batch_size;
        Matrix<BaseFloat> feat(batch_size, feat_dim);
        Matrix<BaseFloat> feat_out;

        kaldi::int64 frame_count = 0;

        while (!wav_reader.Done()) {
            std::string utt_key = wav_reader.Key();
        	const WaveData &wave_data = wav_reader.Value();
            // get the data for channel zero (if the signal is not mono, we only
            // take the first channel).
            SubVector<BaseFloat> data(wave_data.Data(), 0);

            BaseFloat samp_freq = wave_data.SampFreq();
            int32 chunk_length;
			if (chunk_length_secs > 0) {
				chunk_length = int32(samp_freq * chunk_length_secs);
				if (chunk_length == 0) chunk_length = 1;
			} else {
				chunk_length = std::numeric_limits<int32>::max();
			}

			feature_pipeline.Reset();
			forward.ResetHistory();
			int32 samp_offset = 0, frame_offset = 0, frame_ready;
			while (samp_offset < data.Dim()) {
				int32 samp_remaining = data.Dim() - samp_offset;
				int32 num_samp = chunk_length < samp_remaining ? chunk_length : samp_remaining;

				SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
				feature_pipeline.AcceptWaveform(samp_freq, wave_part);
				samp_offset += num_samp;
				if (samp_offset >= data.Dim())
					feature_pipeline.InputFinished();

				while (true) {
					frame_ready = feature_pipeline.NumFramesReady();
					if (!feature_pipeline.IsLastFrame(frame_ready-1) && frame_ready < frame_offset+batch_size)
						break;
					else if (feature_pipeline.IsLastFrame(frame_ready-1)) {
						frame_ready -= frame_offset;
						feat.SetZero();
					}
					else
						frame_ready = batch_size;

					for (int i = 0; i < frame_ready; i++) {
						feature_pipeline.GetFrame(frame_offset, &feat.Row(i));
						frame_offset++;
					}
					forward.Forward(feat, &feat_out);
					decodable.AcceptLoglikes(&feat_out);
					decoder_sync.DecoderSignal();
				}
			} //part wav

			// waiting a utterance finished
			decoder_sync.UtteranceWait();
			KALDI_LOG << "Finish decode utterance : " << utt_key;

			frame_count += frame_offset;
			wav_reader.Next();
			if (wav_reader.Done())
				decoder_sync.Abort();
			decodable.Reset();
        }

        double elapsed = timer.Elapsed();
        KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);

	    delete decode_fst;
	    delete word_syms;
	    return 0;
	} catch(const std::exception& e) {
	    std::cerr << e.what();
	    return -1;
	  }
} // main()
