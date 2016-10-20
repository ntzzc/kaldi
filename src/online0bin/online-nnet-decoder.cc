// online0bin/online-nnet-decoder.cc

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
	        "Usage: online-nnet-decoder [options] <model-in> <fst-in> "
	        "<feature-rspecifier> <mqueue-wspecifier> <transcript-wspecifier> [alignments-wspecifier]\n"
	        "Example: ./online-nnet-decoder --rt-min=0.3 --rt-max=0.5 "
	        "--max-active=4000 --beam=12.0 --acoustic-scale=0.0769 "
	        "model HCLG.fst ark:features.ark forward.mq ark,t:trans.txt ark,t:ali.txt";

	    ParseOptions po(usage);

	    OnlineNnetFasterDecoderOptions decoder_opts;
	    decoder_opts.Register(&po, true);

	    OnlineNnetDecodingOptions decoding_opts(decoder_opts);
	    decoding_opts.Register(&po);

	    po.Read(argc, argv);

		if (po.NumArgs() != 5 && po.NumArgs() != 6) {
			  po.PrintUsage();
			  return 1;
		}

	    std::string
		model_rspecifier = po.GetArg(1),
		fst_rspecifier = po.GetArg(2),
		feature_rspecifier = po.GetArg(3),
		mqueue_wspecifier = po.GetArg(4),
		words_wspecifier = po.GetArg(5),
		alignment_wspecifier = po.GetOptArg(6);

	    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
	    Int32VectorWriter words_writer(words_wspecifier);
	    Int32VectorWriter alignment_writer(alignment_wspecifier);

	    TransitionModel trans_model;
		{
	    	bool binary;
			Input ki(model_rspecifier, &binary);
			trans_model.Read(ki.Stream(), binary);
		}

	    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldi(fst_rspecifier);
	    fst::SymbolTable *word_syms = NULL;
	    if (!(word_syms = fst::SymbolTable::ReadText(decoding_opts.word_syms_filename)))
	        KALDI_ERR << "Could not read symbol table from file " << decoding_opts.word_syms_filename;

		char mq_name[256];
		sprintf(mq_name, "/%d.mq", getpid());
		MessageQueue *mq_decodable = new MessageQueue();
		struct mq_attr decodable_attr;
		decodable_attr.mq_maxmsg = MAX_OUTPUT_MQ_MQXMSG;
		decodable_attr.mq_msgsize = MAX_OUTPUT_MQ_MSGSIZE;
		mq_decodable->Create(mq_name, &decodable_attr);

	    MessageQueue mq_forward;
	    mq_forward.Open(mqueue_wspecifier);

	    DecoderSync decoder_sync;

	    OnlineNnetFasterDecoder decoder(*decode_fst, decoder_opts);

	    OnlineNnetDecodingClass decoding(decoding_opts,
	    								&decoder, mq_decodable, &decoder_sync, trans_model,
										*word_syms, words_writer, alignment_writer);
		// The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<OnlineNnetDecodingClass> m(1, decoding);

	    MQSample mq_sample;
	    std::string utt_key;
	    Matrix<BaseFloat> utt_feat;
	    int chunk = 10;
	    memcpy(mq_sample.mq_callback_name, mq_name, MAX_FILE_PATH);
	    mq_sample.pid = getpid();
	    for (; !feature_reader.Done(); feature_reader.Next()) {
	    	utt_key = feature_reader.Key();
	    	utt_feat = feature_reader.Value();
	    	memcpy(mq_sample.uttt_key, utt_key.c_str(), MAX_KEY_LEN);
	    	mq_sample.dim = utt_feat.NumCols();
	    	decoder_sync.SetUtt(utt_key);

	    	for (int i = 0; i < utt_feat.NumRows(); i += chunk)
	    	{
	    		mq_sample.num_sample = utt_feat.NumRows()-i >= chunk ? chunk : utt_feat.NumRows()-i;
	    		memcpy((char*)mq_sample.sample, (char*)utt_feat.RowData(i), mq_sample.num_sample*utt_feat.NumCols()*sizeof(BaseFloat));
                
	    		if (utt_feat.NumRows()-i <= chunk)
	    			mq_sample.is_end = true;

                if (mq_forward.Send((char*)&mq_sample, sizeof(MQSample), 0) == -1)
                {
                    const char *c = strerror(errno);
                    if (c == NULL) { c = "[NULL]"; }
                    KALDI_ERR << "Error send message to queue " << mqueue_wspecifier << " , errno was: " << c;
                }

	    		if (utt_feat.NumRows()-i <= chunk)
                {
	    			decoder_sync.UtteranceWait();
                    KALDI_LOG << "Finish decode utterance : " << utt_key;
                }
	    	}
	    }

	    decoder_sync.Abort();

	    delete mq_decodable;
	    delete decode_fst;
	    delete word_syms;
	    return 0;
	} catch(const std::exception& e) {
	    std::cerr << e.what();
	    return -1;
	  }
} // main()
