// online0/online-message.h

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

#ifndef ONLINE0_ONLINE_MESSAGE_H_
#define ONLINE0_ONLINE_MESSAGE_H_

#include <sys/types.h>
#include <unistd.h>

#define MAX_FILE_PATH 256
#define MAX_KEY_LEN 256
#define MAX_SAMPLE_SIZE (10*40)
#define MAX_OUTPUT_SIZE (16*10240)

#define MAX_SAMPLE_MQ_MQXMSG (300)
#define MAX_SAMPLE_MQ_MSGSIZE (2048+1024)

#define MAX_OUTPUT_MQ_MQXMSG (4)
#define MAX_OUTPUT_MQ_MSGSIZE (16*10240*4+1024)

namespace kaldi {

struct MQSample {
	MQSample():pid(-1),is_end(false),num_sample(0),dim(0),prio(0){}
	pid_t pid;
	char mq_callback_name[MAX_FILE_PATH];
	char uttt_key[MAX_KEY_LEN];
	bool is_end;
	int  num_sample;
	int	 dim;
	int	 prio;
	float sample[MAX_SAMPLE_SIZE];
};

struct MQDecodable {
	MQDecodable():is_end(false),num_sample(0),dim(0),prio(0){}
	bool is_end;
	int  num_sample;
	int	 dim;
	int	 prio;
	float sample[MAX_OUTPUT_SIZE];
};


}

#endif /* ONLINE0_ONLINE_MESSAGE_H_ */
