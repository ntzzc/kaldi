// thread/kaldi-message-queue.h

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

#ifndef THREAD_KALDI_MESSAGE_QUEUE_H_
#define THREAD_KALDI_MESSAGE_QUEUE_H_

#include "base/kaldi-error.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <mqueue.h>		/* Posix message queues */

#define	FILE_MODE	(S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)

namespace kaldi {

class MessageQueue {

public:
	MessageQueue(std::string pathname, int oflag, mode_t mode = NULL, struct mq_attr attr = NULL):
		pathname_(pathname), oflag_(oflag), mode_(mode), attr_(attr)
	{
		if ( (mqd_ = mq_open(pathname.c_str(), oflag, mode, attr)) == (mqd_t)-1)
			KALDI_ERR << "Cannot open message queue for " << pathname;
	}

	MessageQueue(): pathname_(""), mqd_(NULL), oflag_(0), mode_(NULL), attr_(NULL){}

	~MessageQueue()
	{
		if (mq_unlink(pathname_.c_str()) == -1)
			KALDI_ERR << "Unlink message queue error";
	}

	void Open(std::string pathname, int oflag = O_RDWR)
	{
		if ( (mqd_ = mq_open(pathname.c_str(), oflag, FILE_MODE, NULL)) == (mqd_t)-1)
					KALDI_ERR << "Cannot open message queue for " << pathname;
	}

	void Create(std::string pathname, struct mq_attr attr, int oflag = O_RDWR | O_CREAT | O_EXCL)
	{

		while ( (mqd_ = mq_open(pathname.c_str(), oflag, FILE_MODE, attr)) == (mqd_t)-1)
		{
			if (mq_unlink(pathname_.c_str()) == -1)
				KALDI_ERR << "Cannot create message queue for " << pathname;
		}
	}

	int Send(char *ptr, size_t len, unsigned int prio)
	{
		int n;
		n = mq_send(mqd_, ptr, len, prio);
		return n;
	}

	ssize_t Receive(char *ptr, size_t len, unsigned int *prio)
	{
		ssize_t	n;

		n = mq_receive(mqd_, ptr, len, prio);
		return n;
	}

	void Getattr(struct mq_attr *mqstat)
	{
		if (mq_getattr(mqd_, mqstat) == -1)
			KALDI_ERR << " Message queue getattr error";
	}

	void Setattr(struct mq_attr *mqstat, struct mq_attr *omqstat)
	{
		if (mq_setattr(mqd_, mqstat, omqstat) == -1)
			KALDI_ERR << " Message queue setattr error";
	}

private:

	std::string pathname_;
	mqd_t	mqd_;
	int oflag_;
	int mode_;
	mq_attr attr_;
};


}

#endif /* THREAD_KALDI_MESSAGE_QUEUE_H_ */
