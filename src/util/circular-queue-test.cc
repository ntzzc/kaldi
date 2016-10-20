// util/circular-queue-test.cc

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

#include "util/circular-queue.h"

namespace kaldi {

template<class T>
void TestCircularQueue()
{
	CircularQueue<T> queue;

	for (size_t j = 0; j < 50; j++) {
		queue.push(j);
	}

	for (size_t j = 0; j < 50; j++) {
		int i = queue.front();
		queue.pop();
		KALDI_ASSERT(i == j);
	}

	queue.clear();

	for (size_t j = 0; j < 50; j++) {
		queue.push(j);
	}

	for (size_t j = 0; j < 50; j++) {
		int i = queue.front();
		queue.pop();
		KALDI_ASSERT(i == j);
	}
}


}

int main() {
  using namespace kaldi;

  TestCircularQueue<int>();

  std::cout << "Test OK.\n";
}
