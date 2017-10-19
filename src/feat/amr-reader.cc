// feat/amr-reader.cc
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

#include <cstdio>
#include <limits>
#include <sstream>
#include <vector>
#include <fstream>
#include <interf_dec.h>

#include "feat/amr-reader.h"
#include "base/kaldi-error.h"
#include "base/kaldi-utils.h"


namespace kaldi {


void AmrData::Read(std::istream &is) {

  /* From WmfDecBytesPerFrame in dec_input_format_tab.cpp */
  const int sizes[] = { 12, 13, 15, 17, 19, 20, 26, 31, 5, 6, 5, 5, 0, 0, 0, 0 };

  data_.Resize(0);  // clear the data.
  char header[6];
  void* amr ;
  int n;
  is.read(header, 6);
  n = is.gcount();

  if (n != 6 || memcmp(header, "#!AMR\n", 6)) {
    fprintf(stderr, "Bad header\n");
    return;
  }  
  amr = Decoder_Interface_init();
  int batch = 0;
  while (1) {

    data_.Resize(160+batch*160, kCopyData);
    char buffer[500];
    int size, i ;
    int16_t outbuffer[160];
    /* Read the mode byte */
    is.read(buffer, 1);
    n = is.gcount();
    if (n <= 0)
      break;
    /* Find the packet size */
    size = sizes[((uint8_t)buffer[0] >> 3) & 0x0f];
    is.read(buffer+1, size);
    n = is.gcount();
    if (n != size)
      break;
    /* Decode the packet */
    Decoder_Interface_Decode(amr, (uint8_t*)buffer, outbuffer, 0);

    /* Convert to little endian and write to wav */

    for (i = 0; i < 160; i++) {
      data_(batch*160 + i) = BaseFloat(outbuffer[i]);
    }
    batch++;
  }
  Decoder_Interface_exit(amr);
  return ;
}
}  // end namespace kaldi
