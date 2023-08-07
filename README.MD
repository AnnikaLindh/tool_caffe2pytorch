License: MIT (see LICENSE file)

This a helper tool to allow PyTorch to load Caffe weights in the format used by the Faster R-CNN in the Bottom-Up network for image captioning (https://github.com/peteanderson80/bottom-up-attention).

It has only been tested with my own projects, so there may be unfinished and/or error-prone parts in this repo that I did not use. If used in your own project then please see this only as a work-in-progress starting point that you need to complete on your own for it to work as expected with your own project.  

The code is based on the more general (but also unfinished) Caffe weight loader from:
https://github.com/vadimkantorov/caffemodel2pytorch

Proposal layer code based on code from:
https://github.com/intel/caffe/
