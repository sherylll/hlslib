#include "../../nn_utils/nn_common.h"
//#include "../../nn_utils/nn_conv.h"
#include "ap_fixed.h"
#define IChan 1
#define ISize 28
#define WSize 4
#define OChan 4
#define OSize 13 // calculate output size based on the input size: w2=(w1-filter+2pad)/stride + 1
#define Stride 2
#define Padding 0 // no padding

//#include <ap_fixed.h>
typedef ap_fixed<8,4> data_t;
//typedef float data_t;

//struct conv0: nn::conv2d_config
//{
//    // Internal data type definitions
//    typedef data_t bias_t;
//    typedef data_t weight_t;
//    typedef data_t accum_t;
//
//    static const unsigned in_height = ISize;
//    static const unsigned in_width = ISize;
//    static const unsigned in_chan = IChan;
//    static const unsigned filt_height = WSize;
//    static const unsigned filt_width = WSize;
//    static const unsigned n_filt = OChan; // number of output channel
//    static const unsigned stride_height = Stride;
//    static const unsigned stride_width = Stride;
//    static const unsigned out_height = OSize;
//    static const unsigned out_width = OSize;
//};

struct fc0
{
    typedef data_t bias_t;
    typedef data_t weight_t;
    typedef data_t accum_t;
	static const unsigned n_in = OSize*OSize*OChan;
    static const unsigned n_out = 32;
};

struct fc1
{
    typedef data_t bias_t;
    typedef data_t weight_t;
    typedef data_t accum_t;
	static const unsigned n_in = 32;
    static const unsigned n_out = 10;
};

void conv2d(
    data_t image[IChan * ISize * ISize], // Read-Only Image
    data_t classes[10]                   // testing with MNIST
);
