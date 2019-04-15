#include "../../nn_utils/nn_common.h"

#include "ap_fixed.h"
#define IChan 1
//#define ISize 28
#define IHeight 28
#define IWidth 28
//#define WSize 4
#define WHeight 4
#define WWidth 4
#define OChan 4
//#define OSize 13 // calculate output size based on the input size: w2=(w1-filter+2pad)/stride + 1
#define OHeight 13
#define OWidth 13
//#define Stride 2
#define HStride 2
#define WStride 2
#define Padding 0 // no padding

//typedef ap_fixed<8,4> data_t;
typedef float data_t;


struct fc0
{
    typedef data_t bias_t;
    typedef data_t weight_t;
    typedef data_t accum_t;
	static const unsigned n_in = OHeight*OWidth*OChan;
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

void small_cnn(
    data_t image[IChan * IHeight * IWidth], // Read-Only Image
    data_t classes[10]                   // testing with MNIST
);
