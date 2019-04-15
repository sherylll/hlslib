#include "conv2d.h"

#include "conv2d_weights/w_conv.h"
#include "conv2d_weights/b_conv.h"
#include "conv2d_weights/w_fc0.h"
#include "conv2d_weights/b_fc0.h"
#include "conv2d_weights/w_fc1.h"
#include "conv2d_weights/b_fc1.h"

// void copy_weight(data_t *weight, data_t wgt_lcl[InChan][WSize * WSize])
// {
// #pragma HLS INLINE
//     // Calculate each work_item's weight matrix location
//     int stride = InChan * WSize * WSize;

//     // Each work_item copies weight matrix from DDR to local buffer
//     readWt: for(int itr = 0, i = 0, j = 0; itr < InChan * WSize * WSize; itr++,j++) {
//     #pragma HLS PIPELINE II=1
//         if(j == WSize * WSize) {j = 0; i++;}
//         wgt_lcl[i][j] = weight[stride + itr];
//     }
// }

void flatten_and_relu(data_t out[OChan * OHeight * OWidth], data_t out_lcl[OChan][OHeight * OWidth])
{
#pragma HLS pipeline
    // Calculate each work_item's result update location
    static int stride = OHeight * OWidth;
// Work_item updates output filter/image in DDR
writeOut:
    for (int o = 0; o < OChan; o++)
    {
//#pragma HLS PIPELINE
        for (int itr = 0; itr < OHeight * OWidth; itr++)
        {
//#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable=out inter false
        	data_t temp = out_lcl[o][itr];
            if (temp > 0)
                out[itr + o*stride] = temp;
            else
                out[itr + o*stride] = 0;
        }
    }
}

void convolution_operation(data_t img_lcl[IChan][IHeight * IWidth], data_t wgt_lcl[IChan][WWidth * WHeight], data_t out_lcl[OHeight * OWidth], int y, int x)
{
#pragma HLS PIPELINE
    // Holds temporary accumulator values
    data_t acc[IChan][WHeight][WWidth];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1

    // Holds Image Padding Boundary Check Variables
    int xVal_base = x * WStride - Padding;
    int yVal = y * HStride - Padding;

// Runs over filter window
convYaxis:
    for (int i = 0; i < WHeight; i++, yVal++)
    {
    // Runs over filter window
    convXaxis:
        for (int j = 0, xVal = xVal_base; j < WWidth; j++, xVal++)
        {
//#pragma HLS PIPELINE II = 1
        // Runs over each of the input channels
        convInchan:
            for (int input = 0; input < IChan; input++)
            {
                // Convolution operation
                if (yVal >= 0 && yVal < IHeight && xVal >= 0 && xVal < IWidth)
                {
                    acc[input][i][j] = img_lcl[input][yVal * IWidth + xVal] *
                                       wgt_lcl[input][i * WWidth + j];
                }
                else
                {
                    acc[input][i][j] = 0;
                }
            }
        }
    }
    // Summation of temporary accumulator buffer
    data_t sum = 0;
accJ:
    for (int j = 0; j < WHeight; j++)
    {
    accK:
        for (int k = 0; k < WWidth; k++)
        {
#pragma HLS PIPELINE II = 1
        accI:
            for (int i = 0; i < IChan; i++)
            {
                sum += acc[i][j][k];
            }
        }
    }
    // Update output pixel
    out_lcl[y * OWidth + x] = sum;
}

void conv_1chan(data_t img_lcl[IChan][IHeight * IWidth], data_t w_conv_o[IChan][WHeight*WWidth], data_t out_lcl_o[OHeight*OWidth])
{
    outYaxis:
        for (int y = 0; y < OHeight; y++)
        {
        outXaxis:
            for (int x = 0; x < OWidth; x++)
            {
                // Perform convolution for the current 'pixel'
                convolution_operation(img_lcl, w_conv_o, out_lcl_o, y, x);
            }
        }
}

void small_cnn(
    data_t image[IChan * IHeight * IWidth], // Read-Only Image
    data_t classes[10]                   // testing with MNIST
)
{
//#pragma HLS interface axis port = image,classes

#pragma HLS ARRAY_PARTITION variable = w_conv complete dim=1
#pragma HLS ARRAY_PARTITION variable=b_conv complete

    // Local Buffer to Hold Input Image
    data_t img_lcl[OChan][IChan][IHeight * IWidth];
#pragma HLS ARRAY_PARTITION variable = img_lcl complete

    // Local Buffer to Hold Output Filters/Images
    data_t out_lcl[OChan][OHeight * OWidth];
#pragma HLS ARRAY_PARTITION variable=out_lcl complete
//#pragma HLS ARRAY_PARTITION variable=out_lcl factor=13 dim=2
// Burst Read Image
readImg:
    for (int itr = 0, i = 0, j = 0; itr < IChan * IHeight * IWidth; itr++, j++)
    {
#pragma HLS PIPELINE II = 1
        if (j == IHeight * IWidth)
        {
            j = 0;
            i++;
        }
        img_lcl[0][i][j] = image[itr];
        img_lcl[1][i][j] = image[itr];
        img_lcl[2][i][j] = image[itr];
        img_lcl[3][i][j] = image[itr];
    }

outChans:
    for (int o = 0; o < OChan; o++)
    {
#pragma HLS unroll
    	// copy_weights
    	// must use function for unroll to take effect
    	conv_1chan(img_lcl[o], w_conv[o], out_lcl[o]);
    }

    data_t out[OChan * OHeight * OWidth]; // Output Filters/Images
#pragma HLS ARRAY_PARTITION variable = out cyclic factor=4
    flatten_and_relu(out, out_lcl);

    data_t res0[32]; // Output Filters/Images
#pragma HLS ARRAY_PARTITION variable = res0 complete dim = 1

    nn::fc<data_t, data_t, fc0>(w_fc0, out, b_fc0,res0); 

    // relu
    for (int i =0; i<32; i++){
#pragma HLS pipeline
        if(res0[i] < 0)
            res0[i] = 0;
    }    
    
    nn::fc<data_t, data_t, fc1>(w_fc1, res0, b_fc1,classes);
    return;
}

