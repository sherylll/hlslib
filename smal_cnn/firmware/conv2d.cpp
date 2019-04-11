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

void flatten_and_relu(data_t out[OChan * OSize * OSize], data_t out_lcl[OChan][OSize * OSize])
{
#pragma HLS INLINE
    // Calculate each work_item's result update location
    static int stride = OSize * OSize;
// Work_item updates output filter/image in DDR
writeOut:
    for (int o = 0; o < OChan; o++)
    {
        for (int itr = 0; itr < OSize * OSize; itr++)
        {
#pragma HLS PIPELINE II = 1
            if (out_lcl[o][itr] > 0)
                out[itr + o*stride] = out_lcl[o][itr];
            else
                out[itr + o*stride] = 0;
        }
    }
}

void convolution_operation(data_t img_lcl[IChan][ISize * ISize], data_t wgt_lcl[IChan][WSize * WSize], data_t out_lcl[OSize * OSize], int y, int x)
{
//#pragma HLS INLINE
    // Holds temporary accumulator values
    data_t acc[IChan][WSize][WSize];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 1

    // Holds Image Padding Boundary Check Variables
    int xVal_base = x * Stride - Padding;
    int yVal = y * Stride - Padding;

// Runs over filter window
convYaxis:
    for (int i = 0; i < WSize; i++, yVal++)
    {
    // Runs over filter window
    convXaxis:
        for (int j = 0, xVal = xVal_base; j < WSize; j++, xVal++)
        {
#pragma HLS PIPELINE II = 1
        // Runs over each of the input channels
        convInchan:
            for (int input = 0; input < IChan; input++)
            {
                // Convolution operation
                if (yVal >= 0 && yVal < ISize && xVal >= 0 && xVal < ISize)
                {
                    acc[input][i][j] = img_lcl[input][yVal * ISize + xVal] *
                                       wgt_lcl[input][i * WSize + j];
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
    for (int j = 0; j < WSize; j++)
    {
    accK:
        for (int k = 0; k < WSize; k++)
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
    out_lcl[y * OSize + x] = sum;
}

void conv2d(
    data_t image[IChan * ISize * ISize], // Read-Only Image
    data_t classes[10]                   // testing with MNIST
)
{
//#pragma HLS interface axis port = image,classes
    // computes one channel only

    // Local Buffer to Hold Input Image
    data_t img_lcl[IChan][ISize * ISize];
#pragma HLS ARRAY_PARTITION variable = img_lcl complete dim = 1

    // Local Buffer to Hold Output Filters/Images
    data_t out_lcl[OChan][OSize * OSize];
#pragma HLS ARRAY_PARTITION variable=out_lcl complete dim=1

// Burst Read Image
readImg:
    for (int itr = 0, i = 0, j = 0; itr < IChan * ISize * ISize; itr++, j++)
    {
#pragma HLS PIPELINE II = 1
        if (j == ISize * ISize)
        {
            j = 0;
            i++;
        }
        img_lcl[i][j] = image[itr];
    }

outChans:
    for (int o = 0; o < OChan; o++)
    {
#pragma HLS unroll
    outYaxis:
        for (int y = 0; y < OSize; y++)
        {
#pragma HLS unroll
        outXaxis:
            for (int x = 0; x < OSize; x++)
            {
                // Perform convolution for the current 'pixel'
                convolution_operation(img_lcl, w_conv[o], out_lcl[o], y, x);
            }
        }
    }
    data_t out[OChan * OSize * OSize]; // Output Filters/Images
    flatten_and_relu(out, out_lcl);

    data_t res0[32]; // Output Filters/Images
#pragma HLS ARRAY_PARTITION variable = res0 complete dim = 1

    for (int i =0; i<OChan*OSize*OSize; i++){
        if(out[i] < 0)
            out[i] = 0;
    }    
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

