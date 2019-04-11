#ifndef NN_CONV_H_
#define NN_CONV_H_

#include "nn_common.h"
#include "nn_activation.h"

namespace nn
{

struct conv2d_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Convolutional parameters (padding not used yet)
    // static const unsigned pad_top = 4;
    // static const unsigned pad_bottom = 5;
    // static const unsigned pad_left = 4;
    // static const unsigned pad_right = 5;
    static const unsigned in_height = 128;
    static const unsigned in_width = 128;
    static const unsigned n_chan = 9;
    static const unsigned filt_height = 10;
    static const unsigned filt_width = 10;
    static const unsigned n_filt = 4;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 128;
    static const unsigned out_width = 128;
};

template<typename CONFIG_T>
void convolution_operation(data_t img_lcl[IChan][ISize * ISize], data_t wgt_lcl[IChan][WSize * WSize], data_t out_lcl[OSize * OSize], int y, int x)
{
#pragma HLS INLINE
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

template<data_T, res_T, typename CONFIG_T>
conv2d()
{

}

};