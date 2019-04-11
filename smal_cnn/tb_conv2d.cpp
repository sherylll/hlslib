#include "firmware/conv2d.h"
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

#define IMAGE_WIDTH 28
#define N_INPUTS IMAGE_WIDTH
#define N_OUTPUTS 10
#define TEST_SIZE 10

//// Software solution
//void convGolden(data_t *image, data_t *classes)
//{
//    data_t out[OChan * OSize * OSize]; // Output Filters/Images
//    data_t res0[32]; // Output Filters/Images
//
//    // Runs over output filters
//    for(int output = 0; output < OChan; output++){
//        // Runs over output pixel in Y-direction
//        for(int y = 0; y < OSize; y++){
//            // Runs over output pixel in X-direction
//            for(int x = 0; x < OSize; x++){
//                data_t acc = 0;
//                // Runs over each input channel of input feature map
//                for(int input = 0; input < IChan; input++){
//                    // Runs over filter window
//                    for(int i = 0; i < WSize; i++){
//                        // Runs over filter windows
//                        for(int j = 0; j < WSize; j++){
//                            // Calculate input padding boundaries
//                            int xVal = x*Stride + j-Padding, yVal = y*Stride + i-Padding;
//                            // Convolution operation
//                            if(yVal >= 0 && yVal < ISize && xVal >= 0 && xVal < ISize){
//                                acc +=  image[(input*ISize + yVal)*ISize + xVal] *
//                                    //    weight[((output*IChan + input)*WSize + i)*WSize + j];
//                                    w_conv[output][0][i*WSize + j];
//                            }
//                        }
//                        // Update each output pixel / output filter
//                        out[(output*OSize + y)*OSize + x] = acc;
//                    }
//                }
//            }
//        }
//    }
//
//    for (int i =0; i<OChan*OSize*OSize; i++){
//        if(out[i] < 0)
//            out[i] = 0;
//    }
//    nn::fc<data_t, data_t, fc0>(w_fc0, out, b_fc0,res0);
//
//    for (int i =0; i<32; i++){
//        if(res0[i] < 0)
//            res0[i] = 0;
//    }
//
//    nn::fc<data_t, data_t, fc1>(w_fc1, res0, b_fc1,classes);
//}

int max_likelihood(data_t y[N_OUTPUTS])
{
	int i_likely = 0;
	data_t y_max = 0;
	for (int i = 0; i < N_OUTPUTS; i++)
	{
		if (y[i] > y_max)
		{
			y_max = y[i];
			i_likely = i;
		}
	}
	return i_likely;
}

int read_to_array(char *path, data_t x_test[IMAGE_WIDTH*IMAGE_WIDTH], int *y_test)
{
	std::ifstream inFile;
	inFile.open(path);
	if (!inFile)
		return -1;
	if (inFile.get() == '#')
		inFile >> *y_test;
	for (int i = 0; i < IMAGE_WIDTH; i++)
	{
		for (int j = 0; j < IMAGE_WIDTH; j++)
		{
			inFile >> x_test[i*IMAGE_WIDTH + j];
		}
	}
	inFile.close();
	return 0;
}

int main()
{
	data_t probs[N_OUTPUTS];
	char x_str[10] = "";
	char path_cstr[100];

	data_t x_test[1*N_INPUTS*N_INPUTS];

	int y_test, counter;
	for (int im=0; im < TEST_SIZE; im ++){
		sprintf(x_str, "%d.txt", im);
		std::string image_path = "/home/asap2/hikari/vivado_prj_quant_lstm/test_images/";
		image_path += std::string(x_str);
		strcpy(path_cstr, image_path.c_str());
		if (read_to_array(path_cstr, x_test, &y_test) == 0){
			conv2d(x_test, probs);
			// convGolden(x_test, probs);
            // omitting softmax
			int y_pred = max_likelihood(probs);
			std::cout << im << " " << y_pred <<" "<< y_test<< std::endl;
			if (y_pred == y_test)
				counter++;
		}
		else
			std::cout << "failed to read file" << std::endl;
	}
	std::cout << counter;
}
