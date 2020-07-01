#include <iostream>
#include <vector>
#define _USE_MATH_DEFINES
//Thread building blocks library
#include <tbb/task_scheduler_init.h>
//Free Image library
#include <FreeImagePlus.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <cstdlib>

#include <math.h>
#include <random>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <functional>
#include <string>
using namespace std;
using namespace tbb;
using namespace std::chrono;

void compare(int numElemems, float *inBuff1, float *inBuff2, float *outBuffer, fipImage &outputImage,  char *image)
{
    for (uint64_t g = 0; g < numElemems; g++)
    {
        if (inBuff1[g] == inBuff2[g])
        {
            outBuffer[g] = 0;
        }
        else
        {
            outBuffer[g] = 1;
        }
        
    }
    outputImage.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
    outputImage.convertTo24Bits();
    outputImage.save(image);
}
//gausian filter
float Gaussian2D(float x, float y, float sigma)
{
    float PI = 3.14159;
    return 1.0f / (2.0f *float(PI)*sigma*sigma) * exp(-((x*x + y*y) / (2.0f * sigma*sigma)));
}
void kernalCreator(float kernel[][5], float(gaus)(float, float, float), float sigma)
{
    //float sigma = 10.0f;
    float sum = 0.0f;
    //Generates the kernel
    for (int x = -2; x <= 2; x++)
    {
        for (int y = -2; y <= 2; y++)
        {
            kernel[x + 2][y + 2] = gaus(x, y, sigma);
            sum += kernel[x + 2][y + 2];
        }
    }
    //Normalises the kernel
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
            kernel[i][j] /= sum;
    }

}

void gaussian_blur(float* outputBuffer, float* inputBuffer, int height, int width, float sigma)
{
    const int kSize = 5;

    //vector<vector<float>> kernel = kernalCreator(kSize, kSize);
    float kernel[kSize][kSize];
    kernalCreator(kernel, Gaussian2D, sigma);
   // kSize = kernel.size;
    
    int kernelHalf = kSize / 2;

    for (auto y = 0; y < height; y++)
    {
        for (auto x = 0; x < width; x++)
        {
            for (auto i = -kernelHalf; i <= kernelHalf; i++)
            {
                for (auto j = -kernelHalf; j <= kernelHalf; j++) // all these for loops are a boundry check
                {
                    if ((y + i) > 0 && (y + i) < height && (x + j) > 0 && (x + j) < width)
                    {
                        outputBuffer[y * width + x] += inputBuffer[(y + i) * width + (x + j) - 1] * kernel[j + kernelHalf][i + kernelHalf] +
                            inputBuffer[(y + i) * width + (x + j)] * kernel[j + kernelHalf][i + kernelHalf] +
                            inputBuffer[(y + i) * width + (x + j) + 1] * kernel[j + kernelHalf][i + kernelHalf]; //this gets the average pixel colour and adds it to our ouput buffer.
                    }
                }
            }

        }
    }
    std::cout << "Blurring done. Hopefully. " << endl;
}
void binary_threshold(float* outputBuffer, int threshold)
{
    fipImage inputImage;
    inputImage.load("stage2_blurred.png");

    auto width = inputImage.getWidth();
    auto height = inputImage.getHeight();

    RGBQUAD rgb;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            inputImage.getPixelColor(x, y, &rgb);
            if (abs(rgb.rgbRed - 0) >= threshold &&
                abs(rgb.rgbGreen - 0) >= threshold &&
                abs(rgb.rgbBlue - 0) >= threshold)
            {
                outputBuffer[y * width + x]=255;
            }
        }
        
    }
    std::cout << "Binary threshold complete" << endl;
}

void part1_combine()
{
    cout << "Stage one start" << endl;
    fipImage inTop1;
    fipImage inTop2;
    fipImage inBot1;
    fipImage inBot2;
    fipImage inFinish1, inFinish2;

    inBot1.load("../Images/render_bottom_1.png"); //CHECK FILETYPE DAMNIT
    inBot1.convertToFloat();
    float* inBuff1 = (float*)inBot1.accessPixels();

    inBot2.load("../Images/render_bottom_2.png");
    inBot2.convertToFloat();
    float* inBuff2 = (float*)inBot2.accessPixels();

    inTop1.load("../Images/render_top_1.png");
    inTop1.convertToFloat();
    float* inBuff3 = (float*)inTop1.accessPixels();

    inTop2.load("../Images/render_top_2.png");
    inTop2.convertToFloat();
    float* inBuff4 = (float*)inTop2.accessPixels();

    int width = inBot1.getWidth();
    int height = inBot1.getHeight();

    fipImage outputImage, outputImage2, finishedOutput;
    outputImage = fipImage(FIT_FLOAT, width, height, 32);
    outputImage2 = fipImage(FIT_FLOAT, width, height, 32);
    finishedOutput = fipImage(FIT_FLOAT, width, height, 32);

    float* outputBuffer = (float*)outputImage.accessPixels();
    float* outputBuffer2 = (float*)outputImage2.accessPixels();
    float* outputBuffer3 = (float*)finishedOutput.accessPixels();

    int numElements = width * height;

    auto start = std::chrono::high_resolution_clock::now();
    compare(numElements, inBuff1, inBuff2, outputBuffer, outputImage, "Stage1_bottom.jpg");
    compare(numElements, inBuff3, inBuff4, outputBuffer2, outputImage2, "Stage1_top.jpg");


    inFinish1.load("Stage1_bottom.jpg"); //CHECK FILETYPE DAMNIT
    inFinish1.convertToFloat();
    float* inBuff5 = (float*)inFinish1.accessPixels();

    inFinish2.load("Stage1_top.jpg");
    inFinish2.convertToFloat();
    float* inBuff6 = (float*)inFinish2.accessPixels();

    compare(numElements, inBuff5, inBuff6, outputBuffer3, finishedOutput, "Stage1_Finish.jpg");
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = finish - start;
    cout << "Stage one finished in "<< time.count() << " seconds" << endl;  
}
void part2_sequential_blurring(float sigma)
{
    cout << "----------------------------------------------" << endl;
    cout << "Stage two start" << endl;
    fipImage inputImage;
    inputImage.load("Stage1_Finish.jpg");
    inputImage.convertToFloat();
    float* inBuff = (float*)inputImage.accessPixels();

    const int width = inputImage.getWidth();
    const int height = inputImage.getHeight();

    fipImage outputImage_blurred, outputImage_threshold;

    outputImage_blurred = fipImage(FIT_FLOAT, width, height, 32);
    outputImage_threshold = fipImage(FIT_FLOAT, width, height, 32);

    float* outputBuffer_blurred = (float*)outputImage_blurred.accessPixels();
    float* outputBuffer_threshold = (float*)outputImage_threshold.accessPixels();

    auto start = std::chrono::high_resolution_clock::now();
    gaussian_blur(outputBuffer_blurred, inBuff, height, width, sigma);
    outputImage_blurred.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
    outputImage_blurred.convertTo24Bits();
    outputImage_blurred.save("stage2_blurred.png");


   

    binary_threshold(outputBuffer_threshold, 100);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = finish - start;

    outputImage_threshold.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
    outputImage_threshold.convertTo24Bits();
    outputImage_threshold.save("stage2_threshold.png");
    cout << "Stage two finished in " << time.count() << " seconds" << endl;
    
    
    

}
void part2_Parallel(float sigma)
{
    cout << "----------------------------------------------" << endl;
    cout << "Stage two start" << endl;
    fipImage inputImage;
    inputImage.load("Stage1_Finish.jpg");
    inputImage.convertToFloat();
    float* inputBuffer = (float*)inputImage.accessPixels();

    const int width = inputImage.getWidth();
    const int height = inputImage.getHeight();

    fipImage outputImageBlurred, outputImageThreshold;
    outputImageBlurred = fipImage(FIT_FLOAT, width, height, 32);
    outputImageThreshold = fipImage(FIT_FLOAT, width, height, 32);

    float* outputBufferBlurred = (float*)outputImageBlurred.accessPixels();
    float* outputBufferThreshold = (float*)outputImageThreshold.accessPixels();

    const int kSize = 5;
    float kernel[kSize][kSize];
    auto start = std::chrono::high_resolution_clock::now();
    kernalCreator(kernel, Gaussian2D, sigma);
    int kernelHalf = kSize / 2;

    parallel_for(blocked_range2d<int, int>(0, height, 0, width), [=](const blocked_range2d<int, int>& range)
    {
        auto y1 = range.rows().begin();
        auto y2 = range.rows().end();
        auto x1 = range.cols().begin();
        auto x2 = range.cols().end();

        for (auto y =y1; y < y2; y++)
        {
            for (auto x = x1; x < x2; x++)
            {
                for (int i = -kernelHalf; i <= kernelHalf; i++)
                {
                    for (int j = -kernelHalf; j <= kernelHalf; j++)
                    {
                        if ((y + i) > 0 && (y + i) < height && (x + j) > 0 && (x + j) < width)
                        {
                            outputBufferBlurred[y * width + x] += inputBuffer[(y + i) * width + (x + j) - 1] * kernel[j + kernelHalf][i + kernelHalf] +
                                inputBuffer[(y + i) * width + (x + j)] * kernel[j + kernelHalf][i + kernelHalf] +
                                inputBuffer[(y + i) * width + (x + j) + 1] * kernel[j + kernelHalf][i + kernelHalf];
                        }
                    }
                }
            }
        }


    });
    outputImageBlurred.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
    outputImageBlurred.convertTo24Bits();
    outputImageBlurred.save("stage2_blurred.png");
    std::cout << "Parallel blurring probably done." << endl;

    
    fipImage inputImage2;
    inputImage2.load("stage2_blurred.png");

    auto threshWidth = inputImage2.getWidth();
    auto threshHeight = inputImage2.getHeight();

    int threshold = 20;

    parallel_for(blocked_range2d<int, int>(0, threshHeight, 0, threshWidth), [&](blocked_range2d<int, int>& range)
    {
        auto y1 = range.rows().begin();
        auto y2 = range.rows().end();
        auto x1 = range.cols().begin();
        auto x2 = range.cols().end();
        RGBQUAD rgb;

        for (int y = y1; y < y2; y++)
        {
            for (int x = x1; x < x2; x++)
            {
                inputImage2.getPixelColor(x, y, &rgb);
                if (abs(rgb.rgbRed - 0) >= threshold &&
                    abs(rgb.rgbGreen - 0) >= threshold &&
                    abs(rgb.rgbBlue - 0) >= threshold) //pixels that are not black into white
                {
                    outputBufferThreshold[y * width + x] = 255;
                }
            }

        }
    });
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    outputImageThreshold.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
    outputImageThreshold.convertTo24Bits();
    outputImageThreshold.save("stage2_threshold.png");

    std::cout << "Part 2 parallel : " << elapsed.count() << " seconds" << std::endl;

}
void part3_sequential_task()
{
    std::cout << "----------------------------------------------" << endl;
    std::cout << "Stage three start" << endl;
    fipImage inputTopImage, inputImageThreshold;
    inputTopImage.load("../Images/render_top_1.png");
    inputImageThreshold.load("stage2_threshold.png");

    const int width = inputImageThreshold.getWidth();
    const int height = inputImageThreshold.getHeight();

    int whitePixels = 0;
    int totalPixels = 0;

    fipImage outputImageFilter;
    outputImageFilter = fipImage(FIT_BITMAP, width, height, 24);

    RGBQUAD rgb, rgb2;
    auto start = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            totalPixels++;
            inputImageThreshold.getPixelColor(x, y, &rgb);
            inputTopImage.getPixelColor(x, y, &rgb2);
            
            if (float(rgb.rgbRed) == 255 &&
                float(rgb.rgbGreen) == 255 &&
                float(rgb.rgbBlue) == 255)
            {
                whitePixels++;
                float newRed = abs(255 - float(rgb2.rgbRed));
                float newGreen = abs(255 - float(rgb2.rgbGreen));
                float newBlue = abs(255 - float(rgb2.rgbBlue));

                rgb2.rgbRed = newRed;
                rgb2.rgbGreen = newGreen;
                rgb2.rgbBlue = newBlue;
                outputImageFilter.setPixelColor(x, y, &rgb2);

            }
            else
            {
                outputImageFilter.setPixelColor(x, y, &rgb2);
            }
        }
    }


    float whitePercent = float(whitePixels) / float(totalPixels) * 100;
    std::cout << "Number of White Pixels = " << whitePixels << endl;
    std::cout << "Number of Pixels Total = " << totalPixels << endl;
    std::cout << "Percentage of white Pixels : "<<  whitePercent<<"%" << endl;
    


    outputImageFilter.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
    outputImageFilter.convertTo24Bits();
    outputImageFilter.save("stage3_filter.png");
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Part 3 Sequential: " << elapsed.count() << " seconds" << std::endl;

}

void part3_parallel()
{
    std::cout << "----------------------------------------------" << endl;
    std::cout << "Stage three start" << endl;
    fipImage inputTop, inputThreshold;
    inputTop.load("../Images/render_top_1.png");
    inputThreshold.load("stage2_threshold.png");

    const int width = inputThreshold.getWidth();
    const int height = inputThreshold.getHeight();

    fipImage outputFilter;
    outputFilter = fipImage(FIT_BITMAP, width, height, 24);
    int whitePixels = 0;
    int totalPixels = 0;
    auto start = std::chrono::high_resolution_clock::now();
    parallel_for(blocked_range2d<int, int>(0, height, 0, width), [&](blocked_range2d<int, int>& range)
    {
        auto y1 = range.rows().begin();
        auto y2 = range.rows().end();
        auto x1 = range.cols().begin();
        auto x2 = range.cols().end();
        RGBQUAD rgb, rgb2;
        for (int y = y1; y < y2; y++)
        {
            for (int x = x1; x < x2; x++)
            {
                totalPixels++;
                inputThreshold.getPixelColor(x, y, &rgb);
                inputTop.getPixelColor(x, y, &rgb2);

                if (float(rgb.rgbRed) == 255 &&
                    float(rgb.rgbGreen) == 255 &&
                    float(rgb.rgbBlue) == 255)
                {
                    whitePixels++;
                    float newRed = abs(255 - float(rgb2.rgbRed));
                    float newGreen = abs(255 - float(rgb2.rgbGreen));
                    float newBlue = abs(255 - float(rgb2.rgbBlue));

                    rgb2.rgbRed = newRed;
                    rgb2.rgbGreen = newGreen;
                    rgb2.rgbBlue = newBlue;
                    outputFilter.setPixelColor(x, y, &rgb2);

                }
                else
                {
                    outputFilter.setPixelColor(x, y, &rgb2);
                }
            }
        }
    });
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    float whitePercent = float(whitePixels) / float(totalPixels) * 100;
    std::cout << "Number of White Pixels = " << whitePixels << endl;
    std::cout << "Number of Pixels Total = " << totalPixels << endl;
    std::cout << "Percentage of white Pixels : " << whitePercent << "%" << endl;



    outputFilter.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
    outputFilter.convertTo24Bits();
    outputFilter.save("stage3_filter.png");
    std::cout << "Part 3 Parallel : " << elapsed.count() << " seconds" << std::endl;

}
int main()
{
	int nt = task_scheduler_init::default_num_threads();
	task_scheduler_init T(nt);

	//Part 1 (Image Comparison): -----------------DO NOT REMOVE THIS COMMENT----------------------------//
   
    part1_combine();

   


	//Part 2 (Blur & post-processing): -----------DO NOT REMOVE THIS COMMENT----------------------------//


    //part2_sequential_blurring(10.0f);
    part2_Parallel(10.0f);

	//Part 3 (Image Mask): -----------------------DO NOT REMOVE THIS COMMENT----------------------------//

   // part3_sequential_task();
    part3_parallel();
	return 0;
}