#include "time.h"
#include "opencv2/opencv.hpp"
#include "/usr/local/include/opencv2/core/cuda.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda_types.hpp"
#include "opencv2/core/cuda.inl.hpp"
#include <string>
#include <stdio.h>


using namespace cv;
using namespace std;

void ProccTimePrint( unsigned long Atime , string msg)
{
 unsigned long Btime=0;
 float sec, fps;
 Btime = getTickCount();
 sec = (Btime - Atime)/getTickFrequency();
 fps = 1/sec;
 printf("%s %.4lf(sec) / %.4lf(fps) \n", msg.c_str(),  sec, fps );
}

int main()
{
 unsigned long AAtime=0;

 //image load
 Mat img,img2, outImg, outimg2; 
 img = imread("drone.jpg",IMREAD_COLOR);//učitavanje slike


	


 

/*****cpu version meanshift****/
/*  
 AAtime = getTickCount();
 pyrMeanShiftFiltering(img, outImg, 30, 30, 3);
 ProccTimePrint(AAtime , "CPU:");
*/


 /*****cpu version mean shiftfiltering****/
/*
 cuda::GpuMat pimgGpu, imgGpu, outImgGpu;
 AAtime = getTickCount();
 pimgGpu.upload(img);
 //gpu meanshift only support 8uc4 type.
 cuda::cvtColor(pimgGpu, imgGpu, CV_BGR2BGRA);
 cuda::meanShiftFiltering(imgGpu, outImgGpu, 16.0, 64.0);
 outImgGpu.download(outimg2);
 ProccTimePrint(AAtime , "GPU mean-shift filtering:");

imshow("origin", img);

namedWindow("Display Image", CV_WINDOW_AUTOSIZE);
imshow("MeanShift Filter gpu", outimg2);
//imwrite("/home/nediljko/Desktop/cpp_test/aaxaa.jpg",outimg2);
//cuda::printShortCudaDeviceInfo(cuda::getDevice());
*/
{
Mat izlaz_5;
 cuda::GpuMat pimgGpu, imgGpu, outImgGpu;
 AAtime = getTickCount();
 pimgGpu.upload(img);
 //gpu meanshift only support 8uc4 type.

cuda::cvtColor(pimgGpu, imgGpu, CV_BGR2HSV,4); // CV_8UC4 jedini model za sada 
TermCriteria iteracija = TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 1000);
//(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 100);
//30,30,300

cuda::meanShiftSegmentation(imgGpu, izlaz_5, 30, 30, 5000,iteracija);
//cuda::threshold(imgGpu, izlaz_5, 150, 255, cv::THRESH_TOZERO_INV);

 cout << "Brzina GPU Meanshift segmentacije: ";
 ProccTimePrint(AAtime , "gpu");

imshow("GPU meanshift segmentacija", izlaz_5);
imwrite("/home/nediljko/Desktop/cpp_test/rez/aaaa2.jpg",izlaz_5);
waitKey(0);

 waitKey();
}
}
