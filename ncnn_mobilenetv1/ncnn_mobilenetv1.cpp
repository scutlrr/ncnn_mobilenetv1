#include<stdio.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include  <opencv2\opencv.hpp>
#include <math.h>
#include "net.h"
#include "mat.h"
#include "benchmark.h"

#include "mobilenetv1.id.h"
/*
	@brief 读取标签文件
	@param [input] strFileName 文件名
	@param [input] vecLabels 标签
*/
//using namespace std;
void read_labels(std::string strFileName, std::vector<std::string> &vecLabels)
//&vecLabels是传值引用
{
	std::ifstream in(strFileName);

	if (in)
	{
		std::string line;
		while (std::getline(in, line))
		{
			// std::cout << line << std::endl;
			vecLabels.push_back(line);
		}
	}
	else
	{
		std::cout << "label file is not exit!!!" << std::endl;
	}
}
/*
	@brief squeezenet_v_1			预测单张图的类别
	@param [input] strImagePath		图片路径
*/
void forward_squeezenet_v_1(std::string strImagePath)
{
	// data
	std::string strLabelPath = "../model/synset_words.txt";
	std::vector<std::string> vecLabel;
	read_labels(strLabelPath, vecLabel);

	const float mean_vals[3] = { 0.f, 0.f, 0.f };
	const float norm_vals[3] = { 0.0039, 0.0039, 0.0039 };
	cv::Mat matImage = cv::imread(strImagePath);
	// cv::resize(matImage, matImage, cv::Size(32, 32));
	if (matImage.empty())
	{
		printf("image is empty!!!\n");
	}

	const int nImageWidth = matImage.cols;
	const int nImageHeight = matImage.rows;

	// input and output
	ncnn::Mat matIn;
	ncnn::Mat matOut;
	// net
	ncnn::Net net;
	net.load_param_bin("../model/mobilenetv1.param.bin");
	net.load_model("../model/mobilenetv1.bin");

	const int nNetInputWidth = 32;
	const int nNetInputHeight = 32;

	// time
	double dStart = ncnn::get_current_time();

	// 判断图片大小是否和网络输入相同
	if (nNetInputWidth != nImageWidth || nNetInputHeight != nImageHeight)
	{
		matIn = ncnn::Mat::from_pixels_resize(matImage.data, ncnn::Mat::PIXEL_BGR, nImageWidth, nImageHeight, nNetInputWidth, nNetInputHeight);
	}
	else
	{
		matIn = ncnn::Mat::from_pixels(matImage.data, ncnn::Mat::PIXEL_BGR, nNetInputWidth, nNetInputHeight);
	}
	// 数据预处理
	matIn.substract_mean_normalize(mean_vals, norm_vals);

	// forward
	ncnn::Extractor ex = net.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(4);
	ex.input(mobilenetv1_param_id::BLOB_input_1, matIn);
	ex.extract(mobilenetv1_param_id::BLOB_248, matOut);

	printf("output_size: %d, %d, %d \n", matOut.w, matOut.h, matOut.c);

	// cls 10 class
	ncnn::Mat out_flatterned = matOut.reshape(matOut.w * matOut.h * matOut.c);
	std::vector<float> cls_scores;
	cls_scores.resize(out_flatterned.w);
	for (int i = 0; i < out_flatterned.w; i++)
	{
		cls_scores[i] = out_flatterned[i];
	}
	// return top class
	int top_class = 0;
	float max_score = 0.f;
	float sum = 0.f;
	for (size_t i = 0; i < cls_scores.size(); i++)
	{
		float s = cls_scores[i];
		sum += exp(s);
		if (s > max_score)
		{
			top_class = i;
			max_score = s;
			max_score = exp(max_score);
		}
	}
	double dEnd = ncnn::get_current_time();
	float acc = max_score / sum;

	printf("%d  score: %f   spend time: %.2f ms\n", top_class, acc, (dEnd - dStart));
	std::cout << vecLabel[top_class] << std::endl;

	cv::putText(matImage, vecLabel[top_class], cv::Point(5, 5), 1, 0.3, cv::Scalar(0, 0, 255), 1);
	cv::putText(matImage, " score:" + std::to_string(acc), cv::Point(5, 10), 1, 0.3, cv::Scalar(0, 0, 255), 1);
	cv::putText(matImage, " time: " + std::to_string(dEnd - dStart) + "ms", cv::Point(5, 15), 1, 0.3, cv::Scalar(0, 0, 255), 1);
	cv::imwrite("../images/result.jpg", matImage);
	cv::imshow("result", matImage);
	cv::waitKey(-1);
	net.clear();

}

int main()
{
	forward_squeezenet_v_1("../images/horse.jpg");
	printf("hello ncnn_mobilenetv1");
	system("pause");
}