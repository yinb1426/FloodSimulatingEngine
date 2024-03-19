#include <iostream>
#include <vector>
#include <string>
#include <time.h>

#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc.hpp>  

#include "VirtualPipelineModel.cuh"

//std::vector<double> ReadHeightMap(std::string srcPath)
//{
//	std::vector<double> map;
//	std::ifstream fin(srcPath);
//	double num;
//	while (!fin.eof())
//	{
//		fin >> num;
//		map.push_back(num);
//	}
//	fin.close();
//	return map;
//}
//
void WriteHeightMap(std::string dstPath, const std::vector<double> map)
{
	std::ofstream fout(dstPath);
	for (double num : map)
		fout << num << std::endl;
	fout.close();
}

void ShowImage(string title, vector<double> heightMap, unsigned int sizeX, unsigned int sizeY, int type)
{
	double maxHeight = *max_element(heightMap.begin(), heightMap.end());
	double minHeight = *min_element(heightMap.begin(), heightMap.end());
	double rangeHeight = (maxHeight - minHeight);
	vector<double> heightMapNormalized(heightMap);
	for (auto& height : heightMapNormalized)
		height = (height - minHeight) / rangeHeight * 255.0;
	cv::Mat vecForShow = cv::Mat(heightMapNormalized);
	cv::Mat matForShow = vecForShow.reshape(1, sizeY).clone();
	matForShow.convertTo(matForShow, CV_8UC3);
	cv::Mat colorImgForShow = cv::Mat::zeros(sizeY, sizeX, CV_8UC3);
	cv::applyColorMap(matForShow, colorImgForShow, type);
	cv::imshow(title, colorImgForShow);
	cv::waitKey();
}

using namespace std;

int main()
{
	unsigned int sizeX = 1405;
	unsigned int sizeY = 790;

	cout << "加载数据..." << endl;
	cv::Mat terrainHeightTif = cv::imread("resource/heightmap/FangshanTestArea.tif", cv::IMREAD_UNCHANGED);
	cv::Mat rainfallRateTif  = cv::imread("resource/heightmap/RainData_0.tif", cv::IMREAD_UNCHANGED);
	cv::Mat riverDepthTif   = cv::imread("resource/heightmap/RiverDepth60m.tif", cv::IMREAD_UNCHANGED);

	cv::flip(terrainHeightTif, terrainHeightTif, 0);
	cv::flip(rainfallRateTif, rainfallRateTif, 0);
	cv::flip(riverDepthTif, riverDepthTif, 0);

	vector<double> terrainHeight = terrainHeightTif.reshape(1, 1);
	vector<double> rainfallRate = rainfallRateTif.reshape(1, 1);
	vector<double> riverDepth = riverDepthTif.reshape(1, 1);
	cout << "加载完成！" << endl;

	vector<double> zeros(sizeX * sizeY);
	vector<Vec3> zerosVec3(sizeX * sizeY);

	//FloodSimulator fs = FloodSimulator(sizeX, sizeY, terrainHeight, zeros, zeros, riverDepth, rainfallRate, zeros, zerosVec3);
	VPM fs = VPM(sizeX, sizeY, terrainHeight, zeros, zeros, riverDepth, rainfallRate, zeros, zerosVec3);
	cout << "开始计算..." << endl;
	clock_t start, end;
	start = clock();
	fs.InitDevice();
	fs.SendAllDataToDevice();
	fs.PreparationForSimulaion();
	int step = 0;
	while (step < 500)
	{
		fs.RunSimulation(step);
		step++;	
	}
	fs.GetResultFromDevice();
	vector<double> wh = fs.GetWaterHeight();
	end = clock();
	cout << "计算完成！" << endl;
	fs.FreeAllData();	
	ShowImage("Fangshan", wh, fs.GetSizeX(), fs.GetSizeY(), cv::COLORMAP_OCEAN);
	cout << "输出结果..." << endl;
	// WriteHeightMap("E:/Desktop/CUDAResult2.txt", wh);
	cout << "输出完成！" << endl;
	cout << "计算时间：" << (double)(end - start) / CLOCKS_PER_SEC << endl;
	return 0;
}