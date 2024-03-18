#include <iostream>
#include <vector>
#include <string>
#include <time.h>

#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc.hpp>  

#include "FloodSimulator.cuh"
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
//void WriteHeightMap(std::string dstPath, const std::vector<double> map)
//{
//	std::ofstream fout(dstPath);
//	for (double num : map)
//		fout << num << std::endl;
//	fout.close();
//}

using namespace std;

int main()
{
	unsigned int sizeX = 1405;
	unsigned int sizeY = 790;

	cout << "��������..." << endl;
	cv::Mat terrainHeightTif = cv::imread("resource/heightmap/FangshanTestArea.tif", cv::IMREAD_UNCHANGED);
	cv::Mat rainfallRateTif  = cv::imread("resource/heightmap/RainData_0.tif", cv::IMREAD_UNCHANGED);
	cv::Mat riverDepthTif   = cv::imread("resource/heightmap/RiverDepth60m.tif", cv::IMREAD_UNCHANGED);

	cv::flip(terrainHeightTif, terrainHeightTif, 0);
	cv::flip(rainfallRateTif, rainfallRateTif, 0);
	cv::flip(riverDepthTif, riverDepthTif, 0);

	vector<double> terrainHeight = terrainHeightTif.reshape(1, 1);
	vector<double> rainfallRate = rainfallRateTif.reshape(1, 1);
	vector<double> riverDepth = riverDepthTif.reshape(1, 1);
	cout << "������ɣ�" << endl;

	vector<double> zeros(sizeX * sizeY);
	vector<Vec3> zerosVec3(sizeX * sizeY);

	FloodSimulator fs = FloodSimulator(sizeX, sizeY, terrainHeight, zeros, zeros, riverDepth, rainfallRate, zeros, zerosVec3);
	cout << "��ʼ����..." << endl;
	clock_t start, end;
	start = clock();
	fs.InitDevice();
	fs.SendAllDataToDevice();
	fs.PreparationForSimulaion();
	int step = 0;
	while (step < 500)
	{
		fs.RunSimulation(step);
		++step;
	}
	fs.GetResultFromDevice();
	fs.FreeAllData();
	end = clock();
	cout << "������ɣ�" << endl;
	cout << "������..." << endl;
	//WriteHeightMap("E:/Desktop/CUDAResult.txt", fs.waterHeight);
	cout << "�����ɣ�" << endl;
	cout << "����ʱ�䣺" << (double)(end - start) / CLOCKS_PER_SEC << endl;
	return 0;
}