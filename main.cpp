#include <iostream>
#include <vector>
#include <string>
#include <time.h>
#include "FloodSimulator.cuh"
#include "VirtualPipelineModel.cuh"

std::vector<double> ReadHeightMap(std::string srcPath)
{
	std::vector<double> map;
	std::ifstream fin(srcPath);
	double num;
	while (!fin.eof())
	{
		fin >> num;
		map.push_back(num);
	}
	fin.close();
	return map;
}

void WriteHeightMap(std::string dstPath, const std::vector<double> map)
{
	std::ofstream fout(dstPath);
	for (auto& num : map)
		fout << num << std::endl;
	fout.close();
}

using namespace std;

int main()
{
	unsigned int sizeX = 1405;
	unsigned int sizeY = 790;

	cout << "加载数据..." << endl;
	vector<double> terrainHeight = ReadHeightMap("resource/heightmap/terrainHeight.txt");
	vector<double> rainfallRate = ReadHeightMap("resource/heightmap/rainfallRate.txt");
	vector<double> riverDepth = ReadHeightMap("resource/heightmap/riverDepth.txt");
	cout << "加载完成！" << endl;

	vector<double> zeros(sizeX * sizeY);
	vector<Vec3> zerosVec3(sizeX * sizeY);

	FloodSimulator fs = FloodSimulator(sizeX, sizeY, terrainHeight, zeros, zeros, riverDepth, rainfallRate, zeros, zerosVec3);
	cout << "开始计算..." << endl;
	clock_t start, end;
	start = clock();
	fs.InitDevice();
	fs.SendAllDataToDevice();
	fs.RunSimulation(500);
	fs.GetResultFromDevice();
	fs.FreeAllData();
	end = clock();
	cout << "计算完成！" << endl;
	cout << "输出结果..." << endl;
	//WriteHeightMap("E:/Desktop/CUDAResult2.txt", fs.waterHeight);
	cout << "输出完成！" << endl;
	cout << "计算时间：" << (double)(end - start) / CLOCKS_PER_SEC << endl;
	return 0;
}