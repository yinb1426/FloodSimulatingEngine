#pragma once
#include <iostream>
#include <vector>
#include "Utils.h"
using namespace std;

class FloodSimulator
{
public:
	unsigned int sizeX;
	unsigned int sizeY;
	double deltaT;
	double pipeLength;
	double gravity;
	double Ke;
	unsigned int numRainfallLayer;
	vector<double> terrainHeight;
	vector<double> buildingHeight;
	vector<double> damHeight;
	vector<double> surfaceHeight;
	vector<double> waterHeight;
	vector<double> rainfallRate;
	vector<double> drainRate;
	vector<Vec3> riverInflow;
	vector<Vec2> waterVelocity;
	vector<FlowField> flowField;
	vector<FlowField> newFlowField;
	double* gTerrainHeight;
	double* gBuildingHeight;
	double* gDamHeight;
	double* gSurfaceHeight;
	double* gWaterHeight;
	double* gRainfallRate;
	double* gDrainRate;
	Vec3* gRiverInflow;
	Vec2* gWaterVelocity;
	FlowField* gFlowField;
	FlowField* gNewFlowField;

	FloodSimulator(int _sizeX, int _sizeY, double _deltaT, double _pipeLength, double _gravity, double _Ke,
		vector<double> _terrainHeight, vector<double> _buildingHeight, vector<double> _damHeight, vector<double> _waterHeight,
		vector<double> _rainfallRate, vector<double> _drainRate, vector<Vec3> _riverInflow)
		: sizeX(_sizeX), sizeY(_sizeY), deltaT(_deltaT), pipeLength(_pipeLength), gravity(_gravity), Ke(_Ke),
		terrainHeight(_terrainHeight), buildingHeight(_buildingHeight), damHeight(_damHeight), waterHeight(_waterHeight),
		rainfallRate(_rainfallRate), drainRate(_damHeight), riverInflow(_riverInflow)
	{
		numRainfallLayer = rainfallRate.size() / (sizeX * sizeY);
		surfaceHeight = vector<double>(sizeX * sizeY);
		waterVelocity = vector<Vec2>(sizeX * sizeY);
		flowField = vector<FlowField>(sizeX * sizeY);
		newFlowField = vector<FlowField>(sizeX * sizeY);
	}

	FloodSimulator(int _sizeX, int _sizeY, vector<double> _terrainHeight, vector<double> _buildingHeight, vector<double> _damHeight, vector<double> _waterHeight,
		vector<double> _rainfallRate, vector<double> _drainRate, vector<Vec3> _riverInflow)
		: sizeX(_sizeX), sizeY(_sizeY), terrainHeight(_terrainHeight), buildingHeight(_buildingHeight), damHeight(_damHeight), waterHeight(_waterHeight),
		rainfallRate(_rainfallRate), drainRate(_damHeight), riverInflow(_riverInflow)
	{
		deltaT = 0.5;
		pipeLength = 15.0;
		gravity = 9.8;
		Ke = 0.00001;
		numRainfallLayer = rainfallRate.size() / (sizeX * sizeY);
		surfaceHeight = vector<double>(sizeX * sizeY);
		waterVelocity = vector<Vec2>(sizeX * sizeY);
		flowField = vector<FlowField>(sizeX * sizeY);
		newFlowField = vector<FlowField>(sizeX * sizeY);
	}

	~FloodSimulator() {}

	void SetDeltaT(double newDeltaT);
	double GetDeltaT() const;

	void SetPipeLength(double newPipeLength);
	double GetPipeLength() const;

	void SetGravity(double newGravity);
	double GetGravity() const;

	void SetKe(double newKe);
	double GetKe() const;

	void InitDevice();
	void SendAllDataToDevice();
	void RunSimulation(const unsigned int steps);
	void GetResultFromDevice();
	void FreeAllData();
};