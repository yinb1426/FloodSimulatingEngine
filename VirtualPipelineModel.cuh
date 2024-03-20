#pragma once
#include "Model.h"
#include "Utils.h"

class VPM final : public Model
{
protected:
	double pipeLength;
	double gravity;
	double Ke;
	size_t numRainfallLayer;
	vector<double> buildingHeight;
	vector<double> damHeight;
	vector<double> surfaceHeight;
	vector<double> rainfallRate;
	vector<double> drainRate;
	vector<Vec3> riverInflow;
	vector<Vec2> waterVelocity;
	vector<FlowField> flowField;
	vector<FlowField> newFlowField;
	double* gBuildingHeight;
	double* gDamHeight;
	double* gSurfaceHeight;
	double* gRainfallRate;
	double* gDrainRate;
	Vec3* gRiverInflow;
	Vec2* gWaterVelocity;
	FlowField* gFlowField;
	FlowField* gNewFlowField;

public:
	VPM(size_t _sizeX, size_t _sizeY, double _deltaT, double _pipeLength, double _gravity, double _Ke,
		vector<double> _terrainHeight, vector<double> _buildingHeight, vector<double> _damHeight, vector<double> _waterHeight,
		vector<double> _rainfallRate, vector<double> _drainRate, vector<Vec3> _riverInflow)
		: Model(_sizeX, _sizeY, _deltaT, _terrainHeight, _waterHeight), pipeLength(_pipeLength), gravity(_gravity), Ke(_Ke),
		buildingHeight(_buildingHeight), damHeight(_damHeight), rainfallRate(_rainfallRate), drainRate(_damHeight), riverInflow(_riverInflow)
	{
		numRainfallLayer = rainfallRate.size() / (sizeX * sizeY);
		surfaceHeight = vector<double>(sizeX * sizeY);
		waterVelocity = vector<Vec2>(sizeX * sizeY);
		flowField = vector<FlowField>(sizeX * sizeY);
		newFlowField = vector<FlowField>(sizeX * sizeY);
	}

	VPM(int _sizeX, int _sizeY, vector<double> _terrainHeight, vector<double> _buildingHeight, vector<double> _damHeight, vector<double> _waterHeight,
		vector<double> _rainfallRate, vector<double> _drainRate, vector<Vec3> _riverInflow)
		: Model(_sizeX, _sizeY, 0.5, _terrainHeight, _waterHeight), buildingHeight(_buildingHeight), damHeight(_damHeight),
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

public:
	void SetPipeLength(const double newPipeLength);
	double GetPipeLength() const;

	void SetGravity(const double newGravity);
	double GetGravity() const;

	void SetKe(const double newKe);
	double GetKe() const;

	void InitDevice() override final;
	void SendAllDataToDevice() override final;
	void PreparationForSimulaion() override final;
	void RunSimulation(const size_t step) override final;
	void GetResultFromDevice() override final;
	void FreeAllData() override final;
};