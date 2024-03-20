#pragma once
#include <vector>
using namespace std;

class Model
{
protected:
	size_t sizeX;
	size_t sizeY;
	double deltaT;
	vector<double> terrainHeight;
	vector<double> waterHeight;
	double* gTerrainHeight;
	double* gWaterHeight;

public:
	Model(size_t _sizeX, size_t _sizeY, double _deltaT, vector<double> _terrainHeight, vector<double> _waterHeight)
		: sizeX(_sizeX), sizeY(_sizeY), deltaT(_deltaT), terrainHeight(_terrainHeight), waterHeight(_waterHeight)
	{}

	~Model() {}

	void SetSizeX(const size_t sizeX);
	size_t GetSizeX() const;
	void SetSizeY(const size_t sizeY);
	size_t GetSizeY() const;
	void SetDeltaT(const double deltaT);
	double GetDeltaT() const;
	void SetTerrainHeight(const vector<double> terrainHeight);
	vector<double> GetTerrainHeight() const;
	void SetWaterHeight(const vector<double> waterHeight);
	vector<double> GetWaterHeight() const;

public:
	virtual void InitDevice() = 0;
	virtual void SendAllDataToDevice() = 0;
	virtual void PreparationForSimulaion() = 0;
	virtual void RunSimulation(const size_t step) = 0;
	virtual void GetResultFromDevice() = 0;
	virtual void FreeAllData() = 0;
};

