#pragma once
#include <vector>
#include "Point.h"

class Dam
{
protected:
	std::vector<Point<double>> posList;
	double width;

public:
	Dam() : posList(std::vector<Point<double>>(0)), width(0.0) {}
	Dam(std::vector<Point<double>> _posList, double _width) : posList(_posList), width(_width) {}
	Dam(double _width) : posList(std::vector<Point<double>>(0)), width(_width) {}
	~Dam() {}

public:
	void SetPosList(std::vector<Point<double>> pl)
	{
		this->posList = pl;
	}
	std::vector<Point<double>> GetPosList() const
	{
		return this->posList;
	}
	void SetWidth(const double width)
	{
		this->width = width;
	}
	double GetWidth() const
	{
		return this->width;
	}

	void AddPoint(const Point<double> p)
	{
		this->posList.push_back(p);
	}
	void ClearPosList()
	{
		this->posList.clear();
	}

	size_t GetPointCount()
	{
		return this->posList.size();
	}
	Point<double> GetSelectedPoint(size_t index)
	{
		if (index >= this->posList.size() || index < 0)
			throw "Index Wrong";
		return this->posList[index];
	}
};