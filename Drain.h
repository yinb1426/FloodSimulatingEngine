#pragma once
#include "Point.h"
class Drain
{
protected:
	Point<double> pos;
	double rate;

public:
	Drain() : pos(Point<double>(0.0, 0.0)), rate(0) {}
	Drain(double _x, double _y, double _rate) : pos(Point<double>(_x, _y)), rate(_rate) {}
	~Drain() {}

public:
	void SetPos(const Point<double> p)
	{
		this->pos = p;
	}
	Point<double> GetPos() const
	{
		return this->pos;
	}
	void SetRate(const double rate)
	{
		this->rate = rate;
	}
	double GetRate() const
	{
		return this->rate;
	}
};