#ifndef POLYBESSELK_H
#define POLYBESSELK_H

#include <cmath>
#include <cfloat>
#include "besselk.h"

inline __device__ double polyval(const int plyDeg, double *plyVec, double xVal)
{
	double plyVal = plyVec[plyDeg];
	for ( int powIdx = plyDeg - 1; powIdx >= 0; --powIdx )
		plyVal = plyVal * xVal + plyVec[powIdx];

	return plyVal;
}

inline __device__ void p0Fcn(double &f0Val, double xVal)
{
	if ( xVal < DBL_EPSILON || xVal > xMax )
	{
		f0Val = 0.0;
		return;
	}

	if ( xVal <= 1.0 )
	{
		double xSqu = xVal * xVal;
		double logx = log(xVal);

		double P01Val = polyval(P01Deg, c_P01Vec, xSqu);	
		double Q01Val = polyval(Q01Deg, c_Q01Vec, xSqu);	
		double PQ01   = P01Val / Q01Val;

		double P02Val = polyval(P02Deg, c_P02Vec, xSqu);	
		double Q02Val = polyval(Q02Deg, c_Q02Vec, xSqu);	
		double PQ02   = P02Val / Q02Val;

		f0Val = xSqu * (PQ01 - logx * (xSqu * PQ02 + 1.0)); 

		return;
	}

	// 1 < x <= xMax

	double xInv = 1.0 / xVal;

	double P03Val = polyval(P03Deg, c_P03Vec, xInv);	
	double Q03Val = polyval(Q03Deg, c_Q03Vec, xInv);	
	double PQ03   = P03Val / Q03Val;

	f0Val = xVal * sqrt(xVal) * exp(-xVal) * PQ03;

	return;
}

inline __device__ void p1Fcn(double &f1Val, double xVal)
{
	if ( xVal < DBL_EPSILON )
	{
		f1Val = 1.0;
		return;
	}

	if ( xVal > xMax )
	{
		f1Val = 0.0;
		return;
	}

	if ( xVal <= 1.0 )
	{
		double xSqu = xVal * xVal;
		double logx = log(xVal);

		double P11Val = polyval(P11Deg, c_P11Vec, xSqu);	
		double Q11Val = polyval(Q11Deg, c_Q11Vec, xSqu);	
		double PQ11   = P11Val / Q11Val;

		double P12Val = polyval(P12Deg, c_P12Vec, xSqu);	
		double Q12Val = polyval(Q12Deg, c_Q12Vec, xSqu);	
		double PQ12   = P12Val / Q12Val;

		f1Val = PQ11 + xSqu * logx * PQ12;

		return;
	}

	// 1 < x <= xMax

	double xInv = 1.0 / xVal;

	double P13Val = polyval(P13Deg, c_P13Vec, xInv);	
	double Q13Val = polyval(Q13Deg, c_Q13Vec, xInv);	
	double PQ13   = P13Val / Q13Val;

	f1Val = sqrt(xVal) * exp(-xVal) * PQ13;

	return;
}

inline __device__ void p0Fcn(double &f0Val, double &g0Val, double xVal)
{
	if ( xVal < DBL_EPSILON || xVal > xMax )
	{
		f0Val = 0.0;
		g0Val = 0.0;
		return;
	}

	if ( xVal <= 1.0 )
	{
		double xSqu = xVal * xVal;
		double logx = log(xVal);

		double P01Val = polyval(P01Deg, c_P01Vec, xSqu);	
		double Q01Val = polyval(Q01Deg, c_Q01Vec, xSqu);	
		double PQ01   = P01Val / Q01Val;

		double P02Val = polyval(P02Deg, c_P02Vec, xSqu);	
		double Q02Val = polyval(Q02Deg, c_Q02Vec, xSqu);	
		double PQ02   = P02Val / Q02Val;

		double dP01Val = polyval(dP01Deg, c_dP01Vec, xSqu);
		double dQ01Val = polyval(dQ01Deg, c_dQ01Vec, xSqu);
		double dPQ01   = (dP01Val * Q01Val - P01Val * dQ01Val) / (Q01Val * Q01Val);

		double dP02Val = polyval(dP02Deg, c_dP02Vec, xSqu);
		double dQ02Val = polyval(dQ02Deg, c_dQ02Vec, xSqu);
		double dPQ02   = (dP02Val * Q02Val - P02Val * dQ02Val) / (Q02Val * Q02Val);

		f0Val = xSqu * (PQ01 - logx * (xSqu * PQ02 + 1.0)); 
		g0Val =  2.0 * PQ01 - (2.0 * logx + 1.0)
		       + xSqu * (2.0 * dPQ01 - PQ02 - 2.0 * logx * (xSqu * dPQ02 + 2.0 * PQ02));

		return;
	}

	// 1 < x <= xMax

	double xInv = 1.0 / xVal;

	double P03Val = polyval(P03Deg, c_P03Vec, xInv);	
	double Q03Val = polyval(Q03Deg, c_Q03Vec, xInv);	
	double PQ03   = P03Val / Q03Val;

	double dP03Val = polyval(dP03Deg, c_dP03Vec, xInv);
	double dQ03Val = polyval(dQ03Deg, c_dQ03Vec, xInv);
	double dPQ03   = (dP03Val * Q03Val - P03Val * dQ03Val) / (Q03Val * Q03Val);

	f0Val = xVal * sqrt(xVal) * exp(-xVal) * PQ03;
	g0Val = exp(-xVal) / sqrt(xVal) * (-xInv * dPQ03 + (1.5 - xVal) * PQ03);

	return;
}

inline __device__ void p1Fcn(double &f1Val, double &g1Val, double xVal)
{
	if ( xVal < DBL_EPSILON )
	{
		f1Val = 1.0;
		g1Val = 0.0;
		return;
	}

	if ( xVal > xMax )
	{
		f1Val = 0.0;
		g1Val = 0.0;
		return;
	}

	if ( xVal <= 1.0 )
	{
		double xSqu = xVal * xVal;
		double logx = log(xVal);

		double P11Val = polyval(P11Deg, c_P11Vec, xSqu);	
		double Q11Val = polyval(Q11Deg, c_Q11Vec, xSqu);	
		double PQ11   = P11Val / Q11Val;

		double P12Val = polyval(P12Deg, c_P12Vec, xSqu);	
		double Q12Val = polyval(Q12Deg, c_Q12Vec, xSqu);	
		double PQ12   = P12Val / Q12Val;

		double dP11Val = polyval(dP11Deg, c_dP11Vec, xSqu);
		double dQ11Val = polyval(dQ11Deg, c_dQ11Vec, xSqu);
		double dPQ11   = (dP11Val * Q11Val - P11Val * dQ11Val) / (Q11Val * Q11Val);

		double dP12Val = polyval(dP12Deg, c_dP12Vec, xSqu);
		double dQ12Val = polyval(dQ12Deg, c_dQ12Vec, xSqu);
		double dPQ12   = (dP12Val * Q12Val - P12Val * dQ12Val) / (Q12Val * Q12Val);

		f1Val = PQ11 + xSqu * logx * PQ12;
		g1Val = 2.0 * dPQ11 + PQ12 + 2.0 * logx * (xSqu * dPQ12 + PQ12);

		return;
	}

	// 1 < x <= xMax

	double xInv = 1.0 / xVal;

	double P13Val = polyval(P13Deg, c_P13Vec, xInv);	
	double Q13Val = polyval(Q13Deg, c_Q13Vec, xInv);	
	double PQ13   = P13Val / Q13Val;

	double dP13Val = polyval(dP13Deg, c_dP13Vec, xInv);
	double dQ13Val = polyval(dQ13Deg, c_dQ13Vec, xInv);
	double dPQ13   = (dP13Val * Q13Val - P13Val * dQ13Val) / (Q13Val * Q13Val);

	f1Val = sqrt(xVal) * exp(-xVal) * PQ13;
	g1Val = exp(-xVal) / sqrt(xVal) * (-xInv * xInv * dPQ13 + (0.5 * xInv - 1.0) * PQ13);

	return;
}

#endif
