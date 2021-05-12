#ifndef ACTIVITY_H
#define ACTIVITY_H

inline __device__ void yankActivityTemplate(double &actFcnVal, double tauVal, double dWidth)
{
	// C2 piecewise polynomial supported on [-1, 1] with height 1

	//  a * ( x + 1)^3 * ( x - b)   if tau in [-1,     -1 + d]
	// -c * x^2 + 1                 if tau in [-1 + d,  1 - d]
	//  a * (-x + 1)^3 * (-x - b)   if tau in [ 1 - d,  1    ]
	//  0                           otherwise

	double a = 3.0 * (dWidth - 2.0) / (dWidth * dWidth * dWidth * (6.0 + dWidth * (-6.0 + dWidth)));
	double b = (6.0 + dWidth * (-15.0 + 8 * dWidth)) / (3.0 * (dWidth - 2.0));
	double c = 6.0 / (6.0 + dWidth * (-6.0 + dWidth));

	if ( tauVal <  -1.0 || tauVal >= 1.0 )
	{
		actFcnVal = 0.0;
		return;
	}

	if ( tauVal >= -1.0 && tauVal < -1.0 + dWidth )
	{
		actFcnVal = a * (tauVal + 1.0) * (tauVal + 1.0) * (tauVal + 1.0) * (tauVal - b);
		return;
	}

	if ( tauVal >= -1.0 + dWidth && tauVal < 1.0 - dWidth )
	{
		actFcnVal = -c * tauVal * tauVal + 1.0;
		return;
	}

	// tauVal >= 1 - d && tauVal < 1
	actFcnVal = a * (-tauVal + 1.0) * (-tauVal + 1.0) * (-tauVal + 1.0) * (-tauVal - b);
	return;
}

inline __device__ void yankActivityTemplate(double &actFcnVal, double &actDotVal,
                                            double tauVal, double dWidth)
{
	// C2 piecewise polynomial supported on [-1, 1] with height 1

	//  a * ( x + 1)^3 * ( x - b)   if tau in [-1,     -1 + d]
	// -c * x^2 + 1                 if tau in [-1 + d,  1 - d]
	//  a * (-x + 1)^3 * (-x - b)   if tau in [ 1 - d,  1    ]
	//  0                           otherwise

	double a = 3.0 * (dWidth - 2.0) / (dWidth * dWidth * dWidth * (6.0 + dWidth * (-6.0 + dWidth)));
	double b = (6.0 + dWidth * (-15.0 + 8 * dWidth)) / (3.0 * (dWidth - 2.0));
	double c = 6.0 / (6.0 + dWidth * (-6.0 + dWidth));

	if ( tauVal <  -1.0 || tauVal >= 1.0 )
	{
		actFcnVal = 0.0;
		actDotVal = 0.0;
		return;
	}

	if ( tauVal >= -1.0 && tauVal < -1.0 + dWidth )
	{
		actFcnVal = a * (tauVal + 1.0) * (tauVal + 1.0) * (tauVal + 1.0) * (tauVal - b);
		actDotVal = a * (tauVal + 1.0) * (tauVal + 1.0) * (3.0 * (tauVal - b) + (tauVal + 1.0));
		return;
	}

	if ( tauVal >= -1.0 + dWidth && tauVal < 1.0 - dWidth )
	{
		actFcnVal = -c * tauVal * tauVal + 1.0;
		actDotVal = -2.0 * c * tauVal;
		return;
	}

	// tauVal >= 1 - d && tauVal < 1
	actFcnVal =  a * (-tauVal + 1.0) * (-tauVal + 1.0) * (-tauVal + 1.0) * (-tauVal - b);
	actDotVal = -a * (-tauVal + 1.0) * (-tauVal + 1.0) * (3.0 * (-tauVal - b) + (-tauVal + 1.0));
	return;
}

inline __device__ void yankActivityFunction(double &actFcnVal, double tauVal,
                                            double tauMin, double tauMax, double actMax, double dWidth)
{
	double tauStdVal = -1.0 + 2.0 * (tauVal - tauMin) / (tauMax - tauMin);
	
	yankActivityTemplate(actFcnVal, tauStdVal, dWidth);
	actFcnVal *= actMax;

	return;
}

inline __device__ void yankActivityFunction(double &actFcnVal, double &actDotVal, double tauVal,
                                            double tauMin, double tauMax, double actMax, double dWidth)
{
	double tauStdVal = -1.0 + 2.0 * (tauVal - tauMin) / (tauMax - tauMin);

	yankActivityTemplate(actFcnVal, actDotVal, tauStdVal, dWidth);
	actFcnVal *= actMax;
	actDotVal *= actMax * 2.0 / (tauMax - tauMin);

	return;
}

// ---

inline __device__ void reacActivityTemplate(double &actFcnVal, double tauVal, double dWidth, double hatVal)
{
	// C2 piecewise polynomial supported on [-1, 1] with height 1

	//  b1 * ( x + 1)^3 * ( x - b2)         if tau in [-1,       -1 + d/2]
	//  c1 * x^3 + c2 * x^2 + c3 * x + c4   if tau in [-1 + d/2, -1 + d  ]
	// -a  * x^2 + 1                        if tau in [-1 + d,    1 - d  ]
	// -c1 * x^3 + c2 * x^2 - c3 * x + c4   if tau in [ 1 - d,    1 - d/2]
	//  b1 * (-x + 1)^3 * (-x - b2)         if tau in [ 1 - d/2,  1      ]
	//  0                                   otherwise

	double a  = hatVal;
	double b1 = 8.0 * (6.0 * (a - 1.0) + a * dWidth * (-6.0 + dWidth))
	            / (3.0 * dWidth * dWidth * dWidth * dWidth);
	double b2 =  -(144.0 * (a - 1.0) + dWidth * (24.0 * (5.0 - 11.0 * a) + dWidth * (156.0 * a - dWidth * 29.0 * a)))
	            / (24.0 * (6.0 * (a - 1.0) + a * dWidth * (-6.0 + dWidth)));
	double c1 = -(24.0 * (a - 1.0) + a * dWidth * (-12.0 + dWidth)) / (9.0 * dWidth * dWidth * dWidth);
	double c2 = -(24.0 * (a - 1.0) + dWidth * (12.0 * (2.0 - 3.0 * a) + a * dWidth * (13.0 + dWidth * 2.0)))
	            / (3.0 * dWidth * dWidth * dWidth);
	double c3 = -((dWidth - 1.0) * (dWidth - 1.0) * (24.0 * (a - 1.0) + a * dWidth * (-12.0 + dWidth)))
	            / (3.0 * dWidth * dWidth * dWidth);
	double c4 = -(24.0 * (a - 1.0)
	              + dWidth * (12.0 * (6.0 - 7.0 * a) 
	                          + dWidth * (-72.0 + 109.0 * a
	                                      + dWidth * (3.0 * (5.0 - 21.0 * a)
	                                                  + a * dWidth * (15.0 - dWidth)))))
	            / (9.0 * dWidth * dWidth * dWidth);

	if ( tauVal <  -1.0 || tauVal >= 1.0 )
	{
		actFcnVal = 0.0;
		return;
	}

	if ( tauVal >= -1.0 && tauVal < -1.0 + 0.5 * dWidth )
	{
		actFcnVal = b1 * (tauVal + 1.0) * (tauVal + 1.0) * (tauVal + 1.0) * (tauVal - b2);
		return;
	}

	if ( tauVal >= -1.0 + 0.5 * dWidth && tauVal < -1.0 + dWidth )
	{
		actFcnVal = c4 + tauVal * (c3 + tauVal * (c2 + tauVal * c1));
		return;
	}

	if ( tauVal >= -1.0 + dWidth && tauVal < 1.0 - dWidth )
	{
		actFcnVal = -a * tauVal * tauVal + 1.0;
		return;
	}

	if ( tauVal >= 1.0 - dWidth && tauVal < 1.0 - 0.5 * dWidth )
	{
		actFcnVal = c4 + tauVal * (-c3 + tauVal * (c2 - tauVal * c1));
		return;
	}

	// tauVal >= 1 - d/2 && tauVal < 1
	actFcnVal = b1 * (-tauVal + 1.0) * (-tauVal + 1.0) * (-tauVal + 1.0) * (-tauVal - b2);
	return;
}

inline __device__ void reacActivityTemplate(double &actFcnVal, double &actDotVal,
                                            double tauVal, double dWidth, double hatVal)
{
	// C2 piecewise polynomial supported on [-1, 1] with height 1

	//  b1 * ( x + 1)^3 * ( x - b2)         if tau in [-1,       -1 + d/2]
	//  c1 * x^3 + c2 * x^2 + c3 * x + c4   if tau in [-1 + d/2, -1 + d  ]
	// -a  * x^2 + 1                        if tau in [-1 + d,    1 - d  ]
	// -c1 * x^3 + c2 * x^2 - c3 * x + c4   if tau in [ 1 - d,    1 - d/2]
	//  b1 * (-x + 1)^3 * (-x - b2)         if tau in [ 1 - d/2,  1      ]
	//  0                                   otherwise

	double a  = hatVal;
	double b1 = 8.0 * (6.0 * (a - 1.0) + a * dWidth * (-6.0 + dWidth))
	            / (3.0 * dWidth * dWidth * dWidth * dWidth);
	double b2 =  -(144.0 * (a - 1.0) + dWidth * (24.0 * (5.0 - 11.0 * a) + dWidth * (156.0 * a - dWidth * 29.0 * a)))
	            / (24.0 * (6.0 * (a - 1.0) + a * dWidth * (-6.0 + dWidth)));
	double c1 = -(24.0 * (a - 1.0) + a * dWidth * (-12.0 + dWidth)) / (9.0 * dWidth * dWidth * dWidth);
	double c2 = -(24.0 * (a - 1.0) + dWidth * (12.0 * (2.0 - 3.0 * a) + dWidth * (13.0 * a + dWidth * 2.0 * a)))
	            / (3.0 * dWidth * dWidth * dWidth);
	double c3 = -((dWidth - 1.0) * (dWidth - 1.0) * (24.0 * (a - 1.0) + a * dWidth * (-12.0 + dWidth)))
	            / (3.0 * dWidth * dWidth * dWidth);
	double c4 = -(24.0 * (a - 1.0)
	              + dWidth * (12.0 * (6.0 - 7.0 * a) 
	                          + dWidth * (-72.0 + 109.0 * a
	                                      + dWidth * (3.0 * (5.0 - 21.0 * a)
	                                                  + dWidth * (15.0 - dWidth) * a))))
	            / (9.0 * dWidth * dWidth * dWidth);

	if ( tauVal <  -1.0 || tauVal >= 1.0 )
	{
		actFcnVal = 0.0;
		actDotVal = 0.0;
		return;
	}

	if ( tauVal >= -1.0 && tauVal < -1.0 + 0.5 * dWidth )
	{
		actFcnVal = b1 * (tauVal + 1.0) * (tauVal + 1.0) * (tauVal + 1.0) * (tauVal - b2);
		actDotVal = b1 * (tauVal + 1.0) * (tauVal + 1.0) * (3.0 * (tauVal - b2) + (tauVal + 1.0));
		return;
	}

	if ( tauVal >= -1.0 + 0.5 * dWidth && tauVal < -1.0 + dWidth )
	{
		actFcnVal = c4 + tauVal * (c3 + tauVal * (c2 + tauVal * c1));
		actDotVal = c3 + tauVal * (2.0 * c2 + tauVal * 3.0 * c1);
		return;
	}

	if ( tauVal >= -1.0 + dWidth && tauVal < 1.0 - dWidth )
	{
		actFcnVal = -a * tauVal * tauVal + 1.0;
		actDotVal = -2.0 * a * tauVal;
		return;
	}

	if ( tauVal >= 1.0 - dWidth && tauVal < 1.0 - 0.5 * dWidth )
	{
		actFcnVal =  c4 + tauVal * (-c3 + tauVal * (c2 - tauVal * c1));
		actDotVal = -c3 + tauVal * (2.0 * c2 - tauVal * 3.0 * c1);
		return;
	}

	// tauVal >= 1 - d/2 && tauVal < 1
	actFcnVal =  b1 * (-tauVal + 1.0) * (-tauVal + 1.0) * (-tauVal + 1.0) * (-tauVal - b2);
	actDotVal = -b1 * (-tauVal + 1.0) * (-tauVal + 1.0) * (3.0 * (-tauVal - b2) + (-tauVal + 1.0));
	return;
}

inline __device__ void reacActivityFunction(double &actFcnVal, double tauVal,
                                            double tauMin, double tauMax, double actMax,
                                            double dWidth, double hatVal)
{
	double tauStdVal = -1.0 + 2.0 * (tauVal - tauMin) / (tauMax - tauMin);
	
	reacActivityTemplate(actFcnVal, tauStdVal, dWidth, hatVal);
	actFcnVal *= actMax;

	return;
}

inline __device__ void reacActivityFunction(double &actFcnVal, double &actDotVal, double tauVal,
                                            double tauMin, double tauMax, double actMax,
                                            double dWidth, double hatVal)
{
	double tauStdVal = -1.0 + 2.0 * (tauVal - tauMin) / (tauMax - tauMin);

	reacActivityTemplate(actFcnVal, actDotVal, tauStdVal, dWidth, hatVal);
	actFcnVal *= actMax;
	actDotVal *= actMax * 2.0 / (tauMax - tauMin);

	return;
}

#endif
