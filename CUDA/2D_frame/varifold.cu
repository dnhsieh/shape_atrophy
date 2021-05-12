// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 07/09/2020

#include "matvec.h"
#include "constants.h"

void dsum(double *, double *, double *, int);

__global__ void landmarksToVarifoldKernel(double *d_cenPosMat, double *d_uniDirMat, double *d_elmVolVec,
                                          double *d_lmkPosMat, int *d_elmVtxMat, int lmkNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		int q0Idx = d_elmVtxMat[         elmIdx];
		int q1Idx = d_elmVtxMat[elmNum + elmIdx];

		vector q0Vec, q1Vec;
		getVector(q0Vec, d_lmkPosMat, q0Idx, lmkNum);
		getVector(q1Vec, d_lmkPosMat, q1Idx, lmkNum);

		vector cenVec, dirVec;
		 vectorAverage(cenVec, q0Vec, q1Vec);
		vectorSubtract(dirVec, q1Vec, q0Vec);

		double elmVol = eucnorm(dirVec);
		dirVec.x /= elmVol;
		dirVec.y /= elmVol;
		
		setVector(d_cenPosMat, cenVec, elmIdx, elmNum);
		setVector(d_uniDirMat, dirVec, elmIdx, elmNum);
		d_elmVolVec[elmIdx] = elmVol;
	}

	return;
}

__device__ void geometricFunction(double &knlVal, vector c1Vec, vector c2Vec,
                                  char knlType, double knlWidth)
{
	if ( knlType == 'G' )   // gaussian
	{
		double dstSqu = eucdistSqu(c1Vec, c2Vec);
		knlVal = exp(-dstSqu / (knlWidth * knlWidth));

		return;
	}

	if ( knlType == 'C' )   // cauchy
	{
		double dstSqu = eucdistSqu(c1Vec, c2Vec);
		knlVal = 1.0 / (1.0 + dstSqu / (knlWidth * knlWidth));

		return;
	}

	return;
}

__device__ void geometricFunction(double &knlVal, vector &d1KVec, vector c1Vec, vector c2Vec, 
                                  char knlType, double knlWidth)
{
	if ( knlType == 'G' )   // gaussian
	{
		double dstSqu = eucdistSqu(c1Vec, c2Vec);
		knlVal = exp(-dstSqu / (knlWidth * knlWidth));

		double d1KVal = -2.0 * knlVal / (knlWidth * knlWidth);
		d1KVec.x = d1KVal * (c1Vec.x - c2Vec.x);
		d1KVec.y = d1KVal * (c1Vec.y - c2Vec.y);

		return;
	}

	if ( knlType == 'C' )   // cauchy
	{
		double dstSqu = eucdistSqu(c1Vec, c2Vec);
		knlVal = 1.0 / (1.0 + dstSqu / (knlWidth * knlWidth));

		double d1KVal = -2.0 * knlVal * knlVal / (knlWidth * knlWidth);
		d1KVec.x = d1KVal * (c1Vec.x - c2Vec.x);
		d1KVec.y = d1KVal * (c1Vec.y - c2Vec.y);

		return;
	}

	return;
}

__device__ void grassmanFunction(double &knlVal, vector v1Vec, vector v2Vec,
                                 char knlType, double knlWidth)
{
	if ( knlType == 'B' )   // binet
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = angVal * angVal;

		return;
	}

	if ( knlType == 'L' )   // linear
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = angVal;

		return;
	}

	if ( knlType == 'O' )   // gaussian oriented
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = exp(2.0 * (angVal - 1.0) / (knlWidth * knlWidth));

		return;
	}

	if ( knlType == 'U' )   // gaussian unoriented
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = exp(2.0 * (angVal * angVal - 1.0) / (knlWidth * knlWidth));

		return;
	}

	return;
}

__device__ void grassmanFunction(double &knlVal, vector &d1KVec, vector v1Vec, vector v2Vec,
                                 char knlType, double knlWidth, double v1Vol)
{
	if ( knlType == 'B' )   // binet
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = angVal * angVal;

		double d1KVal = 2.0 * angVal;
		d1KVec.x = d1KVal / v1Vol * (-angVal * v1Vec.x + v2Vec.x);
		d1KVec.y = d1KVal / v1Vol * (-angVal * v1Vec.y + v2Vec.y);

		return;
	}

	if ( knlType == 'L' )   // linear
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = angVal;

		d1KVec.x = 1.0 / v1Vol * (-angVal * v1Vec.x + v2Vec.x);
		d1KVec.y = 1.0 / v1Vol * (-angVal * v1Vec.y + v2Vec.y);

		return;
	}

	if ( knlType == 'O' )   // gaussian oriented
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = exp(2.0 * (angVal - 1.0) / (knlWidth * knlWidth));

		double d1KVal = 2.0 * knlVal / (knlWidth * knlWidth);
		d1KVec.x = d1KVal / v1Vol * (-angVal * v1Vec.x + v2Vec.x);
		d1KVec.y = d1KVal / v1Vol * (-angVal * v1Vec.y + v2Vec.y);

		return;
	}

	if ( knlType == 'U' )   // gaussian unoriented
	{
		double angVal = dotProduct(v1Vec, v2Vec);
		knlVal = exp(2.0 * (angVal * angVal - 1.0) / (knlWidth * knlWidth));

		double d1KVal = 4.0 * angVal * knlVal / (knlWidth * knlWidth);
		d1KVec.x = d1KVal / v1Vol * (-angVal * v1Vec.x + v2Vec.x);
		d1KVec.y = d1KVal / v1Vol * (-angVal * v1Vec.y + v2Vec.y);

		return;
	}

	return;
}

__global__ void vfd_DD_DT_Kernel(double *d_vfdVec, 
                                 double *d_dfmCenPosMat, double *d_dfmUniDirMat, double *d_dfmElmVolVec, 
                                 double *d_tgtCenPosMat, double *d_tgtUniDirMat, double *d_tgtElmVolVec, 
                                 char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth, 
                                 int dfmElmNum, int tgtElmNum)
{
	int dfmElmiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( dfmElmiIdx < dfmElmNum )
	{
		double vfdVal = 0.0;

		vector dfmCeniVec, dfmDiriVec;
		getVector(dfmCeniVec, d_dfmCenPosMat, dfmElmiIdx, dfmElmNum);
		getVector(dfmDiriVec, d_dfmUniDirMat, dfmElmiIdx, dfmElmNum);

		double dfmElmiVol = d_dfmElmVolVec[dfmElmiIdx];

		for ( int dfmElmjIdx = 0; dfmElmjIdx < dfmElmNum; ++dfmElmjIdx )
		{
			vector dfmCenjVec, dfmDirjVec;
			getVector(dfmCenjVec, d_dfmCenPosMat, dfmElmjIdx, dfmElmNum);
			getVector(dfmDirjVec, d_dfmUniDirMat, dfmElmjIdx, dfmElmNum);

			double dfmElmjVol = d_dfmElmVolVec[dfmElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, dfmCeniVec, dfmCenjVec, cenKnlType, cenKnlWidth);
			 grassmanFunction(dirKnlVal, dfmDiriVec, dfmDirjVec, dirKnlType, dirKnlWidth);

			vfdVal += cenKnlVal * dirKnlVal * dfmElmiVol * dfmElmjVol;
		}

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			vector tgtCenjVec, tgtDirjVec;
			getVector(tgtCenjVec, d_tgtCenPosMat, tgtElmjIdx, tgtElmNum);
			getVector(tgtDirjVec, d_tgtUniDirMat, tgtElmjIdx, tgtElmNum);

			double tgtElmjVol = d_tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, dfmCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth);			
			 grassmanFunction(dirKnlVal, dfmDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth);

			vfdVal -= 2.0 * cenKnlVal * dirKnlVal * dfmElmiVol * tgtElmjVol;
		}

		d_vfdVec[dfmElmiIdx] = vfdVal;
	}

	return;
}

__global__ void vfd_TT_Kernel(double *d_vfdVec, double *d_tgtCenPosMat, double *d_tgtUniDirMat, 
                              double *d_tgtElmVolVec, char cenKnlType, double cenKnlWidth,
                              char dirKnlType, double dirKnlWidth, int tgtElmNum)
{
	int tgtElmiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tgtElmiIdx < tgtElmNum )
	{
		double vfdVal = 0.0;
	
		vector tgtCeniVec, tgtDiriVec;
		getVector(tgtCeniVec, d_tgtCenPosMat, tgtElmiIdx, tgtElmNum);
		getVector(tgtDiriVec, d_tgtUniDirMat, tgtElmiIdx, tgtElmNum);

		double tgtElmiVol = d_tgtElmVolVec[tgtElmiIdx];

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			vector tgtCenjVec, tgtDirjVec;
			getVector(tgtCenjVec, d_tgtCenPosMat, tgtElmjIdx, tgtElmNum);
			getVector(tgtDirjVec, d_tgtUniDirMat, tgtElmjIdx, tgtElmNum);

			double tgtElmjVol = d_tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, tgtCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth);			
			 grassmanFunction(dirKnlVal, tgtDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth);

			vfdVal += cenKnlVal * dirKnlVal * tgtElmiVol * tgtElmjVol;
		}

		d_vfdVec[tgtElmiIdx] = vfdVal;
	}

	return;
}

__global__ void vfd_TT_TD_Kernel(double *d_vfdVec, 
                                 double *d_dfmCenPosMat, double *d_dfmUniDirMat, double *d_dfmElmVolVec,
                                 double *d_tgtCenPosMat, double *d_tgtUniDirMat, double *d_tgtElmVolVec, 
                                 char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth, 
                                 int dfmElmNum, int tgtElmNum)
{
	int tgtElmiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tgtElmiIdx < tgtElmNum )
	{
		double vfdVal = 0.0;

		vector tgtCeniVec, tgtDiriVec;
		getVector(tgtCeniVec, d_tgtCenPosMat, tgtElmiIdx, tgtElmNum);
		getVector(tgtDiriVec, d_tgtUniDirMat, tgtElmiIdx, tgtElmNum);

		double tgtElmiVol = d_tgtElmVolVec[tgtElmiIdx];

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			vector tgtCenjVec, tgtDirjVec;
			getVector(tgtCenjVec, d_tgtCenPosMat, tgtElmjIdx, tgtElmNum);
			getVector(tgtDirjVec, d_tgtUniDirMat, tgtElmjIdx, tgtElmNum);

			double tgtElmjVol = d_tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, tgtCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth);			
			 grassmanFunction(dirKnlVal, tgtDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth);

			vfdVal += cenKnlVal * dirKnlVal * tgtElmiVol * tgtElmjVol;
		}

		for ( int dfmElmjIdx = 0; dfmElmjIdx < dfmElmNum; ++dfmElmjIdx )
		{
			vector dfmCenjVec, dfmDirjVec;
			getVector(dfmCenjVec, d_dfmCenPosMat, dfmElmjIdx, dfmElmNum);
			getVector(dfmDirjVec, d_dfmUniDirMat, dfmElmjIdx, dfmElmNum);

			double dfmElmjVol = d_dfmElmVolVec[dfmElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, tgtCeniVec, dfmCenjVec, cenKnlType, cenKnlWidth);
			 grassmanFunction(dirKnlVal, tgtDiriVec, dfmDirjVec, dirKnlType, dirKnlWidth);

			vfdVal -= 2.0 * cenKnlVal * dirKnlVal * tgtElmiVol * dfmElmjVol;
		}

		d_vfdVec[tgtElmiIdx] = vfdVal;
	}

	return;
}

__global__ void vfd_DD_Kernel(double *d_vfdVec, double *d_dfmCenPosMat, double *d_dfmUniDirMat,
                              double *d_dfmElmVolVec, char cenKnlType, double cenKnlWidth, 
                              char dirKnlType, double dirKnlWidth, int dfmElmNum)
{
	int dfmElmiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( dfmElmiIdx < dfmElmNum )
	{
		double vfdVal = 0.0;

		vector dfmCeniVec, dfmDiriVec;
		getVector(dfmCeniVec, d_dfmCenPosMat, dfmElmiIdx, dfmElmNum);
		getVector(dfmDiriVec, d_dfmUniDirMat, dfmElmiIdx, dfmElmNum);

		double dfmElmiVol = d_dfmElmVolVec[dfmElmiIdx];

		for ( int dfmElmjIdx = 0; dfmElmjIdx < dfmElmNum; ++dfmElmjIdx )
		{
			vector dfmCenjVec, dfmDirjVec;
			getVector(dfmCenjVec, d_dfmCenPosMat, dfmElmjIdx, dfmElmNum);
			getVector(dfmDirjVec, d_dfmUniDirMat, dfmElmjIdx, dfmElmNum);

			double dfmElmjVol = d_dfmElmVolVec[dfmElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, dfmCeniVec, dfmCenjVec, cenKnlType, cenKnlWidth);
			 grassmanFunction(dirKnlVal, dfmDiriVec, dfmDirjVec, dirKnlType, dirKnlWidth);

			vfdVal += cenKnlVal * dirKnlVal * dfmElmiVol * dfmElmjVol;
		}

		d_vfdVec[dfmElmiIdx] = vfdVal;
	}

	return;
}

__global__ void dqVfd_DD_DT_Kernel(double *d_vfdVec, double *d_dcVfdMat, double *d_dtVfdMat,
                                   double *d_dfmCenPosMat, double *d_dfmUniDirMat, double *d_dfmElmVolVec, 
                                   double *d_tgtCenPosMat, double *d_tgtUniDirMat, double *d_tgtElmVolVec, 
                                   char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth, 
                                   int dfmElmNum, int tgtElmNum)
{
	int dfmElmiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( dfmElmiIdx < dfmElmNum )
	{
		double vfdVal = 0.0;
		vector dciVfdVec = {0.0, 0.0};
		vector dtiVfdVec = {0.0, 0.0};

		vector dfmCeniVec, dfmDiriVec;
		getVector(dfmCeniVec, d_dfmCenPosMat, dfmElmiIdx, dfmElmNum);
		getVector(dfmDiriVec, d_dfmUniDirMat, dfmElmiIdx, dfmElmNum);

		double dfmElmiVol = d_dfmElmVolVec[dfmElmiIdx];

		for ( int dfmElmjIdx = 0; dfmElmjIdx < dfmElmNum; ++dfmElmjIdx )
		{
			vector dfmCenjVec, dfmDirjVec;
			getVector(dfmCenjVec, d_dfmCenPosMat, dfmElmjIdx, dfmElmNum);
			getVector(dfmDirjVec, d_dfmUniDirMat, dfmElmjIdx, dfmElmNum);

			double dfmElmjVol = d_dfmElmVolVec[dfmElmjIdx];

			double cenKnlVal, dirKnlVal;
			vector dciKnlVec, dtiKnlVec;
			geometricFunction(cenKnlVal, dciKnlVec, dfmCeniVec, dfmCenjVec, cenKnlType, cenKnlWidth            );
			 grassmanFunction(dirKnlVal, dtiKnlVec, dfmDiriVec, dfmDirjVec, dirKnlType, dirKnlWidth, dfmElmiVol);

			vfdVal += cenKnlVal * dirKnlVal * dfmElmiVol * dfmElmjVol;

			dciVfdVec.x += 2.0 * dciKnlVec.x * dirKnlVal * dfmElmiVol * dfmElmjVol;
			dciVfdVec.y += 2.0 * dciKnlVec.y * dirKnlVal * dfmElmiVol * dfmElmjVol;

			dtiVfdVec.x += 2.0 * cenKnlVal * (  dtiKnlVec.x * dfmElmiVol
			                                  + dirKnlVal   * dfmDiriVec.x ) * dfmElmjVol;

			dtiVfdVec.y += 2.0 * cenKnlVal * (  dtiKnlVec.y * dfmElmiVol
			                                  + dirKnlVal   * dfmDiriVec.y ) * dfmElmjVol;
		}

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			vector tgtCenjVec, tgtDirjVec;
			getVector(tgtCenjVec, d_tgtCenPosMat, tgtElmjIdx, tgtElmNum);
			getVector(tgtDirjVec, d_tgtUniDirMat, tgtElmjIdx, tgtElmNum);

			double tgtElmjVol = d_tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			vector dciKnlVec, dtiKnlVec;
			geometricFunction(cenKnlVal, dciKnlVec, dfmCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth            );			
			 grassmanFunction(dirKnlVal, dtiKnlVec, dfmDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth, dfmElmiVol);

			vfdVal -= 2.0 * cenKnlVal * dirKnlVal * dfmElmiVol * tgtElmjVol;

			dciVfdVec.x -= 2.0 * dciKnlVec.x * dirKnlVal * dfmElmiVol * tgtElmjVol;
			dciVfdVec.y -= 2.0 * dciKnlVec.y * dirKnlVal * dfmElmiVol * tgtElmjVol;

			dtiVfdVec.x -= 2.0 * cenKnlVal * (  dtiKnlVec.x * dfmElmiVol 
			                                  + dirKnlVal   * dfmDiriVec.x ) * tgtElmjVol;

			dtiVfdVec.y -= 2.0 * cenKnlVal * (  dtiKnlVec.y * dfmElmiVol
			                                  + dirKnlVal   * dfmDiriVec.y ) * tgtElmjVol;
		}

		d_vfdVec[dfmElmiIdx] = vfdVal;
		setVector(d_dcVfdMat, dciVfdVec, dfmElmiIdx, dfmElmNum);
		setVector(d_dtVfdMat, dtiVfdVec, dfmElmiIdx, dfmElmNum);
	}

	return;
}

__global__ void dqVfd_TT_Kernel(double *d_vfdVec, double *d_tgtCenPosMat, double *d_tgtUniDirMat, 
                                double *d_tgtElmVolVec, char cenKnlType, double cenKnlWidth,
                                char dirKnlType, double dirKnlWidth, int tgtElmNum)
{
	int tgtElmiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tgtElmiIdx < tgtElmNum )
	{
		double vfdVal = 0.0;
	
		vector tgtCeniVec, tgtDiriVec;
		getVector(tgtCeniVec, d_tgtCenPosMat, tgtElmiIdx, tgtElmNum);
		getVector(tgtDiriVec, d_tgtUniDirMat, tgtElmiIdx, tgtElmNum);

		double tgtElmiVol = d_tgtElmVolVec[tgtElmiIdx];

		for ( int tgtElmjIdx = 0; tgtElmjIdx < tgtElmNum; ++tgtElmjIdx )
		{
			vector tgtCenjVec, tgtDirjVec;
			getVector(tgtCenjVec, d_tgtCenPosMat, tgtElmjIdx, tgtElmNum);
			getVector(tgtDirjVec, d_tgtUniDirMat, tgtElmjIdx, tgtElmNum);

			double tgtElmjVol = d_tgtElmVolVec[tgtElmjIdx];

			double cenKnlVal, dirKnlVal;
			geometricFunction(cenKnlVal, tgtCeniVec, tgtCenjVec, cenKnlType, cenKnlWidth);	
			 grassmanFunction(dirKnlVal, tgtDiriVec, tgtDirjVec, dirKnlType, dirKnlWidth);

			vfdVal += cenKnlVal * dirKnlVal * tgtElmiVol * tgtElmjVol;
		}

		d_vfdVec[tgtElmiIdx] = vfdVal;
	}

	return;
}

__global__ void dqVfdGatherKernel(double *d_dqVfdMat, double *d_dcVfdMat, double *d_dtVfdMat,
                                  int *d_dfmElmIfoMat, int dfmElmNum, int dfmLmkNum)
{
	int dfmLmkIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( dfmLmkIdx < dfmLmkNum )
	{
		vector dqVfdVec = {0.0, 0.0};

		int adjNum = d_dfmElmIfoMat[dfmLmkIdx];
		for ( int adjIdx = 0; adjIdx < adjNum; ++adjIdx )
		{
			int elmIdx = d_dfmElmIfoMat[(1 + 2 * adjIdx    ) * dfmLmkNum + dfmLmkIdx];
			int sgnInt = d_dfmElmIfoMat[(1 + 2 * adjIdx + 1) * dfmLmkNum + dfmLmkIdx];

			vector dcVfdVec, dtVfdVec;
			getVector(dcVfdVec, d_dcVfdMat, elmIdx, dfmElmNum);
			getVector(dtVfdVec, d_dtVfdMat, elmIdx, dfmElmNum);

			dqVfdVec.x += 0.5 * dcVfdVec.x + sgnInt * dtVfdVec.x;
			dqVfdVec.y += 0.5 * dcVfdVec.y + sgnInt * dtVfdVec.y;
		}

		setVector(d_dqVfdMat, dqVfdVec, dfmLmkIdx, dfmLmkNum);
	}

	return;
}

void varifold(double *h_vfdPtr, double *d_dfmLmkPosMat, int *d_dfmElmVtxMat,
              double *d_tgtCenPosMat, double *d_tgtUniDirMat, double *d_tgtElmVolVec,
              char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth,
              double *d_dfmCenPosMat, double *d_dfmUniDirMat, double *d_dfmElmVolVec,
              double *d_vfdVec, double *d_sumBufVec,
              int dfmLmkNum, int dfmElmNum, int tgtElmNum)
{
	int blkNum = (dfmElmNum - 1) / BLKDIM + 1;
	landmarksToVarifoldKernel <<<blkNum, BLKDIM>>> (d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec, 
	                                                d_dfmLmkPosMat, d_dfmElmVtxMat, dfmLmkNum, dfmElmNum);
	
	if ( dfmElmNum >= tgtElmNum )
	{
		blkNum = (dfmElmNum - 1) / BLKDIM + 1;
		vfd_DD_DT_Kernel <<<blkNum, BLKDIM>>> (d_vfdVec, 
		                                       d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec,
		                                       d_tgtCenPosMat, d_tgtUniDirMat, d_tgtElmVolVec,
		                                       cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth,
		                                       dfmElmNum, tgtElmNum);

		blkNum = (tgtElmNum - 1) / BLKDIM + 1;
		vfd_TT_Kernel <<<blkNum, BLKDIM>>> (d_vfdVec + dfmElmNum, 
		                                    d_tgtCenPosMat, d_tgtUniDirMat, d_tgtElmVolVec,
		                                    cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth, tgtElmNum);
	}
	else
	{
		blkNum = (tgtElmNum - 1) / BLKDIM + 1;
		vfd_TT_TD_Kernel <<<blkNum, BLKDIM>>> (d_vfdVec, 
		                                       d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec,
		                                       d_tgtCenPosMat, d_tgtUniDirMat, d_tgtElmVolVec,
		                                       cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth,
		                                       dfmElmNum, tgtElmNum);

		blkNum = (dfmElmNum - 1) / BLKDIM + 1;
		vfd_DD_Kernel <<<blkNum, BLKDIM>>> (d_vfdVec + tgtElmNum, 
		                                    d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec,
		                                    cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth, dfmElmNum);
	}

	dsum(h_vfdPtr, d_vfdVec, d_sumBufVec, dfmElmNum + tgtElmNum);

	return;
}

void varifold(double *h_vfdPtr, double *d_dqVfdMat,
              double *d_dfmLmkPosMat, int *d_dfmElmVtxMat, int *d_dfmElmIfoMat,
              double *d_tgtCenPosMat, double *d_tgtUniDirMat, double *d_tgtElmVolVec,
              char cenKnlType, double cenKnlWidth, char dirKnlType, double dirKnlWidth,
              double *d_dfmCenPosMat, double *d_dfmUniDirMat, double *d_dfmElmVolVec,
              double *d_vfdVec, double *d_sumBufVec, double *d_dcVfdMat, double *d_dtVfdMat,
              int dfmLmkNum, int dfmElmNum, int tgtElmNum)
{
	int blkNum = (dfmElmNum - 1) / BLKDIM + 1;
	landmarksToVarifoldKernel <<<blkNum, BLKDIM>>> (d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec, 
	                                                d_dfmLmkPosMat, d_dfmElmVtxMat, dfmLmkNum, dfmElmNum);
	
	blkNum = (dfmElmNum - 1) / BLKDIM + 1;
	dqVfd_DD_DT_Kernel <<<blkNum, BLKDIM>>> (d_vfdVec, d_dcVfdMat, d_dtVfdMat, 
	                                         d_dfmCenPosMat, d_dfmUniDirMat, d_dfmElmVolVec,
	                                         d_tgtCenPosMat, d_tgtUniDirMat, d_tgtElmVolVec,
	                                         cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth,
	                                         dfmElmNum, tgtElmNum);

	blkNum = (tgtElmNum - 1) / BLKDIM + 1;
	dqVfd_TT_Kernel <<<blkNum, BLKDIM>>> (d_vfdVec + dfmElmNum,
	                                      d_tgtCenPosMat, d_tgtUniDirMat, d_tgtElmVolVec,
		                                   cenKnlType, cenKnlWidth, dirKnlType, dirKnlWidth, tgtElmNum);

	dsum(h_vfdPtr, d_vfdVec, d_sumBufVec, dfmElmNum + tgtElmNum);

	blkNum = (dfmLmkNum - 1) / BLKDIM + 1;
	dqVfdGatherKernel <<<blkNum, BLKDIM>>> (d_dqVfdMat, d_dcVfdMat, d_dtVfdMat,
	                                        d_dfmElmIfoMat, dfmElmNum, dfmLmkNum);

	return;
}
