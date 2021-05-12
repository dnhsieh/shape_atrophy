#include "matvec.h"
#include "utility.h"
#include "constants.h"

void computeStrainTensor(double *, double *, double *, int *, int, int);

__device__ void computePreMat(matrix &preMat, vector tanVec, vector tsvVec,
                              double muTanMod, double muTsvMod, double muAngMod, matrix epvMat)
{
	vector evSVec;
	matVecMul(evSVec, epvMat, tsvVec);

	double TvTVal = vecMatVecMul(tanVec, epvMat, tanVec);
	double SvSVal = dotProduct(tsvVec, evSVec);
	double TvSVal = dotProduct(tanVec, evSVec);

	// ---

	preMat.x.x  = muTanMod * (tanVec.x * tanVec.x) * TvTVal;
	preMat.x.y  = muTanMod * (tanVec.x * tanVec.y) * TvTVal;
	preMat.y.x  = muTanMod * (tanVec.y * tanVec.x) * TvTVal;
	preMat.y.y  = muTanMod * (tanVec.y * tanVec.y) * TvTVal;

	// ---

	preMat.x.x += muTsvMod * (tsvVec.x * tsvVec.x) * SvSVal;
	preMat.x.y += muTsvMod * (tsvVec.x * tsvVec.y) * SvSVal;
	preMat.y.x += muTsvMod * (tsvVec.y * tsvVec.x) * SvSVal;
	preMat.y.y += muTsvMod * (tsvVec.y * tsvVec.y) * SvSVal;

	// ---

	preMat.x.x += muAngMod * (                2.0 * tanVec.x * tsvVec.x) * TvSVal;
	preMat.x.y += muAngMod * (tanVec.x * tsvVec.y + tsvVec.x * tanVec.y) * TvSVal;
	preMat.y.x += muAngMod * (tanVec.y * tsvVec.x + tsvVec.y * tanVec.x) * TvSVal;
	preMat.y.y += muAngMod * (                2.0 * tanVec.y * tsvVec.y) * TvSVal;

	return;
}

__global__ void frmElasticComputeKernel(double *d_elYElmMat, double *d_lmkEdgMat,
                                        double *d_tanMat, double *d_tsvMat,
                                        double muTanMod, double muTsvMod, double muAngMod,
                                        double *d_epvMat, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector q10Vec, q20Vec;
		getEdge(q10Vec, q20Vec, d_lmkEdgMat, elmIdx, elmNum);

		double volVal = computeVolume(q10Vec, q20Vec);

		// Q = [q1 - q0, q2 - q0]
		matrix QInvMat;
		matInv(QInvMat, q10Vec, q20Vec);

		matrix epvMat;
		getMatrix(epvMat, d_epvMat, elmIdx, elmNum);

		vector tanVec, tsvVec;
		getVector(tanVec, d_tanMat, elmIdx, elmNum);
		getVector(tsvVec, d_tsvMat, elmIdx, elmNum);

		matrix preMat;
		computePreMat(preMat, tanVec, tsvVec, muTanMod, muTsvMod, muAngMod, epvMat);

		matrix dvElaMat;
		matMatMul(dvElaMat, QInvMat, preMat);
		
		vector dv0EVec, dv1EVec, dv2EVec;
		dv0EVec.x = (-dvElaMat.x.x - dvElaMat.y.x) * volVal;
		dv0EVec.y = (-dvElaMat.x.y - dvElaMat.y.y) * volVal;
			
		dv1EVec.x =   dvElaMat.x.x                 * volVal;
		dv1EVec.y =   dvElaMat.x.y                 * volVal;

		dv2EVec.x =                  dvElaMat.y.x  * volVal;
		dv2EVec.y =                  dvElaMat.y.y  * volVal;

		setElement(d_elYElmMat, dv0EVec, dv1EVec, dv2EVec, elmIdx, elmNum);
	}

	return;
}

__global__ void frmElasticGatherKernel(double *d_elYMat, double *d_elYElmMat,
                                       int *d_vtxElmMat, int elmNum, int lmkNum)
{
	int lmkIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkIdx < lmkNum )
	{
		vector elYVec = {0.0, 0.0};

		int adjNum = d_vtxElmMat[lmkIdx];
		for ( int adjIdx = 0; adjIdx < adjNum; ++adjIdx )
		{
			int elmIdx = d_vtxElmMat[(1 + 2 * adjIdx    ) * lmkNum + lmkIdx];
			int lclIdx = d_vtxElmMat[(1 + 2 * adjIdx + 1) * lmkNum + lmkIdx];

			vector elYElmVec;
			getVector(elYElmVec, d_elYElmMat + lclIdx * elmNum * DIMNUM, elmIdx, elmNum);

			vectorSum(elYVec, elYVec, elYElmVec);
		}

		setVector(d_elYMat, elYVec, lmkIdx, lmkNum);
	}

	return;
}

void applyFrameElasticOperator(double *d_elYMat, double *d_lmkEdgMat,
                               double *d_tanMat, double *d_tsvMat, double *h_modVec,
                               double *d_vlcMat, double *d_epvMat, double *d_elYElmMat,
                               int *d_elmVtxMat, int *d_vtxElmMat, int lmkNum, int elmNum)
{
	double muTanMod = h_modVec[0];
	double muTsvMod = h_modVec[1];
	double muAngMod = h_modVec[2];

	computeStrainTensor(d_epvMat, d_lmkEdgMat, d_vlcMat, d_elmVtxMat, lmkNum, elmNum);

	int blkNum = (elmNum - 1) / BLKDIM + 1;
	frmElasticComputeKernel <<<blkNum, BLKDIM>>> (d_elYElmMat, d_lmkEdgMat, d_tanMat, d_tsvMat,
	                                              muTanMod, muTsvMod, muAngMod, d_epvMat, elmNum);

	blkNum = (lmkNum - 1) / BLKDIM + 1;
	frmElasticGatherKernel <<<blkNum, BLKDIM>>> (d_elYMat, d_elYElmMat,
	                                             d_vtxElmMat, elmNum, lmkNum);

	return;
}
