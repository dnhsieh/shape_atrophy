#include "matvec.h"
#include "utility.h"
#include "constants.h"

void computeStrainTensor(double *, double *, double *, int *, int, int);

__device__ void computePreMat(matrix &preMat,
                              double ldTanMod, double muTanMod, double muTsvMod, double muAngMod,
                              matrix epvMat, vector nmlVec, vector tsvVec)
{
	vector evNVec, evSVec;
	matVecMul(evNVec, epvMat, nmlVec);
	matVecMul(evSVec, epvMat, tsvVec);

	double    NvNVal = dotProduct(nmlVec, evNVec);
	double    SvSVal = dotProduct(tsvVec, evSVec);
	double    NvSVal = dotProduct(nmlVec, evSVec);
	double ldTanvVal = trace(epvMat) - NvNVal;

	// ---

	preMat.x.x = ldTanMod * (1.0 - nmlVec.x * nmlVec.x) * ldTanvVal;
	preMat.x.y = ldTanMod * (    - nmlVec.x * nmlVec.y) * ldTanvVal;
	preMat.x.z = ldTanMod * (    - nmlVec.x * nmlVec.z) * ldTanvVal;

	preMat.y.x = ldTanMod * (    - nmlVec.y * nmlVec.x) * ldTanvVal;
	preMat.y.y = ldTanMod * (1.0 - nmlVec.y * nmlVec.y) * ldTanvVal;
	preMat.y.z = ldTanMod * (    - nmlVec.y * nmlVec.z) * ldTanvVal;

	preMat.z.x = ldTanMod * (    - nmlVec.z * nmlVec.x) * ldTanvVal;
	preMat.z.y = ldTanMod * (    - nmlVec.z * nmlVec.y) * ldTanvVal;
	preMat.z.z = ldTanMod * (1.0 - nmlVec.z * nmlVec.z) * ldTanvVal;

	// ---

	preMat.x.x += muTanMod * (  epvMat.x.x - (2.0 * evNVec.x * nmlVec.x) 
	                          + NvNVal * nmlVec.x * nmlVec.x             );

	preMat.x.y += muTanMod * (  epvMat.x.y - (evNVec.x * nmlVec.y + nmlVec.x * evNVec.y) 
	                          + NvNVal * nmlVec.x * nmlVec.y                             );

	preMat.x.z += muTanMod * (  epvMat.x.z - (evNVec.x * nmlVec.z + nmlVec.x * evNVec.z) 
	                          + NvNVal * nmlVec.x * nmlVec.z                             );


	preMat.y.x += muTanMod * (  epvMat.y.x - (evNVec.y * nmlVec.x + nmlVec.y * evNVec.x) 
	                          + NvNVal * nmlVec.y * nmlVec.x                             );

	preMat.y.y += muTanMod * (  epvMat.y.y - (2.0 * evNVec.y * nmlVec.y) 
	                          + NvNVal * nmlVec.y * nmlVec.y             );

	preMat.y.z += muTanMod * (  epvMat.y.z - (evNVec.y * nmlVec.z + nmlVec.y * evNVec.z) 
	                          + NvNVal * nmlVec.y * nmlVec.z                             );


	preMat.z.x += muTanMod * (  epvMat.z.x - (evNVec.z * nmlVec.x + nmlVec.z * evNVec.x) 
	                          + NvNVal * nmlVec.z * nmlVec.x                             );

	preMat.z.y += muTanMod * (  epvMat.z.y - (evNVec.z * nmlVec.y + nmlVec.z * evNVec.y) 
	                          + NvNVal * nmlVec.z * nmlVec.y                             );

	preMat.z.z += muTanMod * (  epvMat.z.z - (2.0 * evNVec.z * nmlVec.z) 
	                          + NvNVal * nmlVec.z * nmlVec.z             );

	// ---

	preMat.x.x += muTsvMod * SvSVal * tsvVec.x * tsvVec.x;
	preMat.x.y += muTsvMod * SvSVal * tsvVec.x * tsvVec.y;
	preMat.x.z += muTsvMod * SvSVal * tsvVec.x * tsvVec.z;

	preMat.y.x += muTsvMod * SvSVal * tsvVec.y * tsvVec.x;
	preMat.y.y += muTsvMod * SvSVal * tsvVec.y * tsvVec.y;
	preMat.y.z += muTsvMod * SvSVal * tsvVec.y * tsvVec.z;

	preMat.z.x += muTsvMod * SvSVal * tsvVec.z * tsvVec.x;
	preMat.z.y += muTsvMod * SvSVal * tsvVec.z * tsvVec.y;
	preMat.z.z += muTsvMod * SvSVal * tsvVec.z * tsvVec.z;

	// ---

	preMat.x.x += muAngMod * (            2.0 * evSVec.x * tsvVec.x
	                          - NvSVal *  2.0 * nmlVec.x * tsvVec.x );

	preMat.x.y += muAngMod * (           (evSVec.x * tsvVec.y + tsvVec.x * evSVec.y) 
	                          - NvSVal * (nmlVec.x * tsvVec.y + tsvVec.x * nmlVec.y) );

	preMat.x.z += muAngMod * (           (evSVec.x * tsvVec.z + tsvVec.x * evSVec.z) 
	                          - NvSVal * (nmlVec.x * tsvVec.z + tsvVec.x * nmlVec.z) );


	preMat.y.x += muAngMod * (           (evSVec.y * tsvVec.x + tsvVec.y * evSVec.x) 
	                          - NvSVal * (nmlVec.y * tsvVec.x + tsvVec.y * nmlVec.x) );

	preMat.y.y += muAngMod * (            2.0 * evSVec.y * tsvVec.y
	                          - NvSVal *  2.0 * nmlVec.y * tsvVec.y );

	preMat.y.z += muAngMod * (           (evSVec.y * tsvVec.z + tsvVec.y * evSVec.z) 
	                          - NvSVal * (nmlVec.y * tsvVec.z + tsvVec.y * nmlVec.z) );


	preMat.z.x += muAngMod * (           (evSVec.z * tsvVec.x + tsvVec.z * evSVec.x) 
	                          - NvSVal * (nmlVec.z * tsvVec.x + tsvVec.z * nmlVec.x) );

	preMat.z.y += muAngMod * (           (evSVec.z * tsvVec.y + tsvVec.z * evSVec.y) 
	                          - NvSVal * (nmlVec.z * tsvVec.y + tsvVec.z * nmlVec.y) );

	preMat.z.z += muAngMod * (            2.0 * evSVec.z * tsvVec.z
	                          - NvSVal *  2.0 * nmlVec.z * tsvVec.z );

	return;
}

__global__ void frmElasticComputeKernel(double *d_elYElmMat, double *d_lmkEdgMat,
                                        double *d_nmlMat, double *d_tsvMat,
                                        double ldTanMod, double muTanMod, double muTsvMod, double muAngMod,
                                        double *d_epvMat, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		vector q10Vec, q20Vec, q30Vec;
		getEdge(q10Vec, q20Vec, q30Vec, d_lmkEdgMat, elmIdx, elmNum);

		double volVal = computeVolume(q10Vec, q20Vec, q30Vec);

		// Q = [q1 - q0, q2 - q0, q3 - q0]
		matrix QInvMat;
		matInv(QInvMat, q10Vec, q20Vec, q30Vec);

		matrix epvMat;
		getMatrix(epvMat, d_epvMat, elmIdx, elmNum);

		vector nmlVec, tsvVec;
		getVector(nmlVec, d_nmlMat, elmIdx, elmNum);
		getVector(tsvVec, d_tsvMat, elmIdx, elmNum);

		matrix preMat;
		computePreMat(preMat, ldTanMod, muTanMod, muTsvMod, muAngMod, epvMat, nmlVec, tsvVec);
		
		matrix dvElaMat;
		matMatMul(dvElaMat, QInvMat, preMat);

		vector dv0EVec, dv1EVec, dv2EVec, dv3EVec;
		dv0EVec.x = (-dvElaMat.x.x - dvElaMat.y.x - dvElaMat.z.x) * volVal;
		dv0EVec.y = (-dvElaMat.x.y - dvElaMat.y.y - dvElaMat.z.y) * volVal;
		dv0EVec.z = (-dvElaMat.x.z - dvElaMat.y.z - dvElaMat.z.z) * volVal;

		dv1EVec.x =   dvElaMat.x.x                                * volVal;
		dv1EVec.y =   dvElaMat.x.y                                * volVal;
		dv1EVec.z =   dvElaMat.x.z                                * volVal;

		dv2EVec.x =                  dvElaMat.y.x                 * volVal;
		dv2EVec.y =                  dvElaMat.y.y                 * volVal;
		dv2EVec.z =                  dvElaMat.y.z                 * volVal;

		dv3EVec.x =                                 dvElaMat.z.x  * volVal;
		dv3EVec.y =                                 dvElaMat.z.y  * volVal;
		dv3EVec.z =                                 dvElaMat.z.z  * volVal;

		setElement(d_elYElmMat, dv0EVec, dv1EVec, dv2EVec, dv3EVec, elmIdx, elmNum);
	}

	return;
}

__global__ void frmElasticGatherKernel(double *d_elYMat, double *d_elYElmMat,
                                       int *d_vtxElmMat, int elmNum, int lmkNum)
{
	int lmkIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkIdx < lmkNum )
	{
		vector elYVec = {0.0, 0.0, 0.0};

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
                               double *d_nmlMat, double *d_tsvMat, double *h_modVec,
                               double *d_vlcMat, double *d_epvMat, double *d_elYElmMat,
                               int *d_elmVtxMat, int *d_vtxElmMat, int lmkNum, int elmNum) 
{
	double ldTanMod = h_modVec[0];
	double muTanMod = h_modVec[1];
	double muTsvMod = h_modVec[2];
	double muAngMod = h_modVec[3];

	computeStrainTensor(d_epvMat, d_lmkEdgMat, d_vlcMat, d_elmVtxMat, lmkNum, elmNum);

	int blkNum = (elmNum - 1) / BLKDIM + 1;
	frmElasticComputeKernel <<<blkNum, BLKDIM>>> (d_elYElmMat, d_lmkEdgMat, d_nmlMat, d_tsvMat,
	                                              ldTanMod, muTanMod, muTsvMod, muAngMod,
	                                              d_epvMat, elmNum);

	blkNum = (lmkNum - 1) / BLKDIM + 1;
	frmElasticGatherKernel <<<blkNum, BLKDIM>>> (d_elYMat, d_elYElmMat,
	                                             d_vtxElmMat, elmNum, lmkNum);

	return;
}
