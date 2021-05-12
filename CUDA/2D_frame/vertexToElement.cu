#include "matvec.h"
#include "constants.h"

__global__ void vertexToElementKernel(double *d_elmMat, double *d_datMat,
                                      int *d_elmVtxMat, int datNum, int elmNum)
{
	int elmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( elmIdx < elmNum )
	{
		int q0Idx = d_elmVtxMat[             elmIdx];
		int q1Idx = d_elmVtxMat[    elmNum + elmIdx];
		int q2Idx = d_elmVtxMat[2 * elmNum + elmIdx];

		vector q0Vec, q1Vec, q2Vec;
		getVector(q0Vec, d_datMat, q0Idx, datNum);
		getVector(q1Vec, d_datMat, q1Idx, datNum);
		getVector(q2Vec, d_datMat, q2Idx, datNum);

		setElement(d_elmMat, q0Vec, q1Vec, q2Vec, elmIdx, elmNum);
	}

	return;
}

void vertexToElement(double *d_elmMat, double *d_datMat, int *d_elmVtxMat, int datNum, int elmNum)
{
	int blkNum = (elmNum - 1) / BLKDIM + 1;
	vertexToElementKernel <<<blkNum, BLKDIM>>> (d_elmMat, d_datMat, d_elmVtxMat, datNum, elmNum);

	return;
}
