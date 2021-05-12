#include "matvec.h"
#include "constants.h"

__global__ void vertexToBottomKernel(double *d_btmMat, double *d_datMat,
                                     int *d_btmVtxMat, int datNum, int btmElmNum)
{
	int btmElmIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( btmElmIdx < btmElmNum )
	{
		int q0Idx = d_btmVtxMat[                btmElmIdx];
		int q1Idx = d_btmVtxMat[    btmElmNum + btmElmIdx];
		int q2Idx = d_btmVtxMat[2 * btmElmNum + btmElmIdx];

		vector q0Vec, q1Vec, q2Vec;
		getVector(q0Vec, d_datMat, q0Idx, datNum);
		getVector(q1Vec, d_datMat, q1Idx, datNum);
		getVector(q2Vec, d_datMat, q2Idx, datNum);

		setBoundary(d_btmMat, q0Vec, q1Vec, q2Vec, btmElmIdx, btmElmNum);
	}

	return;
}

void vertexToBottom(double *d_btmMat, double *d_datMat, int *d_btmVtxMat,
                    int datNum, int btmElmNum)
{
	int blkNum = (btmElmNum - 1) / BLKDIM + 1;
	vertexToBottomKernel <<<blkNum, BLKDIM>>> (d_btmMat, d_datMat, d_btmVtxMat, datNum, btmElmNum);

	return;
}
