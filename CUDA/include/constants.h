#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifdef DIM2

#define DIMNUM 2
#define VTXNUM 3
#define TANNUM 2
#define TSVNUM 2

#elif DIM3

#define DIMNUM 3
#define VTXNUM 4
#define NMLNUM 2
#define TSVNUM 3

#endif

#define EPS       1e-14
#define BLKDIM    128
#define BLKROW    16
#define SUMBLKDIM 512

#endif
