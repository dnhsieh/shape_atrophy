function elmObj = generateElementObject3D(tetObj, btmVtxMat, nmlVtxMat, tsvVtxMat, vfdLyrVec)

elmVtxMat = tetObj.ConnectivityList';
lmkNum    = max(elmVtxMat(:));
lyrLmkNum = max(btmVtxMat(:));
lyrCntNum = lmkNum / lyrLmkNum;
elmNum    = size(elmVtxMat, 2);

elmCntVec = histcounts(elmVtxMat(:), 1 : (lmkNum + 1));
vtxElmMat = zeros(1 + 2 * max(elmCntVec), lmkNum);
for lmkIdx = 1 : lmkNum

	[lclIdxVec, elmIdxVec] = find(elmVtxMat == lmkIdx);
	adjNum = length(lclIdxVec);
	vtxElmMat(1 : (1 + 2 * adjNum), lmkIdx) = ...
	   [adjNum; reshape([elmIdxVec(:), lclIdxVec(:)]' - 1, 2 * adjNum, 1)];

end

% - - -

bdrVtxMat = tetObj.freeBoundary';

bdrCntVec = histcounts(bdrVtxMat(:), 1 : (lmkNum + 1));
vtxBdrMat = zeros(1 + 2 * max(bdrCntVec), lmkNum);
for lmkIdx = 1 : lmkNum

	[lclIdxVec, bdrIdxVec] = find(bdrVtxMat == lmkIdx);
	adjNum = length(lclIdxVec);
	vtxBdrMat(1 : (1 + 2 * adjNum), lmkIdx) = ...
	   [adjNum; reshape([bdrIdxVec(:), lclIdxVec(:)]' - 1, 2 * adjNum, 1)];

end

% - - -

btmCntVec = histcounts(btmVtxMat(:), 1 : (lyrLmkNum + 1));
vtxBtmMat = zeros(1 + 2 * max(btmCntVec), lyrLmkNum);
for lyrLmkIdx = 1 : lyrLmkNum

	[lclIdxVec, btmElmIdxVec] = find(btmVtxMat == lyrLmkIdx);
	adjNum = length(lclIdxVec);
	vtxBtmMat(1 : (1 + 2 * adjNum), lyrLmkIdx) = ...
	   [adjNum; reshape([btmElmIdxVec(:), lclIdxVec(:)]' - 1, 2 * adjNum, 1)];
		
end

% - - -

edgMat = tetObj.edges;
edgNum = size(edgMat, 1);
nzrNum = lmkNum + 2 * edgNum;

femVtxMat = [repmat(1 : lmkNum, 2, 1), edgMat', edgMat(:, [2, 1])'];

glbIdxMat = reshape(1 : (elmNum * 10), elmNum, 10);
femIfoMat = zeros(1 + 1 + 2 * max(elmCntVec), nzrNum);

femIfoMat(2, 1 : lmkNum) = 2;
femIfoMat(2, (lmkNum + 1) : nzrNum) = 1;

for elmIdx = 1 : elmNum

	elmVtxVec = elmVtxMat(:, elmIdx);

	for lcl1Idx = 1 : 4
		for lcl2Idx = 1 : 4

			lmk1Idx = elmVtxVec(lcl1Idx);
			lmk2Idx = elmVtxVec(lcl2Idx);
			colIdx  = find(sum(abs(femVtxMat - [lmk1Idx; lmk2Idx])) == 0);

			adjNum = femIfoMat(1, colIdx);
			
			femIfoMat(2 + 2 * adjNum + 1, colIdx) = elmIdx - 1;

			if      lcl1Idx == 1 && lcl2Idx == 1
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx,  1) - 1;
			elseif (lcl1Idx == 1 && lcl2Idx == 2) || (lcl1Idx == 2 && lcl2Idx == 1)
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx,  2) - 1;
			elseif (lcl1Idx == 1 && lcl2Idx == 3) || (lcl1Idx == 3 && lcl2Idx == 1)
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx,  3) - 1;
			elseif (lcl1Idx == 1 && lcl2Idx == 4) || (lcl1Idx == 4 && lcl2Idx == 1)
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx,  4) - 1;
			elseif  lcl1Idx == 2 && lcl2Idx == 2
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx,  5) - 1;
			elseif (lcl1Idx == 2 && lcl2Idx == 3) || (lcl1Idx == 3 && lcl2Idx == 2)
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx,  6) - 1;
			elseif (lcl1Idx == 2 && lcl2Idx == 4) || (lcl1Idx == 4 && lcl2Idx == 2)
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx,  7) - 1;
			elseif  lcl1Idx == 3 && lcl2Idx == 3
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx,  8) - 1;
			elseif (lcl1Idx == 3 && lcl2Idx == 4) || (lcl1Idx == 4 && lcl2Idx == 3)
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx,  9) - 1;
			elseif  lcl1Idx == 4 && lcl2Idx == 4
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx, 10) - 1;
			end
		
			femIfoMat(1, colIdx) = adjNum + 1;
			
		end
	end

end

[femVtxMat, srtIdx] = sortrows(femVtxMat');
femVtxMat = femVtxMat';
femIfoMat = femIfoMat(:, srtIdx);

% - - -

vfdVtxStk = btmVtxMat + reshape((vfdLyrVec - 1) * lyrLmkNum, 1, 1, length(vfdLyrVec));
vfdVtxMat = reshape(vfdVtxStk, 3, size(btmVtxMat, 2) * length(vfdLyrVec));

% - - -

elmObj.elmVtxMat = gpuArray(int32(elmVtxMat' - 1));
elmObj.vtxElmMat = gpuArray(int32(vtxElmMat'    ));
elmObj.nmlVtxMat = gpuArray(int32(nmlVtxMat' - 1));
elmObj.tsvVtxMat = gpuArray(int32(tsvVtxMat' - 1));
elmObj.bdrVtxMat = gpuArray(int32(bdrVtxMat' - 1));
elmObj.vtxBdrMat = gpuArray(int32(vtxBdrMat'    ));
elmObj.btmVtxMat = gpuArray(int32(btmVtxMat' - 1));
elmObj.vtxBtmMat = gpuArray(int32(vtxBtmMat'    ));
elmObj.femVtxMat = gpuArray(int32(femVtxMat' - 1));
elmObj.femIfoMat = gpuArray(int32(femIfoMat'    ));
elmObj.vfdVtxMat = gpuArray(int32(vfdVtxMat' - 1));

