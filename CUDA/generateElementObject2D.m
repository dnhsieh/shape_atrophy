function elmObj = generateElementObject2D(lyrLmkNum, lyrCntNum, vfdLyrVec)

lmkNum = lyrLmkNum * lyrCntNum;
elmNum = (lyrLmkNum - 1) * (lyrCntNum - 1) * 2;

% - - -

elmVtxMat = zeros(3, elmNum);
tanVtxMat = zeros(4, elmNum);
tsvVtxMat = zeros(4, elmNum);

elmIdx = 1;
for lyrCntIdx = 1 : (lyrCntNum - 1)
	for lyrLmkIdx = 1 : (lyrLmkNum - 1)

		lmkIdx = (lyrCntIdx - 1) * lyrLmkNum + lyrLmkIdx;

		elmVtxMat(:, elmIdx    ) = lmkIdx + [0; 1; lyrLmkNum];
		elmVtxMat(:, elmIdx + 1) = lmkIdx + [1; lyrLmkNum + 1; lyrLmkNum];

		tanVtxVec = lmkIdx + [0; 1; lyrLmkNum; lyrLmkNum + 1];
		tsvVtxVec = lmkIdx + [0; lyrLmkNum; 1; lyrLmkNum + 1];

		tanVtxMat(:, [elmIdx, elmIdx + 1]) = repmat(tanVtxVec, 1, 2);
		tsvVtxMat(:, [elmIdx, elmIdx + 1]) = repmat(tsvVtxVec, 1, 2);

		elmIdx = elmIdx + 2;

	end
end

elmCntVec = histcounts(elmVtxMat(:), 1 : (lmkNum + 1));
vtxElmMat = zeros(1 + 2 * max(elmCntVec), lmkNum);
for lmkIdx = 1 : lmkNum

	[lclIdxVec, elmIdxVec] = find(elmVtxMat == lmkIdx);
	adjNum = length(lclIdxVec);
	vtxElmMat(1 : (1 + 2 * adjNum), lmkIdx) = ...
	   [adjNum; reshape([elmIdxVec(:), lclIdxVec(:)]' - 1, 2 * adjNum, 1)];

end

% - - -

triObj    = triangulation(elmVtxMat', rand(lmkNum, 2));
bdrVtxMat = triObj.freeBoundary';

bdrCntVec = histcounts(bdrVtxMat(:), 1 : (lmkNum + 1));
vtxBdrMat = zeros(1 + 2 * max(bdrCntVec), lmkNum);
for lmkIdx = 1 : lmkNum

	[lclIdxVec, bdrIdxVec] = find(bdrVtxMat == lmkIdx);
	adjNum = length(lclIdxVec);
	vtxBdrMat(1 : (1 + 2 * adjNum), lmkIdx) = ...
	   [adjNum; reshape([bdrIdxVec(:), lclIdxVec(:)]' - 1, 2 * adjNum, 1)];

end

% - - -

btmVtxMat = [1 : (lyrLmkNum - 1); 2 : lyrLmkNum];

btmCntVec = histcounts(btmVtxMat(:), 1 : (lyrLmkNum + 1));
vtxBtmMat = zeros(1 + 2 * max(btmCntVec), lyrLmkNum);
for lyrLmkIdx = 1 : lyrLmkNum

	[lclIdxVec, btmElmIdxVec] = find(btmVtxMat == lyrLmkIdx);
	adjNum = length(lclIdxVec);
	vtxBtmMat(1 : (1 + 2 * adjNum), lyrLmkIdx) = ...
	   [adjNum; reshape([btmElmIdxVec(:), lclIdxVec(:)]' - 1, 2 * adjNum, 1)];
		
end

% - - -

edgMat = triObj.edges;
edgNum = size(edgMat, 1);
nzrNum = lmkNum + 2 * edgNum;

femVtxMat = [repmat(1 : lmkNum, 2, 1), edgMat', edgMat(:, [2, 1])'];

glbIdxMat = reshape(1 : (elmNum * 6), elmNum, 6);
femIfoMat = zeros(1 + 1 + 2 * max(elmCntVec), nzrNum);

femIfoMat(2, 1 : lmkNum) = 2;
femIfoMat(2, (lmkNum + 1) : nzrNum) = 1;

for elmIdx = 1 : elmNum

	elmVtxVec = elmVtxMat(:, elmIdx);

	for lcl1Idx = 1 : 3
		for lcl2Idx = 1 : 3

			lmk1Idx = elmVtxVec(lcl1Idx);
			lmk2Idx = elmVtxVec(lcl2Idx);
			colIdx  = find(sum(abs(femVtxMat - [lmk1Idx; lmk2Idx])) == 0);

			adjNum = femIfoMat(1, colIdx);
			
			femIfoMat(2 + 2 * adjNum + 1, colIdx) = elmIdx - 1;

			if      lcl1Idx == 1 && lcl2Idx == 1
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx, 1) - 1;
			elseif (lcl1Idx == 1 && lcl2Idx == 2) || (lcl1Idx == 2 && lcl2Idx == 1)
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx, 2) - 1;
			elseif (lcl1Idx == 1 && lcl2Idx == 3) || (lcl1Idx == 3 && lcl2Idx == 1)
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx, 3) - 1;
			elseif  lcl1Idx == 2 && lcl2Idx == 2
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx, 4) - 1;
			elseif (lcl1Idx == 2 && lcl2Idx == 3) || (lcl1Idx == 3 && lcl2Idx == 2)
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx, 5) - 1;
			elseif  lcl1Idx == 3 && lcl2Idx == 3
				femIfoMat(2 + 2 * adjNum + 2, colIdx) = glbIdxMat(elmIdx, 6) - 1;
			end
		
			femIfoMat(1, colIdx) = adjNum + 1;
			
		end
	end

end

[femVtxMat, srtIdx] = sortrows(femVtxMat');
femVtxMat = femVtxMat';
femIfoMat = femIfoMat(:, srtIdx);

% - - -

glbIdxMat = reshape(1 : lmkNum, lyrLmkNum, lyrCntNum);
vfdVtxStk = zeros(2, lyrLmkNum - 1, length(vfdLyrVec));
for bdrLyrIdx = 1 : length(vfdLyrVec)
	vfdVtxStk(:, :, bdrLyrIdx) = [glbIdxMat(1 : (lyrLmkNum - 1), vfdLyrVec(bdrLyrIdx)), ...
	                              glbIdxMat(2 :  lyrLmkNum,      vfdLyrVec(bdrLyrIdx))]';
end
vfdVtxMat = reshape(vfdVtxStk, 2, (lyrLmkNum - 1) * length(vfdLyrVec));

% - - -

elmObj.elmVtxMat = gpuArray(int32(elmVtxMat' - 1));
elmObj.vtxElmMat = gpuArray(int32(vtxElmMat'    ));
elmObj.tanVtxMat = gpuArray(int32(tanVtxMat' - 1));
elmObj.tsvVtxMat = gpuArray(int32(tsvVtxMat' - 1));
elmObj.bdrVtxMat = gpuArray(int32(bdrVtxMat' - 1));
elmObj.vtxBdrMat = gpuArray(int32(vtxBdrMat'    ));
elmObj.btmVtxMat = gpuArray(int32(btmVtxMat' - 1));
elmObj.vtxBtmMat = gpuArray(int32(vtxBtmMat'    ));
elmObj.femVtxMat = gpuArray(int32(femVtxMat' - 1));
elmObj.femIfoMat = gpuArray(int32(femIfoMat'    ));
elmObj.vfdVtxMat = gpuArray(int32(vfdVtxMat' - 1));

