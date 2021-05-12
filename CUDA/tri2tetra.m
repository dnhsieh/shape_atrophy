function [tetVtxMat, nmlVtxMat, tsvVtxMat] = tri2tetra(triVtxMat, lyrCntNum)

lblMat = circshift(triVtxMat, -1) - triVtxMat > 0;

lyrLmkNum = max(triVtxMat(:));
lyrTriNum = size(triVtxMat, 2);

tetVtxStk = zeros(4, 3 * lyrTriNum, lyrCntNum - 1);
nmlVtxStk = zeros(6, 3 * lyrTriNum, lyrCntNum - 1);
tsvVtxStk = zeros(6, 3 * lyrTriNum, lyrCntNum - 1);
for lyrTriIdx = 1 : lyrTriNum

	triVtxLow = triVtxMat(:, lyrTriIdx);
	triVtxUpp = triVtxLow + lyrLmkNum;

	if all(lblMat(:, lyrTriIdx) == [0; 0; 1])

		tetVtxVecs = [triVtxLow(1), triVtxLow(2), triVtxLow(3), triVtxUpp(1); ...
		              triVtxLow(2), triVtxLow(3), triVtxUpp(1), triVtxUpp(2); ...
		              triVtxLow(3), triVtxUpp(1), triVtxUpp(2), triVtxUpp(3)]';

	elseif all(lblMat(:, lyrTriIdx) == [0; 1; 0])

		tetVtxVecs = [triVtxLow(1), triVtxLow(2), triVtxLow(3), triVtxUpp(3); ...
		              triVtxUpp(3), triVtxLow(1), triVtxLow(2), triVtxUpp(1); ...
		              triVtxLow(2), triVtxUpp(1), triVtxUpp(2), triVtxUpp(3)]';

	elseif all(lblMat(:, lyrTriIdx) == [0; 1; 1])

		tetVtxVecs = [triVtxLow(1), triVtxLow(2), triVtxLow(3), triVtxUpp(1); ...
		              triVtxLow(2), triVtxLow(3), triVtxUpp(1), triVtxUpp(3); ...
		              triVtxUpp(3), triVtxUpp(1), triVtxLow(2), triVtxUpp(2)]';

	elseif all(lblMat(:, lyrTriIdx) == [1; 0; 0])

		tetVtxVecs = [triVtxLow(1), triVtxLow(2), triVtxLow(3), triVtxUpp(2); ...
		              triVtxLow(1), triVtxLow(3), triVtxUpp(3), triVtxUpp(2); ...
		              triVtxUpp(2), triVtxUpp(3), triVtxLow(1), triVtxUpp(1)]';

	elseif all(lblMat(:, lyrTriIdx) == [1; 0; 1])

		tetVtxVecs = [triVtxLow(1), triVtxLow(2), triVtxLow(3), triVtxUpp(2); ...
		              triVtxLow(3), triVtxUpp(1), triVtxLow(1), triVtxUpp(2); ...
		              triVtxUpp(1), triVtxLow(3), triVtxUpp(3), triVtxUpp(2)]';

	else % [1; 1; 0]
	
		tetVtxVecs = [triVtxLow(1), triVtxLow(2), triVtxLow(3), triVtxUpp(3); ...
		              triVtxUpp(3), triVtxLow(1), triVtxLow(2), triVtxUpp(2); ...
		              triVtxUpp(2), triVtxUpp(3), triVtxLow(1), triVtxUpp(1)]';

	end

	tetVtxStk(:, 3 * (lyrTriIdx - 1) + (1 : 3), 1) = tetVtxVecs;
	nmlVtxStk(:, 3 * (lyrTriIdx - 1) + (1 : 3), 1) = repmat([triVtxLow(:); triVtxUpp(:)], 1, 3);
	tsvVtxStk(:, 3 * (lyrTriIdx - 1) + (1 : 3), 1) = repmat(reshape([triVtxLow(:), triVtxUpp(:)]', 6, 1), 1, 3);

end

for lyrIdx = 2 : (lyrCntNum - 1)
	tetVtxStk(:, :, lyrIdx) = tetVtxStk(:, :, lyrIdx - 1) + lyrLmkNum;
	nmlVtxStk(:, :, lyrIdx) = nmlVtxStk(:, :, lyrIdx - 1) + lyrLmkNum;
	tsvVtxStk(:, :, lyrIdx) = tsvVtxStk(:, :, lyrIdx - 1) + lyrLmkNum;
end

tetVtxMat = reshape(tetVtxStk, 4, 3 * lyrTriNum * (lyrCntNum - 1));
nmlVtxMat = reshape(nmlVtxStk, 6, 3 * lyrTriNum * (lyrCntNum - 1));
tsvVtxMat = reshape(tsvVtxStk, 6, 3 * lyrTriNum * (lyrCntNum - 1));

