dimNum    = 2;
lyrNdeNum = 101;
lyrCntNum = 31;

xVec = linspace(-3, 3, lyrNdeNum);
yVec = linspace(-1, 1, lyrCntNum);

ndeStk = zeros(dimNum, lyrNdeNum, lyrCntNum);
for lyrIdx = 1 : lyrCntNum
	ndeStk(1, :, lyrIdx) = xVec;
	ndeStk(2, :, lyrIdx) = yVec(lyrIdx);
end

ndeIniMat = reshape(ndeStk, dimNum, lyrNdeNum * lyrCntNum);
save initialShape ndeIniMat lyrNdeNum lyrCntNum
