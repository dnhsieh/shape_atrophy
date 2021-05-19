tauVarVec = [0.5; 0.5; 0.2; 0.3];
tauPrmVec = 1;
% - - -
elaType = 'frame';
modVal  = 3;
modVec  = modVal * [1, 1, 5];
% - - -
spdTanVal = 0.625;
spdTsvVal = 0.125;
% - - -
tauMin       = 0.01;
tauMax       = 1;
actMax       = 4;
ynkActPrmVec = [tauMin, tauMax, actMax, 0.5];
reaActPrmVec = [tauMin, tauMax, actMax, 0.5, 0.3];
% - - -
ldmWgt   = 1e-2;
knlOrder = 3;
knlWidth = 0.2;
knlEps   = 1e-10;
% - - -
btmWgt = 0;
% - - -
timeStp = 0.01;
timeEnd = 1;
% - - -
cgItrMax = 10000;
cgTolVal = 1e-12;
%----------------------------------------------

addpath ../CUDA

load initialShape
elmObj = generateElementObject2D(lyrNdeNum, lyrCntNum, [1, lyrCntNum]);

[lmkStk, tauMat] = ...
   deformation(tauVarVec, tauPrmVec, elaType, ndeIniMat, elmObj, ...
               ldmWgt, knlOrder, knlWidth, knlEps, modVec, ynkActPrmVec, ...
               spdTanVal, spdTsvVal, reaActPrmVec, btmWgt, timeStp, timeEnd, ...
               cgItrMax, cgTolVal);

rmpath ../CUDA
