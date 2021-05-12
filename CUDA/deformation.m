function [h_lmkStk, h_tauMat] = ...
   deformation(h_tauVarVec, h_tauPrmVec, elaType, h_lmkIniMat, elmObj, ...
               ldmWgt, knlOrder, knlWidth, knlEps, h_modVec, h_ynkActPrmVec, ...
               spdTanVal, spdTsvVal, h_reaActPrmVec, ...
               btmWgt, timeStp, timeEnd, cgItrMax, cgTolVal)

[dimNum, lmkNum] = size(h_lmkIniMat);
timeNum = floor(timeEnd / timeStp) + 1;

d_lmkIniMat = gpuArray(h_lmkIniMat');

if strcmpi(elaType, 'isotropic')
	d_modMat = gpuArray(reshape(h_modVec, 2, [])');
elseif strcmpi(elaType, 'frame')
	h_modVec = h_modVec(:);
else
	error('Unknown elasticity type %s.', elaType);
end

if dimNum == 2

	if strcmpi(elaType, 'isotropic')

	else  % frame

		[h_lmkStk, h_tauMat] = ...
		   deformation_2D_frame(h_tauVarVec, h_tauPrmVec, ...
		                        d_lmkIniMat, elmObj.elmVtxMat, elmObj.vtxElmMat, ...
		                        elmObj.tanVtxMat, elmObj.tsvVtxMat, elmObj.bdrVtxMat, elmObj.vtxBdrMat, ...
		                        elmObj.btmVtxMat, elmObj.vtxBtmMat, elmObj.femVtxMat, elmObj.femIfoMat, ...
		                        ldmWgt, knlOrder, knlWidth, knlEps, h_modVec, ...
		                        spdTanVal, spdTsvVal, h_ynkActPrmVec, h_reaActPrmVec, ...
		                        btmWgt, timeStp, timeNum, cgItrMax, cgTolVal^2);

	end

elseif dimNum == 3

	if strcmpi(elaType, 'isotropic')

	else  % frame

		[h_lmkStk, h_tauMat] = ...
		   deformation_3D_frame(h_tauVarVec, h_tauPrmVec, ...
		                        d_lmkIniMat, elmObj.elmVtxMat, elmObj.vtxElmMat, ...
		                        elmObj.nmlVtxMat, elmObj.tsvVtxMat, elmObj.bdrVtxMat, elmObj.vtxBdrMat, ...
		                        elmObj.btmVtxMat, elmObj.vtxBtmMat, elmObj.femVtxMat, elmObj.femIfoMat, ...
		                        ldmWgt, knlOrder, knlWidth, knlEps, h_modVec, ...
		                        spdTanVal, spdTsvVal, h_ynkActPrmVec, h_reaActPrmVec, ...
		                        btmWgt, timeStp, timeNum, cgItrMax, cgTolVal^2);

	end

else

	error('Dimension %d is not supported.', dimNum);

end

h_lmkStk    = permute(reshape(h_lmkStk,    lmkNum, dimNum, timeNum    ), [2, 1, 3]);
h_tauMat    =         reshape(h_tauMat,    lmkNum,         timeNum    );

