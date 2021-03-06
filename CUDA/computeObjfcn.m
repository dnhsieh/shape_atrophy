function h_objVec = ...
   computeObjfcn(h_tauVarMat, h_tauPrmVec, elaType, h_lmkIniMat, elmObj, tgtObj, vfdObj, ...
                 ldmWgt, knlOrder, knlWidth, knlEps, h_modVec, h_ynkActPrmVec, ...
                 spdTanVal, spdTsvVal, h_reaActPrmVec, ...
                 btmWgt, timeStp, timeEnd, cgItrMax, cgTolVal)

dimNum  = size(h_lmkIniMat, 1);
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

		h_objVec = ...
		   computeObjfcn_2D_frame(h_tauVarMat, h_tauPrmVec, ...
		                          d_lmkIniMat, elmObj.elmVtxMat, elmObj.vtxElmMat, ...
		                          elmObj.tanVtxMat, elmObj.tsvVtxMat, elmObj.bdrVtxMat, elmObj.vtxBdrMat, ...
		                          elmObj.btmVtxMat, elmObj.vtxBtmMat, elmObj.femVtxMat, elmObj.femIfoMat, ...
		                          elmObj.vfdVtxMat, tgtObj.cenPosMat, tgtObj.uniDirMat, tgtObj.elmVolVec, ...
		                          vfdObj.cenKnlType, vfdObj.cenKnlWidth, vfdObj.dirKnlType, vfdObj.dirKnlWidth, ...
		                          ldmWgt, knlOrder, knlWidth, knlEps, h_modVec, ...
		                          spdTanVal, spdTsvVal, h_ynkActPrmVec, h_reaActPrmVec, ...
		                          btmWgt, timeStp, timeNum, cgItrMax, cgTolVal^2);

	end

elseif dimNum == 3

	if strcmpi(elaType, 'isotropic')

	else  % frame

		h_objVec = ...
		   computeObjfcn_3D_frame(h_tauVarMat, h_tauPrmVec, ...
		                          d_lmkIniMat, elmObj.elmVtxMat, elmObj.vtxElmMat, ...
		                          elmObj.nmlVtxMat, elmObj.tsvVtxMat, elmObj.bdrVtxMat, elmObj.vtxBdrMat, ...
		                          elmObj.btmVtxMat, elmObj.vtxBtmMat, elmObj.femVtxMat, elmObj.femIfoMat, ...
		                          elmObj.vfdVtxMat, tgtObj.cenPosMat, tgtObj.uniDirMat, tgtObj.elmVolVec, ...
		                          vfdObj.cenKnlType, vfdObj.cenKnlWidth, vfdObj.dirKnlType, vfdObj.dirKnlWidth, ...
		                          ldmWgt, knlOrder, knlWidth, knlEps, h_modVec, ...
		                          spdTanVal, spdTsvVal, h_ynkActPrmVec, h_reaActPrmVec, ...
		                          btmWgt, timeStp, timeNum, cgItrMax, cgTolVal^2);

	end

else

	error('Dimension %d is not supported.', dimNum);

end
