import numpy as np
from Evaluation_Imagefusion import evaluation
from Global_Vars import Global_Vars
from Model_RANet import Model_RANet


def objfun_feat(Soln):
    CT_img = Global_Vars.CT_img
    PET_img = Global_Vars.PET_img
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            Eval, Fused_Img = Model_RANet(CT_img, PET_img, sol)
            Eval = evaluation(CT_img, PET_img, Fused_Img)
            Fitn[i] = 1 / (Eval[1] + Eval[0])  # PSNR + SSIM
        return Fitn
    else:
        sol = Soln
        Eval, Fused_Img = Model_RANet(CT_img, PET_img, sol)
        Eval = evaluation(CT_img, PET_img, Fused_Img)
        Fitn = 1 / (Eval[1] + Eval[0])  # PSNR + SSIM
        return Fitn
