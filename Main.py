import os
from numpy import matlib
from CSA import CSA
from FFO import FFO
from Global_Vars import Global_Vars
from HOA import HOA
from Image_Results import *
from Model_CGAN import Model_CGAN
from Model_GCNN import Model_GCNN
from Model_RANet import Model_RANet
from Model_SCDL import Model_SCDL
from NGO import NGO
from Plot_Results import *
from Proposed import Proposed
from objfun_feat import objfun_feat
import warnings
warnings.filterwarnings('ignore')


def ReadImage(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (512, 512))
    return image


# Read CT Image
an = 0
if an == 1:
    Image = []
    Path = './Dataset/DB/CT'
    in_dir = os.listdir(Path)
    for i in range(len(in_dir)):
        file = Path + '/' + in_dir[i]
        Img = ReadImage(file)
        Image.append(Img)
    np.save('CT_Image.npy', np.asarray(Image))

# Read PET Image
an = 0
if an == 1:
    Image = []
    Path = './Dataset/DB/PET'
    in_dir = os.listdir(Path)
    for i in range(len(in_dir)):
        files = Path + '/' + in_dir[i]
        Img = ReadImage(files)
        Image.append(Img)
    np.save('PET_Image.npy', np.asarray(Image))

# Optimization for Image Fusion
an = 0
if an == 1:
    CT_img = np.load('CT_Image.npy', allow_pickle=True)  # Load the Dataset
    PET_img = np.load('PET_Image.npy', allow_pickle=True)
    Best_sol = []
    Global_Vars.PET_img = PET_img
    Global_Vars.CT_img = CT_img
    Npop = 10
    Chlen = 2
    xmin = matlib.repmat([CT_img.shape[1] * (-0.2), PET_img.shape[1] * (-0.2)], Npop, 1)
    xmax = matlib.repmat([CT_img.shape[1] * 0.2, PET_img.shape[1] * 0.2], Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
    fname = objfun_feat
    max_iter = 50

    print('CSA....')
    [bestfit1, fitness1, bestsol1, Time1] = CSA(initsol, fname, xmin, xmax, max_iter)  # CSA

    print('FFO....')
    [bestfit2, fitness2, bestsol2, Time2] = FFO(initsol, fname, xmin, xmax, max_iter)  # FFO

    print('NGO....')
    [bestfit3, fitness3, bestsol3, Time3] = NGO(initsol, fname, xmin, xmax, max_iter)  # NGO

    print('HOA....')
    [bestfit4, fitness4, bestsol4, Time4] = HOA(initsol, fname, xmin, xmax, max_iter)  # HOA

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)  # Proposed

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('Best_Sol.npy', BestSol)

# Image Fusion
an = 0
if an == 1:
    Image = np.load('CT_Image.npy', allow_pickle=True)
    Pet_Img = np.load('PET_Image.npy', allow_pickle=True)
    BestSol = np.load('Best_Sol.npy', allow_pickle=True)
    Act = []
    for n in range(Image.shape[0]):
        Ct = Image[n]
        Pet = Pet_Img[n]
        Eval = np.zeros((10, 10))
        for j in range(BestSol.shape[0]):
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], fused = Model_RANet(Ct, Pet, sol)  # RANet With optimization
        Eval[5, :], fused_Image1 = Model_CGAN(Ct, Pet)  # Model CGAN
        Eval[6, :], fused_Image2 = Model_GCNN(Ct, Pet)  # Model GCNN
        Eval[7, :], fused_Image3 = Model_SCDL(Ct, Pet)  # Model SCDL
        Eval[8, :], fused_Image4 = Model_RANet(Ct, Pet)  # RANet  Without optimization
        Eval[9, :] = Eval[4, :]
        Act.append(Eval)
        np.save('Fused_Image.npy', fused_Image4)
    np.save('Evaluate_all.npy', Act)  # Save Eval all


plotConvResults()
plot_Alg_Results()
plot_Bar_Results()
Table()
Image_Results()
