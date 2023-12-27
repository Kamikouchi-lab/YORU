import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import ndimage
from skimage import io
from skimage.transform import resize


def test():
    pass


class loadingParam:  # loading parameters from serial
    def __init__(self, paramFile="../asovi_default.yaml"):
        self.file = paramFile
        print(self.file)
        with open(self.file) as file:
            self.yml = yaml.safe_load(file)
            yaml.safe_load_all


class asvTexture:
    def __init__(self, paramFile="asovi_default.ini"):
        self.yml = loadingParam(paramFile)
        self.yml = self.yml.yml
        self.Textures = []
        self.FileNames = []
        self.Sizes = []

    def showParameters(self):
        print(self.yml)

    def drawFloor(self, fName="Floor.png", show=True):
        np.random.seed(seed=self.yml["texture"]["rSeed"])
        tex1 = np.random.rand(2 * 2, 256 * 2) * 1.0

        # Texture Symm
        thre = self.yml["texture"]["cd_thre"]
        tex1[tex1 > thre] = self.yml["texture"]["contrast"][1]
        tex1[tex1 <= thre] = self.yml["texture"]["contrast"][0]
        tex1 = resize(tex1, (tex1.shape[0] * 8, tex1.shape[1] * 8))
        tex1 = np.concatenate([tex1, np.flipud(tex1)], axis=0)
        tex1 = ndimage.filters.gaussian_filter(tex1, self.yml["texture"]["FreqFloor"])
        tex1 = tex1 / np.max(tex1[:])

        tex1RGB = np.zeros((np.shape(tex1)[0], np.shape(tex1)[1], 3), dtype="uint8")
        for idx, lumi in enumerate(self.yml["texture"]["RGB"]):
            print(idx)
            tex1RGB[:, :, idx] = tex1 * lumi * 255.0

        if show:  # show drawn texture
            plt.figure(figsize=[20, 3])
            plt.subplot(2, 1, 1)
            plt.imshow(tex1RGB)
            plt.subplot(2, 1, 2)
            plt.imshow(tex1, cmap="gray")
            plt.show()

        self.Textures.append(tex1RGB)
        self.FileNames.append(fName)
        self.Sizes.append(np.shape(tex1RGB))

    def drawSideWall(self, fName="Wall.png", show=True):
        np.random.seed(seed=self.yml["texture"]["rSeed"])
        tex1 = np.random.rand(4 * 2, 256 * 2) * 1.0

        # Texture  no Symm
        thre = self.yml["texture"]["cd_thre"]
        tex1[tex1 > thre] = self.yml["texture"]["contrast"][1]
        tex1[tex1 <= thre] = self.yml["texture"]["contrast"][0]
        tex1 = resize(tex1, (tex1.shape[0] * 4 * 4, tex1.shape[1] * 4 * 4))
        tex1 = ndimage.filters.gaussian_filter(tex1, self.yml["texture"]["FreqWall"])
        tex1 = tex1 / np.max(tex1[:])
        tex1RGB = np.zeros((np.shape(tex1)[0], np.shape(tex1)[1], 3), dtype="uint8")
        for idx, lumi in enumerate(self.yml["texture"]["RGB"]):
            print(idx)
            tex1RGB[:, :, idx] = tex1 * lumi * 255.0

        if show:  # show drawn texture
            plt.figure(figsize=[20, 3])
            plt.subplot(2, 1, 1)
            plt.imshow(tex1RGB)
            plt.subplot(2, 1, 2)
            plt.imshow(tex1, cmap="gray")
            plt.show()

        self.Textures.append(tex1RGB)
        self.FileNames.append(fName)
        self.Sizes.append(np.shape(tex1RGB))

    def drawFloor_testPattern(self, fName="Floor.png", show=True):
        np.random.seed(seed=10)
        tex1 = np.random.rand(4, 512) * 1.0

        # Texture Symm
        thre = 0.5
        tex1[tex1 > thre] = 1.0
        tex1[tex1 <= thre] = 0
        tex1 = resize(tex1, (tex1.shape[0] * 4, tex1.shape[1] * 4))
        tex1 = np.concatenate([tex1, np.flipud(tex1)], axis=0)
        tex1 = ndimage.filters.gaussian_filter(tex1, 10)
        tex1 = tex1 / np.max(tex1[:])

        tex1RGB = np.zeros((np.shape(tex1)[0], np.shape(tex1)[1], 3), dtype="uint8")
        for idx, lumi in enumerate([0.1, 0.1, 0.1]):
            print(idx)
            tex1RGB[:, :, idx] = tex1 * lumi * 255.0
        Rvec = np.linspace(0, 1, np.shape(tex1)[0])
        Bvec = np.linspace(0, 1, np.shape(tex1)[1])
        for idx in range(np.shape(tex1)[0]):
            tex1RGB[idx, :, 0] = Rvec[idx] * 255
        for idx in range(np.shape(tex1)[1]):
            tex1RGB[:, idx, 2] = Bvec[idx] * 255

        self.Textures.append(tex1RGB)
        self.FileNames.append(fName)
        self.Sizes.append(np.shape(tex1RGB))

    def drawGoalWall(self, fName="Goal.png"):
        texF = 1 * np.ones((8, 8), dtype="uint8")
        texF[1:7, 1:7] = 0
        texF[2:6, 2:6] = 1
        texF[3:5, 3:5] = 0
        texF[texF <= 0] = 0
        texF[texF > 0] = 255.0

    def exportAll(self, exportpath="." + os.sep):
        print("=== Texture Exporting ===")
        for idx, texfile in enumerate(self.FileNames):
            io.imsave(exportpath + texfile, np.uint8(self.Textures[idx]))
            print("Exported: " + exportpath + texfile)
        print("**** Complete ****")


if __name__ == "__main__":
    hoge = loadingParam(
        paramFile=r"C:\Users\Acetylcholine1\Documents\VirtualReality\Asovi_beta_v20220505\asovi_default.yaml"
    )
