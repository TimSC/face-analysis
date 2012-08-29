
from PIL import Image
import numpy as np
import math

def GetBilinearPixel(im, imload, pos):
	modX, modY = map(math.modf, pos)
	bl = np.array(imload[modX[1], modY[1]])
	br = np.array(imload[modX[1]+1, modY[1]])
	tl = np.array(imload[modX[1], modY[1]+1])
	tr = np.array(imload[modX[1]+1, modY[1]+1])
	
	b = modX[0] * br + (1. - modX[0]) * bl
	t = modX[0] * tr + (1. - modX[0]) * tl

	return modY[0] * t + (1. - modY[0]) * b

def Warp(inImg, inImgL, outImg, outImgL, inTriangle, triAffines):
	#Synthesis shape norm image		
	for i in range(outImg.size[0]):
		for j in range(outImg.size[1]):
			normSpaceCoord = (float(i)/outImg.size[0],float(j)/outImg.size[1])
			tri = inTriangle[i,j]
			if tri == -1: continue
			affine = triAffines[tri]
				
			#Calculate position in the input image
			homogCoord = (normSpaceCoord[0], normSpaceCoord[1], 1.)
			outImgCoord = np.dot(affine, homogCoord)

			try:
				#Nearest neighbour
				#outImgL[i,j] = inImgL[int(round(inImgCoord[0])),int(round(inImgCoord[1]))]

				#Bilinear sampling
				#print i,j,outImgCoord[0:2],im.size
				outImgL[i,j] = tuple(map(int,np.round(GetBilinearPixel(inImg, inImgL, outImgCoord[0:2]))))
				#print outImgL[i,j]
			except IndexError:
				pass

