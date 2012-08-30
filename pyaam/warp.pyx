### cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from PIL import Image
import numpy as np
cimport numpy as np
import math

cdef GetBilinearPixel(np.ndarray[np.float32_t,ndim=3] imArr, float posX, float posY, np.ndarray[np.int32_t,ndim=1] out):
	cdef float modXf, modYf
	cdef int modXi, modYi, chan
	cdef float bl, br, tl, tr, b, t, pxf

	#Get integer and fractional parts of numbers
	modXi = int(posX)
	modYi = int(posY)
	modXf = posX - modXi
	modYf = posY - modYi

	#Get pixels in four corners
	for chan in range(imArr.shape[2]):
		bl = imArr[modYi, modXi, chan]
		br = imArr[modYi, modXi+1, chan]
		tl = imArr[modYi+1, modXi, chan]
		tr = imArr[modYi+1, modXi+1, chan]
	
		#Calculate interpolation
		b = modXf * br + (1. - modXf) * bl
		t = modXf * tr + (1. - modXf) * tl
		pxf = modYf * t + (1. - modYf) * b
		out[chan] = int(pxf+0.5) #Do fast rounding to integer

	return None #Helps with profiling view


def GetBilinearPixelSlow(inArr, pos):
	cdef np.ndarray[np.int32_t,ndim=1] out = np.empty((inArr.shape[2],), dtype=np.int32)
	GetBilinearPixel(inArr, pos[0], pos[1], out)
	return out

def Warp(inImg, inImgL, np.ndarray[np.uint8_t, ndim=3] outArr, np.ndarray[np.int_t, ndim=2] inTriangle, triAffines):
	cdef int i, j, height, width, tri, chan
	cdef float ifl, jfl, heightf, widthf, normSpaceCoordX, normSpaceCoordY

	cdef np.ndarray[np.float32_t, ndim=3] inArr = np.asarray(inImg, dtype=np.float32)
	cdef np.ndarray[np.int32_t, ndim=1] px = np.empty((inArr.shape[2],), dtype=np.int32)
	cdef np.ndarray[np.float32_t, ndim=1] homogCoord = np.ones((3,), dtype=np.float32)
	cdef np.ndarray[double, ndim=1] outImgCoord
	cdef np.ndarray[double, ndim=2] affine
	width = outArr.shape[1]
	height = outArr.shape[0]
	heightf = height
	widthf = width

	#Synthesis shape norm image		
	for i in range(width):
		for j in range(height):
			ifl = i
			jfl = j
			homogCoord[0] = ifl/widthf
			homogCoord[1] = jfl/heightf
			tri = inTriangle[i,j]
			if tri == -1: continue
			affine = triAffines[tri]
				
			#Calculate position in the input image
			outImgCoord = np.dot(affine, homogCoord)

			if outImgCoord[0] < 0 or outImgCoord[0] >= inArr.shape[1]:
				for chan in range(px.shape[0]): px[chan] = 0
				continue
			if outImgCoord[1] < 0 or outImgCoord[1] >= inArr.shape[0]:
				for chan in range(px.shape[0]): px[chan] = 0
				continue

			#Nearest neighbour
			#outImgL[i,j] = inImgL[int(round(inImgCoord[0])),int(round(inImgCoord[1]))]

			#Bilinear sampling
			#print i,j,outImgCoord[0:2],im.size
			GetBilinearPixel(inArr, outImgCoord[0], outImgCoord[1], px)
			for chan in range(px.shape[0]):
				outArr[j,i,chan] = px[chan]
			#print outImgL[i,j]

	return None