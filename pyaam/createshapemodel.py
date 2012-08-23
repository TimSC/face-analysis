import readposdata, procrustes, pcashape, pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def RescaleXYZeroToOne(posdataArr):
	#This operates in place on the array

	for rowNum in range(posdataArr.shape[0]):
		#Determine range
		xmin = posdataArr[rowNum,:,0].min()
		xmax = posdataArr[rowNum,:,0].max()
		ymin = posdataArr[rowNum,:,1].min()
		ymax = posdataArr[rowNum,:,1].max()
		xrang = xmax - xmin		
		yrang = ymax - ymin	

		#Prevent divide by zero
		if xrang == 0.: xrang = 1.
		if yrang == 0.: yrang = 1.

		#Rescale row
		posdataArr[rowNum,:,0] = (posdataArr[rowNum,:,0] - xmin) / xrang
		posdataArr[rowNum,:,1] = (posdataArr[rowNum,:,1] - ymin) / yrang


if __name__ == "__main__":
	posdata = readposdata.ReadPosData(open("/home/tim/dev/facedb/tim/marks.dat"))
	idReflection = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 22, 21, 20, 19, 18, 25, 24,\
		 23, 14, 13, 12, 11, 10, 17, 16, 15, 36, 35, 34, 33, 32, 31, 30, 29,\
		 28, 27, 26, 41, 40, 39, 38, 37, 44, 43, 42, 57, 56, 55, 54, 53, 52,\
		 51, 50, 49, 48, 47, 46, 45]

	posdata2 = readposdata.ReadPosDataMirror(open("/home/tim/dev/facedb/tim/marks.dat"), idReflection)
	posdata.update(posdata2)

	#Convert to 3D matrix, first axis are frames, second axis selects the point, third axis is X or Y selector
	numPoints = len(posdata[posdata.keys()[0]])
	posdataArr = np.empty((len(posdata), numPoints, 2))
	for countFrame, frameNum in enumerate(posdata):
		for ptNum, pt in enumerate(posdata[frameNum]):
			posdataArr[countFrame, ptNum, 0] = pt[0]
			posdataArr[countFrame, ptNum, 1] = pt[1]

	#First calculate a reference shape to be used as the basis of rescaling data to 0->1 range
	#I don't think this approach is particularly satisfactory but it should work
	#for simple cases
	RescaleXYZeroToOne(posdataArr)
	meanScaleShape = posdataArr.mean(axis=0)

	#Transform frames by procrustes analysis
	frameProc = procrustes.CalcProcrustes(posdataArr, meanScaleShape)
	
	#for frameNum in frameProc:
	#	framePos = frameProc[frameNum]
	#	plt.plot([pt[0] for pt in framePos], [-pt[1] for pt in framePos])
	#plt.show()

	#Perform PCA on shape
	shapeModel = pcashape.CalcShapeModel(frameProc)
	#a = shapeModel.GenShape(-0.1)
	#b = shapeModel.GenShape(0.)
	#c = shapeModel.GenShape(.1)

	shapeModel.CalcTesselation()

	pickle.dump(shapeModel, open("shapemodel.dat","wb"), protocol =  pickle.HIGHEST_PROTOCOL)

	#im = Image.open("/home/tim/dev/facedb/tim/cropped/100.jpg")
	#shapeModel.NormaliseFace(im, posdata[99])

	#plt.plot(a[:,0],-a[:,1])
	#plt.plot(b[:,0],-b[:,1])
	#plt.plot(c[:,0],-c[:,1])
	
	
	#plt.plot([p[0] for p in meanProcShape], [p[1] for p in meanProcShape],'.')

	#print meanScaleShape
	#plt.plot([p[0] for p in normScaleShape[0]], [p[1] for p in normScaleShape[0]])
	#plt.show()

	

