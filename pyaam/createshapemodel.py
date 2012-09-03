import picseqloader, procrustes, pcashape, pickle
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
	(pics, posData) = picseqloader.LoadTimDatabase()

	#Convert to 3D matrix, first axis are frames, second axis selects the point, third axis is X or Y selector
	numPoints = len(posData[0])
	posdataArr = np.empty((len(posData), numPoints, 2))
	for countFrame, framePos in enumerate(posData):
		for ptNum, pt in enumerate(framePos):
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
	shapeModel, frameProcPcaSpace = pcashape.CalcShapeModel(frameProc)
	#a = shapeModel.GenShape(-0.1)
	#b = shapeModel.GenShape(0.)
	#c = shapeModel.GenShape(.1)

	#shapeModel.CalcTesselation()

	pickle.dump(shapeModel, open("shapemodel.dat","wb"), protocol =  pickle.HIGHEST_PROTOCOL)
	pickle.dump(frameProcPcaSpace, open("shapepcaspace.dat","wb"), protocol =  pickle.HIGHEST_PROTOCOL)

	#im = Image.open("/home/tim/dev/facedb/tim/cropped/100.jpg")
	#shapeModel.NormaliseFace(im, posdata[99])

	#plt.plot(a[:,0],-a[:,1])
	#plt.plot(b[:,0],-b[:,1])
	#plt.plot(c[:,0],-c[:,1])
	
	
	#plt.plot([p[0] for p in meanProcShape], [p[1] for p in meanProcShape],'.')

	#print meanScaleShape
	#plt.plot([p[0] for p in normScaleShape[0]], [p[1] for p in normScaleShape[0]])
	#plt.show()

	

