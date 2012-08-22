import readposdata, procrustes, pcashape
import numpy as np
import matplotlib.pyplot as plt

def GetNormalisedScaleShape(posdata):
	#Makes points range from zero to one in both x and y, for a range of frames
	out = {}

	for frameNum in posdata:
		framePos = posdata[frameNum]
		frameX = np.array([pos[0] for pos in framePos])
		frameY = np.array([pos[1] for pos in framePos])

		xmin, xmax = frameX.min(), frameX.max()
		ymin, ymax = frameY.min(), frameY.max()
		xrang = xmax - xmin
		yrang = ymax - ymin
		#Prevent divide by zero
		if xrang == 0.: xrang = 1.
		if yrang == 0.: yrang = 1.
		
		normX = (frameX - xmin) / xrang
		normY = (frameY - ymin) / yrang
		out[frameNum] = (zip(normX, normY))
	return out

if __name__ == "__main__":
	posdata = readposdata.ReadPosData(open("/home/tim/Desktop/facedb/tim/marks.dat"))
	idReflection = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 22, 21, 20, 19, 18, 25, 24,\
		 23, 14, 13, 12, 11, 10, 17, 16, 15, 36, 35, 34, 33, 32, 31, 30, 29,\
		 28, 27, 26, 41, 40, 39, 38, 37, 44, 43, 42, 57, 56, 55, 54, 53, 52,\
		 51, 50, 49, 48, 47, 46, 45]

	posdata2 = readposdata.ReadPosDataMirror(open("/home/tim/Desktop/facedb/tim/marks.dat"), idReflection)


	posdata.update(posdata2)

	#First calculate a reference shape to be used as the basis of procrustes analysis
	#I don't think this approach is particularly satisfactory but it should work
	#for simple cases
	normScaleShape = GetNormalisedScaleShape(posdata)
	meanScaleShape = pcashape.CalcMeanShape(normScaleShape)

	#Transform frames by procrustes analysis
	frameProc = procrustes.CalcProcrustes(posdata, meanScaleShape)
	
	#for frameNum in frameProc:
	#	framePos = frameProc[frameNum]
	#	plt.plot([pt[0] for pt in framePos], [-pt[1] for pt in framePos])
	#plt.show()

	#Perform PCA on shape
	shapeModel = pcashape.CalcShapeModel(frameProc)
	a = shapeModel.GenShape(-0.1)
	b = shapeModel.GenShape(0.)
	c = shapeModel.GenShape(.1)

	plt.plot(a[::2],-a[1::2])
	plt.plot(b[::2],-b[1::2])
	plt.plot(c[::2],-c[1::2])
	
	
	#plt.plot([p[0] for p in meanProcShape], [p[1] for p in meanProcShape],'.')

	#print meanScaleShape
	#plt.plot([p[0] for p in normScaleShape[0]], [p[1] for p in normScaleShape[0]])
	plt.show()

	

