
import numpy as np
import matplotlib.pyplot as plt

def CalcMeanShape(posdata):
	#Calculates the mean shape for a range of ranges
	dataX, dataY = [], []
	for frameNum in posdata:
		framePos = posdata[frameNum]
		dataX.append([pos[0] for pos in framePos])
		dataY.append([pos[1] for pos in framePos])
	avX = np.array(dataX).mean(axis=0)
	avY = np.array(dataY).mean(axis=0)
	return zip(avX, avY)

def SubtractShape(posdata, subShape):
	out = {}
	for frameNum in posdata:
		framePos = posdata[frameNum]
		centFrame = []
		for pt, subPt in zip(framePos, subShape):
			centFrame.append((pt[0] - subPt[0], pt[1] - subPt[1]))
		out[frameNum] = centFrame
	return out

class ShapeModel:
	def __init__(self, meanShape, eigenShapes, variances):
		self.meanShape = meanShape
		self.eigenShapes = eigenShapes
		self.variances = variances

	def GenShape(self, shapeParam):
		test = np.power( self.eigenShapes, 2.)
		
		print self.variances[:10]
		shape = self.meanShape + self.eigenShapes[0,:] * shapeParam * self.variances[0]
		return shape

def CalcShapeModel(frameProc):

	#Convert shape on frames into 2D array	
	shape2D = []
	for frameNum in frameProc:
		framePos = frameProc[frameNum]
		frame1D = []
		for pt in framePos:
			frame1D.extend(pt)
		shape2D.append(frame1D)
	shapeArr = np.array(shape2D)

	#Prepare for PCA by subtracting the mean shape
	meanShape = shapeArr.mean(axis=0)
	centreShape = shapeArr - meanShape

	#Perform SVD
	u, s, v = np.linalg.svd(centreShape)
	print u.shape
	print s.shape
	print v.shape

	#plt.plot(s)
	#plt.show()

	

	shapeModel = ShapeModel(meanShape, v, s)
	return shapeModel

