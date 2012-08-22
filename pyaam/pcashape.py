
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial

class ShapeModel:
	def __init__(self, meanShape, eigenShapes, variances):
		self.meanShape = meanShape
		self.eigenShapes = eigenShapes
		self.variances = variances

	def GenShape(self, shapeParam):
		numPoints = self.meanShape.shape[0]
		
		xVar = self.eigenShapes[0,:numPoints] * shapeParam * self.variances[0]
		yVar = self.eigenShapes[0,numPoints:] * shapeParam * self.variances[0]

		final = np.vstack((self.meanShape[:,0] + xVar, self.meanShape[:,1] + yVar)).transpose()

		#print self.variances[:10]
		return final

	def CalcTesselation(self):
		self.tess = spatial.Delaunay(self.meanShape)
		print self.tess.points
		print self.tess.vertices
		print spatial.Delaunay.find_simplex(self.tess, [(0.5,0.5)])

def CalcShapeModel(shapeArr):

	#Prepare for PCA by subtracting the mean shape
	meanShape = shapeArr.mean(axis=0)

	centreShape = shapeArr - meanShape

	flattenedArr = np.hstack((centreShape[:,:,0],centreShape[:,:,1]))

	#Perform SVD
	u, s, v = np.linalg.svd(flattenedArr)
	#print u.shape
	#print s.shape
	#print v.shape

	#plt.plot(s)
	#plt.show()

	shapeModel = ShapeModel(meanShape, v, s)
	return shapeModel

