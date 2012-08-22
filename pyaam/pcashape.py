
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from PIL import Image

class ShapeModel:
	def __init__(self, meanShape, eigenShapes, variances):
		self.meanShape = meanShape
		self.eigenShapes = eigenShapes
		self.variances = variances
		self.tess = None

	def GenShape(self, shapeParam):
		numPoints = self.meanShape.shape[0]
		
		xVar = self.eigenShapes[0,:numPoints] * shapeParam * self.variances[0]
		yVar = self.eigenShapes[0,numPoints:] * shapeParam * self.variances[0]

		final = np.vstack((self.meanShape[:,0] + xVar, self.meanShape[:,1] + yVar)).transpose()

		#print self.variances[:10]
		return final

	def CalcTesselation(self):
		self.tess = spatial.Delaunay(self.meanShape)
		#print len(self.tess.points)
		#print len(self.tess.vertices)

	def NormaliseFace(self, im, pos, targetImageSize):
		if self.tess is None: self.CalcTesselation()
		pos = np.array(pos)
		iml = im.load()
	
		#Find affine mapping from mean shape to input positions
		triAffines = []
		for tri in self.tess.vertices:
			meanVertPos = np.hstack((self.tess.points[tri], np.ones((3,1))))
			inputVertPos = np.hstack((pos[tri,:], np.ones((3,1))))
			affine = np.dot(inputVertPos, np.linalg.inv(meanVertPos)) 
			triAffines.append(affine)

		#Determine which tesselation triangle contains each pixel in the shape norm image
		inTriangle = np.ones((targetImageSize), dtype=np.int)
		for i in range(targetImageSize[0]):
			for j in range(targetImageSize[1]):
				normSpaceCoord = (float(i)/im.size[0],float(j)/im.size[1])
				simp = self.tess.find_simplex([normSpaceCoord])
				inTriangle[i,j] = simp
		
		#Synthesis shape norm image		
		synth = Image.new("RGB",targetImageSize)
		synthl = synth.load()
		for i in range(targetImageSize[0]):
			for j in range(targetImageSize[1]):
				normSpaceCoord = (float(i)/im.size[0],float(j)/im.size[1])
				tri = inTriangle[i,j]
				if tri == -1: continue
				affine = triAffines[tri]
				
				#Calculate position in the input image
				homogCoord = (normSpaceCoord[0], normSpaceCoord[1], 1.)
				inImgCoord = np.dot(affine, homogCoord)
				#print i, j
				#print homogCoord
				#print self.tess.points[self.tess.vertices[tri]]
				print inImgCoord
				try:
					synthl[i,j] = iml[int(round(inImgCoord[0])),int(round(inImgCoord[1]))]
				except IndexError:
					pass

		synth.show()

		#verts = self.tess.vertices[simp]
		#print verts
		#for v in verts:
		#	print self.tess.points[v]

		#print self.meanShape

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

