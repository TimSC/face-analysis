
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from PIL import Image
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

class ShapeModel:
	def __init__(self, meanShape, eigenShapes, variances):
		self.meanShape = meanShape
		self.eigenShapes = eigenShapes
		self.variances = variances
		self.tess, self.inTriangle, self.vertices = None, None, None
		self.sizeImage = (400, 400)

	def GenShape(self, shapeParam):
		numPoints = self.meanShape.shape[0]
		
		xVar = self.eigenShapes[0,:numPoints] * shapeParam * self.variances[0]
		yVar = self.eigenShapes[0,numPoints:] * shapeParam * self.variances[0]

		final = np.vstack((self.meanShape[:,0] + xVar, self.meanShape[:,1] + yVar)).transpose()

		#print self.variances[:10]
		return final

	def CalcTesselation(self):
		self.tess = spatial.Delaunay(self.meanShape)
		self.vertices = self.tess.vertices
		#print len(self.tess.points)
		#print len(self.tess.vertices)

		#Determine which tesselation triangle contains each pixel in the shape norm image
		self.inTriangle = np.ones(self.sizeImage, dtype=np.int) * -1
		for i in range(self.sizeImage[0]):
			for j in range(self.sizeImage[1]):
				normSpaceCoord = (float(i)/self.sizeImage[0],float(j)/self.sizeImage[1])
				simp = self.tess.find_simplex([normSpaceCoord])
				self.inTriangle[i,j] = simp
		
		#Visualise tess mesh
		#for tri in self.tess.vertices:
		#	pos = self.tess.points[tri]
		#	plt.plot(pos[:,0], pos[:,1])
		#plt.show()

	def NormaliseFace(self, im, pos):
		if self.tess is None or self.inTriangle is None: self.CalcTesselation()
		pos = np.array(pos)
		iml = im.load()
	
		#Find affine mapping from mean shape to input positions
		triAffines = []
		for i, tri in enumerate(self.vertices):
			meanVertPos = np.hstack((self.meanShape[tri], np.ones((3,1)))).transpose()
			inputVertPos = np.hstack((pos[tri,:], np.ones((3,1)))).transpose()
			affine = np.dot(inputVertPos, np.linalg.inv(meanVertPos)) 
			#print i, meanVertPos, np.dot(affine, meanVertPos)#, affine
			triAffines.append(affine)

		#Synthesis shape norm image		
		synth = Image.new("RGB",self.sizeImage)
		synthl = synth.load()
		for i in range(synth.size[0]):
			for j in range(synth.size[1]):
				normSpaceCoord = (float(i)/synth.size[0],float(j)/synth.size[1])
				tri = self.inTriangle[i,j]
				if tri == -1: continue
				affine = triAffines[tri]
				
				#Calculate position in the input image
				homogCoord = (normSpaceCoord[0], normSpaceCoord[1], 1.)
				inImgCoord = np.dot(affine, homogCoord)

				try:
					#Nearest neighbour
					#synthl[i,j] = iml[int(round(inImgCoord[0])),int(round(inImgCoord[1]))]

					#Bilinear sampling
					synthl[i,j] = tuple(map(int,np.round(GetBilinearPixel(im, iml, inImgCoord[0:2]))))
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

