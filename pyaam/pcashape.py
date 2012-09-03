
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from PIL import Image
import math, warpcython

class ShapeModel:
	def __init__(self, meanShape, eigenShapes, variances):
		self.meanShape = meanShape
		self.eigenShapes = eigenShapes
		self.variances = variances
		self.inTriangle, self.vertices = None, None

	def GenShape(self, shapeParam, stdDevScaled = True):
		#Generate shape from eigenvalues based on mean shape and eigenvectors
		numPoints = self.meanShape.shape[0]
		
		shapeX = np.copy(self.meanShape[:,0])
		shapeY = np.copy(self.meanShape[:,1])
		for row, val in enumerate(shapeParam):
			if stdDevScaled: stdDevScaling = (self.variances[row] ** 0.5) #Scale by standard deviations
			else: stdDevScaling = 1.
			shapeX += self.eigenShapes[row,:numPoints] * val * stdDevScaling
			shapeY += self.eigenShapes[row,numPoints:] * val * stdDevScaling

		final = np.vstack((shapeX, shapeY)).transpose()
		return final

	def CalcTesselation(self, imSize):
		tess = spatial.Delaunay(self.meanShape)
		self.vertices = tess.vertices
		#print len(tess.points)
		#print len(tess.vertices)

		#Determine which tesselation triangle contains each pixel in the shape norm image
		self.inTriangle = np.ones(imSize, dtype=np.int) * -1
		for i in range(imSize[0]):
			for j in range(imSize[1]):
				normSpaceCoord = (float(i)/imSize[0],float(j)/imSize[1])
				simp = tess.find_simplex([normSpaceCoord])
				self.inTriangle[i,j] = simp
		
		#Visualise tess mesh
		#for tri in tess.vertices:
		#	pos = tess.points[tri]
		#	plt.plot(pos[:,0], pos[:,1])
		#plt.show()

	#def TransformImageToNormalisedFace(self, im, vec):
	#	#Normalise face based on (extended) eigenvector	with position, scale and rotation	
	#
	#	shape = self.GenShape(vec[4:])
	#	scaledShape = (2. * shape - 1.) * vec[2] + (vec[0], vec[1])
	#	return self.NormaliseFace(im, scaledShape)

	def NormaliseFace(self, im, pos, imSize):
		#Normalise face based on image position points

		if self.inTriangle is None: self.CalcTesselation(imSize)
		pos = np.array(pos)
	
		#Find affine mapping from mean shape to input positions
		triAffines = []
		scaledShape = np.array(zip([pt[0] * imSize[0] for pt in self.meanShape],[pt[1] * imSize[1] for pt in self.meanShape]))

		for i, tri in enumerate(self.vertices):
			meanVertPos = np.hstack((scaledShape[tri], np.ones((3,1)))).transpose()
			inputVertPos = np.hstack((pos[tri,:], np.ones((3,1)))).transpose()
			affine = np.dot(inputVertPos, np.linalg.inv(meanVertPos)) 
			#print i, meanVertPos, np.dot(affine, meanVertPos)#, affine
			triAffines.append(affine)

		#Synthesis shape norm image		
		imArr = np.asarray(im, dtype=np.float32)
		synthArr = np.zeros((imSize[1], imSize[0], len(im.mode)), dtype=np.uint8)
		warpcython.WarpProcessing(im, imArr, synthArr, self.inTriangle, triAffines, scaledShape)

		synth = Image.fromarray(synthArr)

		#synth.show()

		#verts = tess.vertices[simp]
		#print verts
		#for v in verts:
		#	print tess.points[v]

		#print self.meanShape
		return synth

	def CopyShapeFreeFaceToImg(self, targetIm, faceIm, shape):

		#print self.meanShape
		#plt.plot([pt[0] for pt in self.meanShape], [pt[1] for pt in self.meanShape])
		#plt.show()

		targetIml = targetIm.load()
		faceIml = faceIm.load()
		faceArr = np.asarray(faceIm, dtype=np.float32)
		shape = np.array(shape)

		#Split input shape into mesh
		tess = spatial.Delaunay(shape)

		#Calculate ROI in target image
		xmin, xmax = shape[:,0].min(), shape[:,0].max()
		ymin, ymax = shape[:,1].min(), shape[:,1].max()
		#print xmin, xmax, ymin, ymax

		#Determine which tesselation triangle contains each pixel in the shape norm image
		inTessTriangle = np.ones(targetIm.size, dtype=np.int) * -1
		for i in range(int(xmin), int(xmax+1.)):
			for j in range(int(ymin), int(ymax+1.)):
				if i < 0 or i >= inTessTriangle.shape[0]: continue
				if j < 0 or j >= inTessTriangle.shape[1]: continue
				normSpaceCoord = (float(i),float(j))
				simp = tess.find_simplex([normSpaceCoord])
				inTessTriangle[i,j] = simp

		#Find affine mapping from input positions to mean shape
		triAffines = []
		for i, tri in enumerate(tess.vertices):
			meanVertPos = np.hstack((self.meanShape[tri] * faceIm.size[0], np.ones((3,1)))).transpose()
			shapeVertPos = np.hstack((shape[tri,:], np.ones((3,1)))).transpose()
			#print meanVertPos
			#print shapeVertPos
			affine = np.dot(meanVertPos, np.linalg.inv(shapeVertPos)) 
			triAffines.append(affine)

		#Calculate pixel colours
		targetArr = np.copy(np.asarray(targetIm, dtype=np.uint8))
		warpcython.WarpProcessing(faceIm, faceArr, targetArr, inTessTriangle, triAffines, shape)
		targetIm.paste(Image.fromarray(targetArr))

		#Plot key points on target image
		#for pt in shape:
		#	try:
		#		targetIml[pt[0],pt[1]] = (255,255,255)
		#	except:
		#		pass

	def ShapeToEigenVec(self, shape):
		#Subtract mean shape
		centreShape = shape - self.meanShape

		#Convert to 1D array
		flattenedArr = np.hstack((centreShape[:,0],centreShape[:,1]))
		
		#Project shapes into PCA space
		shapePcaSpace = np.dot(self.eigenShapes, flattenedArr.transpose()).transpose()		

		return shapePcaSpace

def CalcShapeModel(shapeArr):

	#Prepare for PCA by subtracting the mean shape
	meanShape = shapeArr.mean(axis=0)

	centreShape = shapeArr - meanShape

	flattenedArr = np.hstack((centreShape[:,:,0],centreShape[:,:,1]))

	#Perform SVD
	u, s, v = np.linalg.svd(flattenedArr)

	#Project shapes into PCA space
	shapePcaSpace = np.dot(v, flattenedArr.transpose()).transpose()

	#Calculate variance of variation mode
	variance = shapePcaSpace.var(axis=0)

	shapeModel = ShapeModel(meanShape, v, variance)
	return shapeModel, shapePcaSpace

