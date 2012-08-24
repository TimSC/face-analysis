
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
		self.inTriangle, self.vertices = None, None
		self.sizeImage = (400, 400)

	def GenShape(self, shapeParam, stdDevScaled = True):
		#Generate shape from eigenvalues based on mean shape and eigenvectors
		numPoints = self.meanShape.shape[0]
		
		shapeX = self.meanShape[:,0]
		shapeY = self.meanShape[:,1]
		for row, val in enumerate(shapeParam):
			if stdDevScaled: stdDevScaling = (self.variances[row] ** 0.5) #Scale by standard deviations
			else: stdDevScaling = 1.
			shapeX += self.eigenShapes[row,:numPoints] * val * stdDevScaling
			shapeY += self.eigenShapes[row,numPoints:] * val * stdDevScaling

		final = np.vstack((shapeX, shapeY)).transpose()
		return final

	def CalcTesselation(self):
		tess = spatial.Delaunay(self.meanShape)
		self.vertices = tess.vertices
		#print len(tess.points)
		#print len(tess.vertices)

		#Determine which tesselation triangle contains each pixel in the shape norm image
		self.inTriangle = np.ones(self.sizeImage, dtype=np.int) * -1
		for i in range(self.sizeImage[0]):
			for j in range(self.sizeImage[1]):
				normSpaceCoord = (float(i)/self.sizeImage[0],float(j)/self.sizeImage[1])
				simp = tess.find_simplex([normSpaceCoord])
				self.inTriangle[i,j] = simp
		
		#Visualise tess mesh
		#for tri in tess.vertices:
		#	pos = tess.points[tri]
		#	plt.plot(pos[:,0], pos[:,1])
		#plt.show()

	def GetNormFaceFromEigVec(self, im, vec):
		shape = self.GenShape(vec[3:])
		scaledShape = (2. * shape - 1.) * vec[2] + (vec[0], vec[1])
		return self.NormaliseFace(im, scaledShape)

	def NormaliseFace(self, im, pos):
		if self.inTriangle is None: self.CalcTesselation()
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
					#print i,j,inImgCoord[0:2],im.size
					synthl[i,j] = tuple(map(int,np.round(GetBilinearPixel(im, iml, inImgCoord[0:2]))))
					#print synthl[i,j]
				except IndexError:
					pass

		#synth.show()

		#verts = tess.vertices[simp]
		#print verts
		#for v in verts:
		#	print tess.points[v]

		#print self.meanShape
		return synth

	def CopyShapeFreeFaceToImg(self, targetIm, faceIm, shape):

		pos = np.array(shape)
		targetIml = targetIm.load()
		faceIml = faceIm.load()
		shape = np.array(shape)

		#Split input shape into mesh
		tess = spatial.Delaunay(shape)

		#Find affine mapping from input positions to mean shape
		triAffines = []
		for i, tri in enumerate(tess.vertices):
			meanVertPos = np.hstack((self.meanShape[tri], np.ones((3,1)))).transpose()
			shapeVertPos = np.hstack((pos[tri,:], np.ones((3,1)))).transpose()
			affine = np.dot(meanVertPos, np.linalg.inv(shapeVertPos)) 
			triAffines.append(affine)

		#Calculate ROI in target image
		xmin, xmax = shape[:,0].min(), shape[:,0].max()
		ymin, ymax = shape[:,1].min(), shape[:,1].max()

		#Calculate pixel colours
		for i in range(int(xmin), int(xmax+1)):
			for j in range(int(ymin), int(ymax+1)):
				#normSpaceCoordX = (i - xmin) / (xmax - xmin)
				#normSpaceCoordY = (j - ymin) / (ymax - ymin)

				#Determine which tesselation triangle contains each pixel in the shape norm image
				simp = tess.find_simplex([i, j])
				affine = triAffines[simp]

				#Calculate position in the input image
				homogCoord = (i, j, 1.)
				normImgCoord = np.dot(affine, homogCoord)

				#Scale normalised coordinate by image size
				shapeFreeImgCoord = ((normImgCoord[0]+0.5)*faceIm.size[0], (normImgCoord[1]+0.5)*faceIm.size[1])

				#print i, j, simp, shapeFreeImgCoord

				try:
					targetIml[i,j] = tuple(map(int,np.round(GetBilinearPixel(faceIm, faceIml, shapeFreeImgCoord))))
				except IndexError:
					pass


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

