import math, procrustes
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class CombinedModel:
	def __init__(self, shapeModel, appModel, eigenVec, variances, numShapeComp, numAppComp, shapeScaleFactor):
		self.shapeModel = shapeModel
		self.appModel = appModel
		self.eigenVec = eigenVec
		self.variances = variances
		self.numShapeComp = numShapeComp
		self.numAppComp = numAppComp
		self.shapeScaleFactor = shapeScaleFactor

	def NumComponentsCompact(self):
		return self.eigenVec.shape[1]

	def NumComponentsExtended(self):
		return self.NumComponentsCompact() + 4

	def GenerateFace(self, combinedVals):
	
		#Convert from combined PCA space to appearance and shape PCA space
		shapeFreeImg, shape = self.EigenVecToNormFaceAndShape(combinedVals)

		#plt.plot([pt[0] for pt in shape],[-pt[1] for pt in shape])
		#plt.show()

		targetIm = Image.new("RGB", shapeFreeImg.size)

		#Transform the shape free image and paste into the target image
		self.shapeModel.CopyShapeFreeFaceToImg(targetIm, shapeFreeImg, shape)

		return targetIm

	def NumComponentsNormalisedFace(self):
		return self.numShapeComp + self.numAppComp

	def TransformImageToNormalisedFace(self, im, combinedVals):
		#Normalise face based on (extended) eigenvector	with position, scale and rotation	

		#Convert from combined PCA space to appearance and shape PCA space
		result = np.zeros((self.eigenVec.shape[1]))
		for row, val in enumerate(combinedVals[4:]):
			stdDevScaling = (self.variances[row] ** 0.5) #Scale by standard deviations
			result += self.eigenVec[row,:] * val * stdDevScaling

		shapeValues = result[:self.numShapeComp] / self.shapeScaleFactor
		appValues = result[self.numShapeComp:]

		shape = self.shapeModel.GenShape(shapeValues)
		scaledShape = (shape - 0.5) * combinedVals[2] 

		rotatedShape = []
		for pt in scaledShape:
			angr = math.radians(combinedVals[3])
			rotatedShape.append((pt[0]*math.cos(angr)-pt[1]*math.sin(angr), pt[0]*math.sin(angr)+pt[1]*math.cos(angr)))
		translatedShape = np.array(rotatedShape) + (combinedVals[0], combinedVals[1])
		return self.shapeModel.NormaliseFace(im, translatedShape)

	def ImageToNormaliseFace(self, im, pos):
		#Return a shape free face based on annotated positions
		return self.shapeModel.NormaliseFace(im, pos, (self.appModel.imgShape[0], self.appModel.imgShape[1]))

	def NormalisedFaceAndShapeToEigenVec(self, normFace, shape):
		
		#Do procrustes on input shape
		procShape, procParams = procrustes.CalcProcrustesOnFrame(shape, self.shapeModel.meanShape)

		#Appearance to PCA space
		appearanceVals = self.appModel.NormalisedFaceToEigenVals(normFace)

		#Shape to PCA space
		shapeVals = self.shapeModel.ShapeToEigenVec(procShape)

		#Scale shape values
		shapeValsScaled = shapeVals * self.shapeScaleFactor

		#Convert to combined vector
		comb = np.hstack((shapeValsScaled, appearanceVals))
		
		#Project shapes into combined PCA space
		combPcaSpace = np.dot(self.eigenVec, comb.transpose()).transpose()

		#Scale by variance
		scaleComb = []
		for i, val in enumerate(combPcaSpace):
			scaleComb.append(val / (self.variances[i] ** 0.5))

		return np.concatenate((procParams, scaleComb))

	def EigenVecToNormFaceAndShape(self, extendedVals):

		procParams = extendedVals[:4]
		combinedVals = extendedVals[4:]

		#Convert from combined PCA space to appearance and shape PCA space
		result = np.zeros((self.eigenVec.shape[1]))
		for row, val in enumerate(combinedVals):
			stdDevScaling = (self.variances[row] ** 0.5) #Scale by standard deviations
			result += self.eigenVec[row,:] * val * stdDevScaling

		shapeValues = result[:self.numShapeComp] / self.shapeScaleFactor
		appValues = result[self.numShapeComp:]

		#Reconstruct appearance
		synthApp = self.appModel.GenerateFace(appValues, False)

		#Reconstruct shape
		synthShape = self.shapeModel.GenShape(shapeValues, False)

		#Transform shape based on procrustes parameters
		#Zero centre
		synthShapeCent = synthShape - synthShape.mean(axis=0)

		#Scale shape
		scaledShape = synthShapeCent * procParams[2]

		#Rotate shape
		rotatedShape = []
		for pt in scaledShape:
			angr = math.radians(procParams[3])
			rotatedShape.append((pt[0]*math.cos(angr)-pt[1]*math.sin(angr), pt[0]*math.sin(angr)+pt[1]*math.cos(angr)))

		#Translate shape
		finalShape = np.array(rotatedShape) + (procParams[0], procParams[1])

		return synthApp, finalShape

	def CopyShapeFreeFaceToImg(self, targetIm, faceIm, shape):
		return self.shapeModel.CopyShapeFreeFaceToImg(targetIm, faceIm, shape)

def CreateCombinedModel(shapeModel, appModel, shapePcaSpace, appPcaShape):

	#From Statistical Models of Face Images - Improving Specificity
	#by G.J. Edwards, A. Lanitis, C.J. Taylor, T. F. Cootes

	#Scale shape to match the appearance variance
	shapeVar = shapePcaSpace.var()
	appVar = appPcaShape.var()

	print "Shape variance", shapeVar
	print "Appearance variance", appVar

	shapeScaleFactor = (appVar / shapeVar) ** 0.5
	shapeScaled = shapePcaSpace * shapeScaleFactor

	#Combined shape and appearance into single array
	comb = np.hstack((shapeScaled, appPcaShape))

	#Skip subtracting the mean at this stage, as the data should already be zero centred

	#Perform SVD
	u, s, v = np.linalg.svd(comb)

	#Project shapes into PCA space
	combPcaSpace = np.dot(v, comb.transpose()).transpose()

	#Calculate variance of variation mode
	variances = combPcaSpace.var(axis=0)
	combModel = CombinedModel(shapeModel, appModel, v, variances, shapeScaled.shape[1], appPcaShape.shape[1], shapeScaleFactor)

	return combModel
