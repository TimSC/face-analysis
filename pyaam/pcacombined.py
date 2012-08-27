
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

	def GenerateFace(self, combinedVals):
	
		#Convert from combined PCA space to appearance and shape PCA space
		result = np.zeros((self.eigenVec.shape[1]))
		for row, val in enumerate(combinedVals):
			stdDevScaling = (self.variances[row] ** 0.5) #Scale by standard deviations
			result += self.eigenVec[row,:] * val * stdDevScaling

		shapeValues = result[:self.numShapeComp] / self.shapeScaleFactor
		appValues = result[self.numShapeComp:]

		shapeFreeImg = self.appModel.GenerateFace(appValues, stdDevScaled = False)
		#img = self.appModel.GetAverageFace()

		shape = self.shapeModel.GenShape(shapeValues, stdDevScaled = False)
		#plt.plot([pt[0] for pt in shape],[-pt[1] for pt in shape])
		#plt.show()

		targetIm = Image.new("RGB", shapeFreeImg.size)

		#Scale coordinates to keep the output image the same as the shape free image
		scaleShapeToImg = []
		for pt in shape:
			scaleShapeToImg.append(((pt[0]) * shapeFreeImg.size[0], (pt[1]) * shapeFreeImg.size[1]))

		#Transform the shape free image and paste into the target image
		self.shapeModel.CopyShapeFreeFaceToImg(targetIm, shapeFreeImg, scaleShapeToImg)

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
		scaledShape = (2. * shape - 1.) * combinedVals[2] + (combinedVals[1], combinedVals[0])
		return self.shapeModel.NormaliseFace(im, scaledShape)

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
