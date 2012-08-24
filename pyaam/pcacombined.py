
import numpy as np

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

		img = self.appModel.GenerateFace(appValues, stdDevScaled = 0)
		#img = self.appModel.GetAverageFace()
		return img

def CreateCombinedModel(shapeModel, appModel, shapePcaSpace, appPcaShape):

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
