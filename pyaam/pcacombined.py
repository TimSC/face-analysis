
import numpy as np

class CombinedModel:
	def __init__(self):
		pass
		#self.shapeModel = shapeModel
		#self.appModel = appModel

def CreateCombinedModel():

	#Scale shape to match the appearance variance
	shapeVar = shapePcaSpace.var()
	appVar = appPcaShape.var()

	print "Shape variance", shapeVar
	print "Appearance variance", appVar

	shapeScaleFactor = (appVar / shapeVar) ** 0.5
	shapeScaled = shapePcaSpace * shapeScaleFactor

	#Combined shape and appearance into single array
	comb = np.hstack((shapeScaled, appVar))

	#Perform SVD
	u, s, v = np.linalg.svd(comb)

	#Project shapes into PCA space
	combPcaSpace = np.dot(v, comb.transpose())

	#Calculate variance of variation mode
	variance = combPcaSpace.var(axis=1)

	combModel = CombinedModel()

	return combModel
