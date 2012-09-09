import pickle, pcacombined, picseqloader
import numpy as np
import matplotlib.pyplot as plt

def AamPredict(combinedModel, pixelSubset, predictors, im, changedVals):

	#Reconstruct synthetic image
	synthApp, synthShape = combinedModel.EigenVecToNormFaceAndShape(changedVals)

	#Get norm face from source, based on perturbed shape
	perturbSourceFace = combinedModel.ImageToNormaliseFace(im, synthShape)

	synthAppArr = np.asarray(synthApp, dtype=np.float)
	diff = synthAppArr - perturbSourceFace

	#Extract pixels from diff
	diffVals = []
	for px in pixelSubset:
		#print px, diff[px[0],px[1],:]
		diffVals.extend(diff[px[0],px[1],:])

	pred = predictors[0]
	predVal = pred.predict(diffVals)
	return predVal + changedVals

if __name__ == "__main__":
	perturboutFiNa = "perturbs.dat"
	diffoutFiNa = "diffVals.dat"

	#Load combined model, annotations and images
	combinedModel = pickle.load(open("combinedmodel.dat","rb"))
	(pics, posData) = picseqloader.LoadTimDatabase()

	#Load prediction model
	predictors = pickle.load(open("predictors.dat", "rb"))
	pixelSubset = pickle.load(open("pixelSubset.dat","rb"))

	#For each training image
	frameNum = 0
	framePos, im = posData[frameNum], pics[frameNum]

	#Get shape free face
	shapefree = combinedModel.ImageToNormaliseFace(im, framePos)

	#Convert normalised face and shape to combined model eigenvalues
	vals = combinedModel.NormalisedFaceAndShapeToEigenVec(shapefree, framePos)

	#Purturb
	horizonalSamples = [pt[0] for pt in framePos]
	horizontalRange = max(horizonalSamples) - min(horizonalSamples)
	offsetExample = -0.15

	x, y = [], []	
	for off in range(-200, 200):

		changedVals = np.copy(vals)
		perturb = np.zeros(changedVals.shape)
		perturb[0] = off#offsetExample * horizontalRange
		changedVals = changedVals + perturb

		#Reconstruct synthetic image
		synthApp, synthShape = combinedModel.EigenVecToNormFaceAndShape(changedVals)

		predVal = AamPredict(combinedModel, pixelSubset, predictors, im, changedVals)
		print off, changedVals[0], predVal[0]
		x.append(changedVals[0])
		y.append(predVal[0])

	plt.plot(x,y)
	plt.show()

