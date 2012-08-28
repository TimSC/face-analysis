import pickle, picseqloader, pcacombined
from PIL import Image
import numpy as np

if __name__ == "__main__":
	combinedModel = pickle.load(open("combinedmodel.dat","rb"))
	(pics, posData) = picseqloader.LoadTimDatabase()

	im = pics[0]
	framePos = posData[0]

	#Get shape free face
	shapefree = combinedModel.ImageToNormaliseFace(im, framePos)
	shapefreeArr = np.asarray(shapefree, dtype=np.float)

	#Convert normalised face and shape to combined model eigenvalues
	vals = combinedModel.NormalisedFaceAndShapeToEigenVec(shapefree, framePos)

	#Perturb values
	changedVals = np.copy(vals)
	changedVals[0] = changedVals[0] + 10

	#Reconstruct synthetic image
	synthApp, synthShape = combinedModel.EigenVecToNormFaceAndShape(changedVals)
	
	#Get observed face at perturbed position
	changedValNormImage = combinedModel.ImageToNormaliseFace(im, synthShape)
	changedValNormImage.show()

	#Calculate difference
	changedValNormImageArr = np.asarray(changedValNormImage, dtype=np.float)
	
	diff = changedValNormImageArr - shapefreeArr
	print diff.mean(), diff.shape

	diffIm = Image.new("RGB", shapefree.size)
	diffImL = diffIm.load()
	for i in range(diff.shape[0]):
		for j in range(diff.shape[1]):
			for k in range(diff.shape[2]):
				col = list(diffImL[j,i])
				col[k] = int(round(128.+diff[i,j,k]))
				if col[k] < 0: col[k] = 0
				if col[k] > 255: col[k] = 255
				diffImL[j,i] = tuple(col)
	diffIm.show()
