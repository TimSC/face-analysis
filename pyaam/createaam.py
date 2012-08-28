import pickle, picseqloader, pcacombined, random
from PIL import Image
import numpy as np


def CalculateOffsetEffect(combinedModel, vals, im, shapefree, pixelSubset):

	#Perturb values
	changedVals = np.copy(vals)
	perturb = np.zeros(changedVals.shape)	
	perturb[0] = random.randint(-50, 50)
	perturb[1] = random.randint(-50, 50)
	perturb[2] = random.randint(-50, 50)
	perturb[3] = random.randint(-10, 10)
	for i in range(4, perturb.shape[0]):
		perturb[i] = random.random() - 0.5
	changedVals = changedVals + perturb

	#Reconstruct synthetic image
	synthApp, synthShape = combinedModel.EigenVecToNormFaceAndShape(changedVals)
	
	#Get observed face at perturbed position
	changedValNormImage = combinedModel.ImageToNormaliseFace(im, synthShape)
	#changedValNormImage.show()

	#Calculate difference
	changedValNormImageArr = np.asarray(changedValNormImage, dtype=np.float)
	
	diff = changedValNormImageArr - shapefreeArr

	#diffIm = Image.new("RGB", shapefree.size)
	#diffImL = diffIm.load()
	#for i in range(diff.shape[0]):
	#	for j in range(diff.shape[1]):
	#		for k in range(diff.shape[2]):
	#			col = list(diffImL[j,i])
	#			col[k] = int(round(128.+diff[i,j,k]))
	#			if col[k] < 0: col[k] = 0
	#			if col[k] > 255: col[k] = 255
	#			diffImL[j,i] = tuple(col)
	#diffIm.show()

	diffVals = []
	for px in pixelSubset:
		#print px, diff[px[0],px[1],:]
		diffVals.extend(diff[px[0],px[1],:])

	return perturb, diffVals

if __name__ == "__main__":
	combinedModel = pickle.load(open("combinedmodel.dat","rb"))
	(pics, posData) = picseqloader.LoadTimDatabase()

	if 1:
		im = pics[0]
		framePos = posData[0]

		#Get shape free face
		shapefree = combinedModel.ImageToNormaliseFace(im, framePos)
		shapefreeArr = np.asarray(shapefree, dtype=np.float)

		#Convert normalised face and shape to combined model eigenvalues
		vals = combinedModel.NormalisedFaceAndShapeToEigenVec(shapefree, framePos)

		#Select a sample of pixels to base predictions	
		#I am unsure if this is part of the canonical AAM system
		pixList = []
		for i in range(shapefree.size[0]):
			for j in range(shapefree.size[1]):
				pixList.append((i,j))
		pixelSubset = random.sample(pixList, 100)

	if 1:
		perturbs = []
		diffVals = []
		for i in range(100):
			print i
			perturb, diffVal = CalculateOffsetEffect(combinedModel, vals, im, shapefree, pixelSubset)
			perturbs.append(perturb)
			diffVals.append(diffVal)
	
		pickle.dump(perturbs, open("perturbs.dat","wb"), protocol =  pickle.HIGHEST_PROTOCOL)
		pickle.dump(diffVals, open("diffVals.dat","wb"), protocol =  pickle.HIGHEST_PROTOCOL)

	if 1:	
		perturbs = pickle.load(open("perturbs.dat","rb"))
		diffVals = pickle.load(open("diffVals.dat","rb"))

		print len(perturbs)
		print len(diffVals)

