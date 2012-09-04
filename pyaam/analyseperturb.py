#!/usr/bin/env python
import pickle, picseqloader, pcacombined, random, time, shelve, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import multiprocessing
import sklearn.linear_model as linear_model
import sklearn.ensemble as ensemble

def CalculateOffsetEffect(combinedModel, changedVals, im, shapefree, pixelSubset):

	#Reconstruct synthetic image
	synthApp, synthShape = combinedModel.EigenVecToNormFaceAndShape(changedVals)
	
	#Get observed face at perturbed position
	changedValNormImage = combinedModel.ImageToNormaliseFace(im, synthShape)
	#changedValNormImage.show()

	#Calculate difference
	changedValNormImageArr = np.asarray(changedValNormImage, dtype=np.float)
	
	shapefreeArr = np.asarray(shapefree, dtype=np.float)
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

	return diffVals

if __name__ == "__main__":

	perturboutFiNa = "perturbs.dat"
	diffoutFiNa = "diffVals.dat"

	#Process command line args
	parser = OptionParser()
	parser.add_option("-d", "--diffout", dest="diffout",
				          help="Output intensity difference data to filename")
	parser.add_option("-p", "--perturbout", dest="perturbout",
				          help="Output perturbation distance data to filename")
	(options, args) = parser.parse_args()
	if options.diffout is not None: diffoutFiNa = options.diffout
	if options.perturbout is not None: perturboutFiNa = options.perturbout

	#Load combined model, annotations and images
	combinedModel = pickle.load(open("combinedmodel.dat","rb"))
	(pics, posData) = picseqloader.LoadTimDatabase()

	#Delete existing output files
	numProcessors = multiprocessing.cpu_count()
	for count in range(numProcessors):
		fina = "perterbs{0}.dat".format(count)
		if os.path.exists(fina): os.unlink(fina)

	#Select a sample of pixels to base predictions	
	#I am unsure if this is part of the canonical AAM system
	pixList = []
	for i in range(combinedModel.appModel.imgShape[0]):
		for j in range(combinedModel.appModel.imgShape[1]):
			pixList.append((i,j))
	pixelSubset = random.sample(pixList, 300)

	countExamples = 0
	perturbsOut = []
	diffvalOut = []
	for frameCount, (framePos, im) in enumerate(zip(posData, pics)):
		
		if frameCount != 0:
			continue

		#Get shape free face
		shapefree = combinedModel.ImageToNormaliseFace(im, framePos)

		#Convert normalised face and shape to combined model eigenvalues
		vals = combinedModel.NormalisedFaceAndShapeToEigenVec(shapefree, framePos)

		horizonalSamples = [pt[0] for pt in framePos]
		horizontalRange = max(horizonalSamples) - min(horizonalSamples)

		#The parameters used in this section are taken from
		#Active Appearance Models: Theory, Extensions and Cases by Mikkel Bille Stegmann, 2000
		#http://www2.imm.dtu.dk/~aam/main/node16.html

		offsetExamples = np.arange(-1.5,1.5,0.01)
		np.random.shuffle(offsetExamples)
		#Perturb X
		for offsetExample in offsetExamples:
			print offsetExample

			#Perturb values
			changedVals = np.copy(vals)
			perturb = np.zeros(changedVals.shape)	
			perturb[0] = offsetExample * horizontalRange
			changedVals = changedVals + perturb

			diffVal = CalculateOffsetEffect(combinedModel, changedVals, im, shapefree, pixelSubset)
			perturbsOut.append(perturb)
			diffvalOut.append(diffVal)
			countExamples += 1

	diffvalOut = np.array(diffvalOut)
	perturbsOut = np.array(perturbsOut)

	#model = ensemble.GradientBoostingRegressor()
	numTrainSamples = int(round(len(offsetExamples)*0.8))
	model = linear_model.LinearRegression()
	model.fit(diffvalOut[:numTrainSamples,:], perturbsOut[:numTrainSamples,0])

	pred = model.predict(diffvalOut[numTrainSamples:,:])
	plt.plot(perturbsOut[numTrainSamples:,0], pred,'.')
	plt.show()

