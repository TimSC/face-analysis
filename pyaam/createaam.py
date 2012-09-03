#!/usr/bin/env python
import pickle, picseqloader, pcacombined, random, time, shelve, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import multiprocessing

def GenerateTrainingSamples(processNum, numProcesses, posData, pics, combinedModel, perturbsOut, pixelSubset):

	countExamples = 0
	for frameCount, (framePos, im) in enumerate(zip(posData, pics)):
		#Distribute frames between the threads
		if (frameCount + processNum) % numProcesses != 0:
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

		offsetExamples = [-0.2, -0.1, -0.03, 0.03, 0.1, 0.2]
		#Perturb X
		for offsetExample in offsetExamples:
			print "frame=",frameCount,",process=",processNum

			#Perturb values
			changedVals = np.copy(vals)
			perturb = np.zeros(changedVals.shape)	
			perturb[0] = offsetExample * horizontalRange
			changedVals = changedVals + perturb

			diffVal = CalculateOffsetEffect(combinedModel, changedVals, im, shapefree, pixelSubset)
			perturbsOut[str(countExamples)] = (perturb, diffVal)
			countExamples += 1

			time.sleep(0.01)

		perturbsOut.sync()

		#Perturb Y
		for offsetExample in offsetExamples:
			print "frame=",frameCount,",process=",processNum

			#Perturb values
			changedVals = np.copy(vals)
			perturb = np.zeros(changedVals.shape)	
			perturb[1] = offsetExample * horizontalRange
			changedVals = changedVals + perturb

			diffVal = CalculateOffsetEffect(combinedModel, changedVals, im, shapefree, pixelSubset)
			perturbsOut[str(countExamples)] = (perturb, diffVal)
			countExamples += 1

			time.sleep(0.01)

		perturbsOut.sync()

		#Perturb Scale
		scaleExamples = [0.95, 0.97, 0.99, 1.01, 1.03, 1.05]
		for scaleExample in scaleExamples:
			print "frame=",frameCount,",process=",processNum

			#Perturb values
			changedVals = np.copy(vals)
			perturb = np.zeros(changedVals.shape)	
			perturb[2] = scaleExample * vals[2] #Scale by current value
			changedVals = changedVals + perturb

			diffVal = CalculateOffsetEffect(combinedModel, changedVals, im, shapefree, pixelSubset)
			perturbsOut[str(countExamples)] = (perturb, diffVal)
			countExamples += 1

			time.sleep(0.01)

		perturbsOut.sync()

		#Perturb rotation
		rotationExamples = [-5, -3, -1, 1, 3, 5]
		for rotationExample in rotationExamples:
			print "frame=",frameCount,",process=",processNum

			#Perturb values
			changedVals = np.copy(vals)
			perturb = np.zeros(changedVals.shape)	
			perturb[3] = rotationExample
			changedVals = changedVals + perturb

			diffVal = CalculateOffsetEffect(combinedModel, changedVals, im, shapefree, pixelSubset)
			perturbsOut[str(countExamples)] = (perturb, diffVal)
			countExamples += 1

			time.sleep(0.01)		

		perturbsOut.sync()

		#Perturb combined model, for each feature
		perturbExamples = [-0.5, -0.25, 0.25, 0.5]
		numEigVals = int(round(len(vals) * 0.3))
		for featNum in range(4, 4+numEigVals):
			for perturbExample in perturbExamples:
				print "frame=",frameCount,"of",len(posData),",featNum=",featNum,"of",numEigVals,",process=",processNum

				#Perturb values
				changedVals = np.copy(vals)
				perturb = np.zeros(changedVals.shape)	
				perturb[featNum] = perturbExample
				changedVals = changedVals + perturb

				diffVal = CalculateOffsetEffect(combinedModel, changedVals, im, shapefree, pixelSubset)
				perturbsOut[str(countExamples)] = (perturb, diffVal)
				countExamples += 1

				time.sleep(0.01)	
			perturbsOut.sync()

	return None

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

	#Generate training data with multiple processors
	manager = multiprocessing.Manager()
	processes, perturbsBank, diffValsBank = [], [], []
	for count in range(numProcessors):
		perturbsOut = shelve.open("perterbs{0}.dat".format(count), protocol =  pickle.HIGHEST_PROTOCOL)
		p = multiprocessing.Process(target=GenerateTrainingSamples, args=(count, \
			numProcessors, posData, pics, combinedModel, perturbsOut, pixelSubset))
		p.start()
		processes.append(p)
		perturbsBank.append(perturbsOut)

	for p in processes:
		p.join()

	for per in perturbsBank:
		print len(per.keys())

	if 0:
		#Collect process results into final data structure
		perturbMerge = []
		for li in perturbsBank:
			perturbMerge.extend(li)

		diffValsMerge = []
		for li in diffValsBank:
			diffValsMerge.extend(li)

		#perturbMerge = np.array(perturbMerge)
		#diffValsMerge = np.array(diffValsMerge)

		#Save result
		pickle.dump(perturbMerge, open(perturboutFiNa,"wb"), protocol =  pickle.HIGHEST_PROTOCOL)
		pickle.dump(diffValsMerge, open(diffoutFiNa,"wb"), protocol =  pickle.HIGHEST_PROTOCOL)

		print "perturbMerge size",perturbMerge.shape
		print "diffValsMerge size",diffValsMerge.shape


	if 0:	
		perturbs = pickle.load(open(perturboutFiNa,"rb"))
		diffVals = pickle.load(open(diffoutFiNa,"rb"))

		perturbs = np.array(perturbs)
		diffVals = np.array(diffVals)

		leastSqFit = np.linalg.lstsq(diffVals[:80,:], perturbs[:80,:])

		predict = np.dot(diffVals[80:,:], leastSqFit[0])
	
		col = 3
		print np.corrcoef(perturbs[80:,col], predict[:,col])
		plt.plot(perturbs[80:,col], predict[:,col])
		plt.show()

	if 0:
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

