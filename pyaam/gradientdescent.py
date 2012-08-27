import pcacombined, pickle, time
from PIL import Image
import numpy as np
import scipy.optimize as opt

def DifferenceIm(im1, im2):
	arr1 = np.asarray(im1)
	arr2 = np.asarray(im2)
	diff = 180 + arr1 - arr2

	pcaappearance.LimitPixelIntensity(diff)

	#Format data into an image
	outIm = np.array(diff, dtype=np.uint8)
	return Image.fromarray(outIm)	

evalCount = 0

def Eval(eigVec, targetImg, combinedModel):
	global evalCount
	print evalCount, eigVec

	#Warp target image and get shape free face
	im2 = combinedModel.TransformImageToNormalisedFace(targetImg, eigVec)
	#im2.show()
	im2.save("warped"+str(evalCount)+".jpg")

	#Generate shape free appearance from eigenvector
	im1 = combinedModel.GenerateFace(eigVec[5:])
	#im1.show()
	im1.save("synth"+str(evalCount)+".jpg")

	#Calculate difference
	arr1 = np.asarray(im1)
	arr2 = np.asarray(im2)
	diff = np.array(arr1, dtype=np.float) - np.array(arr2, dtype=np.float)
	
	#Ignore areas in synthetic appearance that are black
	filteredDiff = []
	#diffImg = Image.new("RGB",im1.size)
	#test1 = Image.new("RGB",im1.size)
	#test2 = Image.new("RGB",im1.size)
	#diffImgL = diffImg.load()
	#test1L = test1.load()
	#test2L = test2.load()

	for i in range(arr1.shape[0]):
		for j in range(arr1.shape[1]):
			black = True
			for k in range(arr1.shape[2]):
				
				if arr1[i,j,k] != 0:
					black = False

			if not black:
				filteredDiff.append(diff[i,j,k])
				#for k in range(arr1.shape[2]):
				#	current = list(diffImgL[i,j])
				#	current[k] = int(round(abs(diff[i,j,k])))
				#	diffImgL[i,j] = tuple(current)

				#for k in range(arr1.shape[2]):
				#	current = list(test1L[i,j])
				#	current[k] = abs(arr1[i,j,k])
				#	test1L[i,j] = tuple(current)

				#for k in range(arr1.shape[2]):
				#	current = list(test2L[i,j])
				#	current[k] = abs(arr2[i,j,k])
				#	test2L[i,j] = tuple(current)

	#print test1L[120,120]
	#print test2L[120,120]
	#print diffImgL[120,120]

	#test1.show()
	#test2.show()
	#diffImg.show()
	filteredDiff = np.array(filteredDiff)

	evalCount += 1
	meanDiff = np.abs(filteredDiff).mean() #Take mean absolute difference
	print meanDiff
	return meanDiff

if __name__ == "__main__":
	combinedModel = pickle.load(open("combinedmodel.dat","rb"))
	#targetImg = Image.open("shapefree/99.png")
	targetImg = Image.open("/home/tim/dev/facedb/tim/cropped/100.jpg")

	#im = combinedModel.GenerateFace([0.])
	#im.show()

	numComponentsNormalisedFace = combinedModel.NumComponentsNormalisedFace()

	#DifferenceIm(im, test).show()
	initial = np.zeros((10,))
	initial[0] = 695. #Position
	initial[1] = 510.
	initial[2] = 240. #Scale
	initial[3] = 0. #Rotation

	#Fit model to target image by gradient descent
	result = opt.fmin_powell(Eval, initial, args = (targetImg, combinedModel))

	#Report result
	print result
	print "Cost", Eval(result, targetImg, combinedModel).mean()

	#print "x", result.tolist()
	time.sleep(1.)

	im = combinedModel.GenerateFace(result)
	im.show()

	diffIm = DifferenceIm(im, targetImg)
	diffIm.show()

