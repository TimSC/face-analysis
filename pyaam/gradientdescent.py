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

def Eval(eigVec, targetImg, combinedModel):
	print eigVec

	#Warp target image and get shape free face
	im2 = combinedModel.TransformImageToNormalisedFace(targetImg, eigVec)
	im2.show()

	#Generate shape free appearance from eigenvector
	im1 = combinedModel.GenerateFace(eigVec[5:])
	im1.show()

	#Calculate difference
	arr1 = np.asarray(im1)
	arr2 = np.asarray(im2)
	diff = arr1 - arr2
	print np.abs(diff).mean()
	return np.abs(diff).reshape(diff.size).mean()

if __name__ == "__main__":
	combinedModel = pickle.load(open("combinedmodel.dat","rb"))
	#targetImg = Image.open("shapefree/99.png")
	targetImg = Image.open("/home/tim/dev/facedb/tim/cropped/100.jpg")

	#im = combinedModel.GenerateFace([0.])
	#im.show()

	numComponentsNormalisedFace = combinedModel.NumComponentsNormalisedFace()

	#DifferenceIm(im, test).show()
	initial = np.zeros((10,))
	initial[0] = 550. #Horizontal position
	initial[1] = 570. #Vertical position
	initial[2] = 300. #Scale
	initial[2] = 0. #Rotation

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

