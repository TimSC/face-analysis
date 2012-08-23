import pcashape, pcaappearance, pickle, time
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

def Eval(eigVec, targetImg, appearanceModel):
	print eigVec
	im1 = appearanceModel.GenerateFace(eigVec[4:])
	im2 = shapeModel.GetNormFaceFromEigVec(test, eigVec[0:4])
	arr1 = np.asarray(im1)
	arr2 = np.asarray(im2)
	diff = arr1 - arr2
	print np.power(diff,1.).sum()
	#print np.abs(diff)
	return np.abs(diff).reshape(diff.size).sum()

if __name__ == "__main__":
	shapeModel = pickle.load(open("shapemodel.dat","rb"))
	appearanceModel = pickle.load(open("appmodel.dat","rb"))
	#test = Image.open("shapefree/99.png")
	test = Image.open("/home/tim/dev/facedb/tim/cropped/100.jpg")

	
	#im.show()
	#exit(0)

	#im = appearanceModel.GenerateFace([0.])
	#im.show()
	#DifferenceIm(im, test).show()
	initial = np.zeros((20,))
	initial[0] = 550.
	initial[1] = 570.
	initial[2] = 300.

	result = opt.fmin_powell(Eval, initial, args = (test, appearanceModel))

	print result
	print "Cost", Eval(result, test, appearanceModel).sum()

	#print "x", result.tolist()
	time.sleep(1.)

	im = appearanceModel.GenerateFace(result)
	im.show()

	diffIm = DifferenceIm(im, test)
	diffIm.show()

