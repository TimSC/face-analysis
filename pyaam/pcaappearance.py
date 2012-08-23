import numpy as np
from PIL import Image

def LimitPixelIntensity(app):
	#Limit pixel intensity to range of 0 to 255
	negativePix = np.where(app < 0.)
	app[negativePix] = 0 #Sets negative pixels to zero
	saturatedPix = np.where(app > 255.)
	app[saturatedPix] = 255 #Sets negative pixels to zero

class AppearanceModel:
	def __init__(self, meanAppearance, eigenFaces, variances, imgShape):
		self.meanAppearance = meanAppearance
		self.eigenFaces = eigenFaces
		self.variances = variances
		self.imgShape = imgShape

	def GetAverageFace(self):
		#Format data into an image
		outIm = np.array(self.meanAppearance, dtype=np.uint8).reshape(self.imgShape)
		out = Image.fromarray(outIm)
		return out		

	def GetEigenface(self, num):
		img = self.eigenFaces[num,:]

		#Normalise pixel intensity
		img = img - img.min()
		img = img * (255. / img.max())

		#Format data into an image
		outIm = np.array(img, dtype=np.uint8).reshape(self.imgShape)
		out = Image.fromarray(outIm)
		return out

	def GenerateFace(self, eigenValues):
		#Construct face from average and Eigenfaces
		app = self.meanAppearance
		print self.eigenFaces.dtype
		app = app + (self.eigenFaces[0,:] * eigenValues[0])
		
		LimitPixelIntensity(app)

		#Format data into an image
		outIm = np.array(app, dtype=np.uint8).reshape(self.imgShape)
		out = Image.fromarray(outIm)
		return out		

def CalcApperanceModel(imageData, imgShape):
	
	#Zero centre the pixel data
	meanAppearance = imageData.mean(axis=0)
	imageDataCent = imageData - meanAppearance

	#Use M. Turk and A. Pentland trick (A^T T) from Eigenfaces (1991)
	#print imageDataCent.shape
	covm = np.dot(imageDataCent,imageDataCent.transpose())

	#Perform PCA on A^T T matrix
	u, s, v = np.linalg.svd(covm)
	#print s.shape
	#print v.shape

	#Construct eigenvectors
	eigenFaces = np.dot(v, imageDataCent)

	return AppearanceModel(meanAppearance, eigenFaces, s, imgShape)

