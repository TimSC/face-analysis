import pickle, pcacombined, picseqloader
from PIL import Image

if __name__ == "__main__":

	combinedModel = pickle.load(open("combinedmodel.dat","rb"))
	(pics, posData) = picseqloader.LoadTimDatabase()

	im = pics[0]
	framePos = posData[0]

	#Get shape free face
	shapefree = combinedModel.ImageToNormaliseFace(im, framePos)

	#Convert normalised face and shape to combined model eigenvalues
	vals = combinedModel.NormalisedFaceAndShapeToEigenVec(shapefree, framePos)

	#vals2 = []
	#for i, v in enumerate(vals):
	#	if i < 20:
	#		vals2.append(v)
	#	else:
	#		vals2.append(0.)

	#Reconstruct face
	synthApp, synthShape = combinedModel.EigenVecToNormFaceAndShape(vals)
	#synthApp.show()

	#Paint synthetic face back on to original image
	combinedModel.CopyShapeFreeFaceToImg(im, synthApp, synthShape)
	im.show()

