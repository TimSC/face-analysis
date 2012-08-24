from PIL import Image
import numpy as np
import os, pcaappearance, pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
	path = "shapefree"
	fiNames = os.listdir(path)
	imageData = None

	#Load images into 2D numpy array by flattening each image to a row
	print "Loading images"
	for imNum, fiName in enumerate(fiNames):
		im = Image.open(path+"/"+fiName)

		#Convert to greyscale
		#im = im.convert("L")
		pix = np.asarray(im)

		if imageData is None:
			imageData = np.empty((len(fiNames), pix.size))
		imageData[imNum,:] = pix.reshape((pix.size,))

	print "Performing PCA analysis"
	appModel, appPcaSpace = pcaappearance.CalcApperanceModel(imageData, pix.shape)
	
	pickle.dump(appModel, open("appmodel.dat","wb"), protocol =  pickle.HIGHEST_PROTOCOL)
	pickle.dump(appPcaSpace, open("apppcaspace.dat","wb"), protocol =  pickle.HIGHEST_PROTOCOL)

	#print appModel.variances
	#plt.plot(appModel.variances)
	#plt.show()

	#for i in range(10):
	#	im = appModel.GetEigenface(i)
	#	im.save("eigenface"+str(i)+".png")
	
	#appModel.GetAverageFace().save("averageface.png")

