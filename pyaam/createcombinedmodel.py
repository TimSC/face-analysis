
import pcaappearance, pcashape, pickle, pcacombined, random
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

	shapeModel = pickle.load(open("shapemodel.dat","rb"))
	appModel = pickle.load(open("appmodel.dat","rb"))

	shapePcaSpace = pickle.load(open("shapepcaspace.dat","rb"))
	appPcaShape = pickle.load(open("apppcaspace.dat","rb"))	

	print "Size of shape features", shapePcaSpace.shape
	print "Size of appearance features", appPcaShape.shape

	#print shapeModel.meanShape
	#plt.plot([pt[0] for pt in shapeModel.meanShape], [pt[1] for pt in shapeModel.meanShape])
	#plt.show()

	combModel = pcacombined.CreateCombinedModel(shapeModel,\
		appModel, shapePcaSpace, appPcaShape)

	#im = combModel.GenerateFace(np.random.rand((10)) * 4. - 2.)
	#im.show()

	pickle.dump(combModel, open("combinedmodel.dat","wb"), protocol =  pickle.HIGHEST_PROTOCOL)

