import numpy as np
import pickle, pcacombined

if __name__ == "__main__":
	combModel = pickle.load(open("combinedmodel.dat","rb"))

	numEigenVals = int(combModel.eigenVec.shape[0] * 0.1)
	im = combModel.GenerateFace(np.random.rand((numEigenVals)) * 4. - 2.)

	print "Saving as randomface.png"
	im.save("randomface.png")

