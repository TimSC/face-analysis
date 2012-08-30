import numpy as np
import pickle, pcacombined

if __name__ == "__main__":
	combModel = pickle.load(open("combinedmodel.dat","rb"))

	numEigenVals = int(round(combModel.eigenVec.shape[0] * 0.1))
	extendedVals = np.concatenate(([200.,160.,400.,0.],np.random.rand((numEigenVals)) * 4. - 2.))
	#extendedVals = np.array([200.,150.,400.,0.])

	im = combModel.GenerateFace(extendedVals)

	im.show()

	print "Saving as randomface.png"
	im.save("randomface.png")

