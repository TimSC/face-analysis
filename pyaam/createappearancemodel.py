from PIL import Image
import numpy as np
import os

if __name__ == "__main__":
	path = "shapefree"
	fiNames = os.listdir(path)
	total = None
	count = 0

	for fiName in fiNames:
		print fiName
		im = Image.open(path+"/"+fiName)
		pix = np.asarray(im)
	
		if total is None:
			total = np.zeros(pix.shape)
		total = total + pix
		count += 1

	print total.max()
	print total.min()

	total = total / count

	print total.max()
	print total.min()

	print pix.shape
	print total.shape
	total = np.array(total, dtype=np.uint8)

	print pix.dtype
	print total.dtype

	out = Image.fromarray(total)
	out.show()

