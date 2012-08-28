import math
import numpy as np
import scipy.optimize as opt

def FrameToArray(framePos):
	out = np.empty((len(framePos),2))
	for i, pos in enumerate(framePos):
		out[i,0] = pos[0]
		out[i,1] = pos[1]
	return out

def EvalRms(angIn, frameScaled, targetZeroCent):
	out = []
	ang = angIn[0]

	for p1, p2 in zip(frameScaled, targetZeroCent):
		p1rx = p1[0] * math.cos(ang) - p1[1] * math.sin(ang)
		p1ry = p1[0] * math.sin(ang) + p1[1] * math.cos(ang)

		out.append(((p1rx-p2[0])**2.+(p1ry-p2[1])**2.)**0.5) #Calculate distance
	return out

def CalcProcrustesOnFrame(frameArr, targetArr):
	#Perform Procrustes analysis

	#Calc centeroids
	params = []
	targetCent = np.array(targetArr).mean(axis=0)
	frameCent = np.array(frameArr).mean(axis=0)
	params.extend(frameCent)

	#Zero centre frame
	targetZeroCent = targetArr - targetCent
	frameZeroCent = frameArr - frameCent

	#Calculate RMS for distance of points to the origin
	targetSquaredDistance = np.power(targetZeroCent, 2.).sum(axis=1)
	targetRmsd = targetSquaredDistance.mean() ** 0.5
	frameSquaredDistance = np.power(frameZeroCent, 2.).sum(axis=1)
	frameRmsd = frameSquaredDistance.mean() ** 0.5

	#Scale the frame to match the target shape
	if frameRmsd > 0.:
		scaling = targetRmsd / frameRmsd
	else:
		scaling = 1.
	frameScaled = frameZeroCent * scaling
	params.append(1./scaling)

	#Find the rotation that minimises the RMS difference
	#Possibly improve this by using the Jacobian of the angle
	optRet = opt.leastsq(EvalRms, (0.), args=(frameScaled, targetZeroCent))
	ang = optRet[0][0]
	params.append(math.degrees(-ang))

	#Rotate frame points
	frameRot = []
	for pt in frameScaled:
		prx = pt[0] * math.cos(ang) - pt[1] * math.sin(ang)
		pry = pt[0] * math.sin(ang) + pt[1] * math.cos(ang)		
		frameRot.append((prx, pry))

	#Translate to target centeroid
	frameFinal = frameRot + targetCent

	return frameFinal, params

def CalcProcrustes(posdata, targetShape):
	out = np.empty(posdata.shape)
	for rowNum in range(posdata.shape[0]):
		out[rowNum,:,:], params = CalcProcrustesOnFrame(posdata[rowNum,:], targetShape)
	return out

