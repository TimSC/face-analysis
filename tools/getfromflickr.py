import flickrapi, urlparse, urllib2, os
import xml.etree.ElementTree as ET

#Download pictures and metadata from flickr
#Depends on flickrapi from http://stuvel.eu/flickrapi

if __name__ == "__main__":
	
	api_key = 'f488e923e06006f4d3432075463dd587'
	baseDir = '/media/data/main/dev/facedb/flickr'

	photoIds = []
	flickrUrls = open("faces.txt")
	for li in flickrUrls:
		urlParts = urlparse.urlparse(li)
		pathSplit = urlParts.path.strip().split("/")
		if len(pathSplit) < 2: continue
		photoIds.append(pathSplit[-2])		

	print photoIds

	flickr = flickrapi.FlickrAPI(api_key)
	#info = flickr.photos_getInfo(photo_id='7867022578')

	#for photo in info:
		#print photo.keys()
		#for i in photo:	
			#print i.tag
			#print i.text
			#print ET.tostring(i)

	for photoId in photoIds:
		sizeQuery = flickr.photos_getSizes(photo_id=photoId)
	
		original = None
		for sizes in sizeQuery:
			for size in sizes:
				#print size.tag, size.attrib
				if size.attrib['label'] == "Original":
					original = size
		if original is None:
			print "Warning: could not find original for", photoId
			continue
		photoUrl = original.attrib['source']
		urlParts = urlparse.urlparse(photoUrl)
		
		destFiNa = urlParts.path.strip().split("/")[-1]
		destPath = baseDir+"/"+destFiNa
		print photoUrl, destFiNa
		if os.path.exists(destPath): continue
	
		photoData = urllib2.urlopen(photoUrl)
		photoDataRead = photoData.read()
		fi = open(destPath, "wb")
		fi.write(photoDataRead)
		fi.close()

