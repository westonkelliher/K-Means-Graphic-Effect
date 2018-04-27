from PIL import ImageEnhance
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import time


def posterize(imageFile, numColors=5, contrast=40, comic=3, save=False, show=True, verbose=False, dot=False): 
    img = Image.open(imageFile)
    pixels = img.load() # create the pixel map


    if verbose: print('formatting data for kmeans')
    numPixels = img.size[0]*img.size[1]
    X = np.resize(np.asarray(img), (numPixels ,3))
    

    if verbose: print('doing kmeans')
    kmeans = KMeans(n_clusters=numColors, random_state=0)
    kmeans.fit(X)
    #predictions = np.resize(kmeans.predict(X), (img.size[1], img.size[0]))

    
    if verbose: print('Adding pixel gradient to image')
    
    if dot:
        for i in range(img.size[0]):    # for every pixel:
            for j in range(img.size[1]):
                modLevel = ((i%6)/3 == (j%6)/3)*12*comic - 6*comic
                modLevel -= comic*((i%3==0)*4 + (i%3==1)*12 + (i%3==2)*6)
                modLevel -= comic*((j%3==0)*1 + (j%3==1)*8 + (j%3==2)*5)            
                modLevel += comic*12.0
                newColor = np.add([modLevel, modLevel, modLevel], pixels[i, j]).astype(int)
                pixels[i, j] = tuple(newColor)
    else:
        for i in range(img.size[0]):    # for every pixel:
            for j in range(img.size[1]):
                #modLevel = ((i%4)/2 == (j%4)/2)*16 + ((i%8)/4 == (j%8)/4)*8 - 12
                modLevel = ((i%4)/2 == (j%4)/2)*12*comic - 6*comic
                modLevel += (1-i%2)*(1-j%2)*4*comic
                modLevel -= (i%2)*(j%2)*4*comic
                modLevel += (i%2)*(1-j%2)*2*comic
                modLevel -= (1-i%2)*(j%2)*2*comic
                #modLevel += ((i%2)/1 == (j%2)/1)*8 - 4
                #modLevel = (i%3 == 1)*4 + (j%3 == 1)*4 - 4
                #modLevel += ((i%6)/3 == (j%6)/3)*6 - 2
                newColor = np.add([modLevel, modLevel, modLevel], pixels[i, j]).astype(int)
                pixels[i, j] = tuple(newColor)
    
    if verbose: print('formatting data for kmeans')
    numPixels = img.size[0]*img.size[1]
    X = np.resize(np.asarray(img), (numPixels ,3))
    

    if verbose: print('doing kmeans')
    #kmeans = KMeans(n_clusters=numColors, random_state=0)
    #kmeans.fit(X)
    predictions = np.resize(kmeans.predict(X), (img.size[1], img.size[0]))
    
    if verbose: print('contrasting clusters')
    average = [0, 0, 0]
    normAverage = average
    #darkest = [255, 255, 255]
    #lightest = [0, 0, 0]
    for k in kmeans.cluster_centers_:
        average += k
        #if np.average(
    average /= numColors
    for l in kmeans.cluster_centers_:
            dif = l-average
            push = dif/(np.linalg.norm(dif))
            l += push*max(contrast - np.linalg.norm(dif), contrast/3)
    
    
    if verbose: print('creating image')
    for i in range(img.size[0]):    # for every pixel:
        for j in range(img.size[1]):
            predictedCluster = predictions[j, i]
            predictedColor = kmeans.cluster_centers_[predictedCluster].astype(int)
            pixels[i, j] = tuple(predictedColor)

    converter = ImageEnhance.Color(img)
    img = converter.enhance(1.1)
    if show:
        img.show()
    if save:
        img.save(imageFile[:-3]+"_"+str(numColors)+"__"+str(contrast)+"_"+str(comic)+".jpg")



#MAIN:
print ("Use: place desired image in same directory as this program then follow the prompts below")
print ("Note: using images larger than ~500x500 will take considerable time")
print ("    : increasing number of colors will increase execution time")
print ()
pic = raw_input("Name of image (i.e. happyface.jpg):")
nc = input("Number of colors (generally 2-20):")
con = input("Contrast level (generally 0-200):")
cm = input("Comic level (generally 0-20):")
dt_ = raw_input("Dot? (y/n):")
sav_ = raw_input("Save image? (y/n):")
shw_ = raw_input("Show image? (y/n):")

if dt_ == "y":
    dt = True
else:
    dt = False

if sav_ == "y":
    sav = True
else:
    sav = False

if shw_ == "y":
    shw = True
else:
    shw = False

posterize(pic, numColors=nc, contrast=con, save=sav, comic=cm, show=shw, dot=dt)


