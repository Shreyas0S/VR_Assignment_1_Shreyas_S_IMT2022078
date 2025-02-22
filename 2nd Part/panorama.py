import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Create a blend filter for the image also known as distance transform
def create_blend_filter(img):
    h,w,_=img.shape
    filter=np.zeros((h,w),dtype=np.float32)
    for i in range(h):
        for j in range(w):
            filter[i][j]=float(min(i,j,h-i-1,w-j-1))
    filter=filter/np.max(filter)
    return filter

#Blend the images using the blend filter
def blending(img1,img2,filter1,filter2):
    result_img=np.zeros_like(img1)
    h,w,_=img1.shape
    for i in range(h):
        for j in range(w):
            if(filter1[i][j]+filter2[i][j]>1e-6):
                result_img[i][j]=(img1[i][j]*filter1[i][j]+img2[i][j]*filter2[i][j])/((filter1[i][j]+filter2[i][j]))
    return result_img

def create_panorama(img1,img2):
    #Using sift find interest points
    sift=cv.SIFT.create()
    keypoints1,descriptor1=sift.detectAndCompute(img1,None)
    keypoints2,descriptor2=sift.detectAndCompute(img2,None)
    #Use Flann based matcher to match the interest points
    FLANN_INDEX_KDTREE=1
    nKDtrees=5
    nLeafChecks = 50
    nNeighbors=2
    indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=nKDtrees)
    searchParams=dict(checks=nLeafChecks)
    flann=cv.FlannBasedMatcher(indexParams,searchParams)
    matches=flann.knnMatch(descriptor1,descriptor2,k=nNeighbors)

    goodMatches=[]
    #Find good matches
    for m,n in matches:
        if m.distance<0.7*n.distance:
            goodMatches.append(m)

    minGoodMatches=5
    panorama=None
    #using the matches find homography matrix
    if len(goodMatches)>minGoodMatches:
        scrPts=np.float32([keypoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1,1,2)
        dstPts=np.float32([keypoints2[m.trainIdx].pt for m in goodMatches]).reshape(-1,1,2)
        errorThreshold=5
        M,mask=cv.findHomography(dstPts,scrPts,cv.RANSAC,errorThreshold)

        h1,w1,_=img1.shape
        h2,w2,_=img2.shape
        #Warp the 1st image and combine it with the 2nd image to create the stitched image
        #Create blend filters for both the images img1_mask and panorama
        panorama=cv.warpPerspective(img2,M,(w1*17//12,h1))
        filter1=create_blend_filter(img2)
        filter2=cv.warpPerspective(filter1,M,(w1*17//12,h1))
        img1_mask=np.zeros_like(panorama)
        img1_mask[:h1,:w1]=img1
        filter1=create_blend_filter(img1)
        filter1_mask=np.zeros_like(filter2)
        filter1_mask[:h1,:w1]=filter1
        panorama=blending(img1_mask,panorama,filter1_mask,filter2)
    else:
        print("Not enough matches")
    return panorama

def main():
    #Read the images
    img1=cv.imread("../Images/Left.png")
    img2=cv.imread("../Images/Right.png")
    #Resize the images
    h1,w1,_= img1.shape
    h2,w2,_=img2.shape
    scale=1/2
    h1=int(h1*scale)
    w1=int(w1*scale*0.7)
    h2=int(h2*scale)
    w2=int(w2*scale*0.7)
    img1=cv.resize(img1,(h1,w1),interpolation=cv.INTER_LINEAR)
    img2=cv.resize(img2,(h2,w2),interpolation=cv.INTER_LINEAR)
    #Show the images
    cv.imshow("Image 1",img1)
    cv.imshow("Image 2",img2)
    cv.waitKey(0)
    panorama=create_panorama(img1,img2)
    #Show the panorama
    cv.imshow("Panorama",panorama)
    cv.waitKey(0)
    cv.imwrite("../Images/Panorama.png",panorama)
    cv.destroyAllWindows()
if __name__=="__main__":
    main()