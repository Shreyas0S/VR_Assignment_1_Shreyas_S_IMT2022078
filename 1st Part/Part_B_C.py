import cv2 as cv
import numpy as np

#Function that counts number of coins using contours and connected components
def countNumberOfCoins(img,original_img):
    retval,label= cv.connectedComponents(img,connectivity=4,ltype=cv.CV_32S)
    print("Connected Components:",retval)
    contours,_=cv.findContours(img,mode=cv.RETR_TREE,method=cv.CHAIN_APPROX_SIMPLE)
    img_contours=cv.drawContours(original_img,contours,-1,(255,0,0),3)
    return img_contours,contours
def main():
    #Read the image
    img=cv.imread("../Images/Coins.png")
    #Display Original image
    cv.imshow("Original",img)
    cv.waitKey(0)
    #Convert to Gray Scale
    gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #Segment the coins
    cv.imshow("Gray Scale",gray_img)
    cv.waitKey(0)
    threshold,segmented_image=cv.threshold(gray_img,127,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    #Opening operation for removing noise
    opening = cv.morphologyEx(segmented_image, cv.MORPH_OPEN, kernel)
    kernel = np.ones((7,7),np.uint8)
    #Closing operation for filling holes
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    cv.imshow("Closed Image",closing)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #Count number of coins using function
    img_contours,contours=countNumberOfCoins(closing,img)
    #Display the image with contours
    cv.imshow("Contours",img_contours)
    cv.waitKey(0)
    print("Number of Contours:",len(contours)) 
    print("Number of Coins:",len(contours))
    #Display each coin
    for i in range(len(contours)):
        mask=np.zeros_like(gray_img)
        cv.drawContours(mask,contours,i,255,-1)
        cv.imshow("Coins",cv.bitwise_and(img,img,mask=mask))
        #save the individual coins
        cv.imwrite(f"../Images/Coin_{i}.png",cv.bitwise_and(img,img,mask=mask))
        cv.waitKey(0)
    cv.destroyAllWindows()

if __name__=="__main__":
    main()