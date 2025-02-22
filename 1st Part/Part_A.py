import cv2 as cv

def callback(imput):
    pass
def main():
    #Read the image
    img=cv.imread("../Images/Coins.png")

    cv.imshow("Original",img)
    cv.waitKey(0)
    #using canny edge detector to detect coin edges
    winname='canny'
    cv.namedWindow(winname=winname)
    cv.createTrackbar('minThreshold',winname,0,255,callback)
    cv.createTrackbar('maxThreshold',winname,0,255,callback)
    while True:
        if cv.waitKey(1)== ord('q'):
            break
        #Keeping a track bar for min and max threshold for canny
        minThres=cv.getTrackbarPos('minThreshold',winname)
        maxThres=cv.getTrackbarPos('maxThreshold',winname)
        cannyEdge=cv.Canny(img,minThres,maxThres)
        #Displaying the detected coin edges
        cv.imshow(winname,cannyEdge)
    #Save the image
    cv.imwrite("../Images/Coin_Edges.png",cannyEdge)
    cv.waitKey(0)
    cv.destroyAllWindows()
if __name__=="__main__":
    main()