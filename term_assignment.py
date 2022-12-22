import cv2
import numpy as np
from matplotlib import pyplot as plt

def display(image):
    cv2.imshow('image',image)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path='./img1.jpeg'
image=cv2.imread(image_path)  #loaded an image(reading)
#-------------------------------------displaying-------------------------------------
# cv2.imshow('hayan',image)    
# cv2.waitKey(0)  waitkey() 
# (waitKey is function of Python OpenCV allows users to display a window for given
#  milliseconds or until any key is pressed. It takes time in milliseconds
#  as a parameter and waits for the given time to destroy the window, if 0
#  is passed in the argument it waits till any key is pressed.
# cv2.destroyAllWindows())

# # saving
# directory=r'\home\abdulhayan\Documents\7th-samester\CVIP\TEXST'
# filename='savedImage.jpeg' 
# cv2.imwrite(filename,image)
# print('successfully saved')

#-------------------------------------playing with image-------------------------------------
print(image.shape)  #it will resolution of image

#-------------------------------------changing colorspace from RGB to GrayScale-------------------------------------
# grey_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('hayan',image)    
# cv2.imshow('gray_hayan',grey_image)    
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-------------------------------------resize an image-------------------------------------
# resized_iamge=cv2.resize(image,(800,800))
# cv2.imshow('hayan',image)    
# cv2.imshow('changed-image',resized_iamge)    
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-------------------------------------putting text on an image-------------------------------------
# text='Abdul Hayan'
# coordinates=(300,300)
# font=cv2.FONT_HERSHEY_SIMPLEX
# thickness=4
# font_scale=2
# font_color=(255,0,0)
# text_image = cv2.putText(image, text,coordinates,font, font_scale,font_color,thickness)
# display(text_image)

#-------------------------------------drawing a line on an image-------------------------------------
# s_coordinates=(400,300)
# e_coordinates=(700,600)
# font_color=(255,0,0)
# thickness=4
# text_image = cv2.line(image, s_coordinates,e_coordinates,font_color,thickness)
# display(text_image)


#-------------------------------------drawing a circle on an image-------------------------------------
# center_coordinates=(400,300)
# color=(255,0,0)
# radius=16
# tichkeness=6
# image = cv2.circle(image, center_coordinates,radius,color, tichkeness)
# display(image)

#-------------------------------------drawing a ellipse on an image-------------------------------------
# center_coordinates=(400,300)
# axis_length=(100,50)
# angle=30
# start_angle=0
# end_angle=360
# color=(255,0,0)
# thickness=6
# image = cv2.ellipse(image, center_coordinates,axis_length,angle,start_angle,end_angle,color, thickness
# )
# display(image)

# -------------------------------------drawing a rectangle on an image-------------------------------------
# s_coordinates=(400,300)
# e_coordinates=(700,600)
# font_color=(255,0,0)
# thickness=4
# text_image = cv2.rectangle(image, s_coordinates,e_coordinates,font_color,thickness)
# display(text_image)


# -------------------------------------different modes of color-------------------------------------
# image_path='./img1.jpeg'
# test_image=cv2.imread(image_path,0) #image will be grayscale
# display(test_image)

# image_path='./img1.jpeg'
# test_image=cv2.imread(image_path,1) #image will be RGB if it is grayscale originally
# display(test_image)

# image_path='./img1.jpeg'
# test_image=cv2.imread(image_path,-1) #image will remain as it is.
# display(test_image)

# -------------------------------------capture video using attached camera-------------------------------------
# 0 means camera will start recording
# abc=cv2.VideoCapture(0)
# to capture all the frames, we used loop. 25 per second is frame capturing speed
# while(True):
#     ret, frame = abc.read()    # read all the frames one by one
#     cv2.imshow('abc', frame)  # displaying all the frames one by one
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#         abc.release()
# cv2.destroyAllWindows()

# -------------------------------------storing the captured video-------------------------------------
# cap=cv2.VideoCapture(0)
# if(cap.isOpened() == False):
#     print('Camera is not opened')

# # setting resolution
# f_widtgh = int(cap.get(3))
# f_height = int(cap.get(4))

# # codec is used for removing unncessary details from video
# vodeo_cod = cv2.VideoWriter_fourcc(*'XVID')   #setting encodec
# video_output = cv2.VideoWriter('Captured_video.MP4',vodeo_cod, 30, (f_widtgh, f_height))  #creating video writer object for wrting video on disk (30 is number of frames per second)

# while(True):
#     # cap.read() methods returns a tuple, first element is a bool 
#     # and the second is frame
 
#     ret, frame = cap.read()
#     if ret == True:
#            # Write the frame to the output files
#            video_output.write(frame)
#            cv2.imshow('frame', frame)
#            if cv2.waitKey(1) & 0xFF == ord('q'):
#                break  
#     else:
#          print("Stream disconnected")
#          break
# cap.release()
# video_output.release()
# cv2.destroyAllWindows()

# -------------------------------------Read Video from disk-------------------------------------
# vid_capture = cv2.VideoCapture('./Captured_video.MP4')

# # getting metadata
# if (vid_capture.isOpened() == False):
#   print("Error opening the video file")
# else:
#   # Get frame rate information
 
#   fps = int(vid_capture.get(5))
#   print("Frame Rate : ",fps,"frames per second")  
 
#   # Get frame count
#   frame_count = vid_capture.get(7)
#   print("Frame count : ", frame_count)

# # playing video
# while(vid_capture.isOpened()):
#   # vCapture.read() methods returns a tuple, first element is a bool 
#   # and the second is frame
 
#   ret, frame = vid_capture.read()
#   if ret == True:
#     cv2.imshow('Frame',frame)
#     k = cv2.waitKey(50)
#     # 113 is ASCII code for q key
#     if k == 113:
#       break
#   else:
#     break

# # Release the objects
# vid_capture.release()
# cv2.destroyAllWindows()

# -------------------------------------blurring with averaging-------------------------------------
# kernal_size=(3,3)
# blurImg = cv2.blur(image,kernal_size) 
# display(blurImg)
# other blurring are Gaussian Blurring, Median Blurring, Bilateral Filtering

# -------------------------------------cropping an image-------------------------------------
# cropped_image=image[580:1280, 350:930]
# display(cropped_image)
# To slice an array, you need to specify the start and end index of the first as well as the second dimension. 
#(580:1280) The first dimension is always the number of rows or the height of the image.
#( 350:930) The second dimension is the number of columns or the width of the image. 

# -------------------------------------edge detection of an image-------------------------------------
#  # Read the original image as grayscale
# img = cv2.imread('./img1.jpeg',flags=0)  
# # Blur the image for better edge detection
# img_blur = cv2.GaussianBlur(img,(3,3), 0) 
# sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
# sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
# sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
 
# # Display Sobel Edge Detection Images
# # cv2.imshow('Sobel X', sobelx)
# # cv2.waitKey(0)
 
# # cv2.imshow('Sobel Y', sobely)
# # cv2.waitKey(0)
 
# cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# -------------------------------------contour detection of an image-------------------------------------
# # Using contour detection, we can detect the borders of objects, and localize them easily in an image
# img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # apply binary thresholding
# ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

# # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
# contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                      
# # draw contours on the original image
# image_copy = image.copy()
# cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
# # see the results
# cv2.imshow('None approximation', image_copy)
# cv2.waitKey(0)
# cv2.imwrite('contours_none_image1.jpg', image_copy)
# cv2.destroyAllWindows()

# -------------------------------------sharpening of an image(filter2D is used for sharpening an image)-------------------------------------
# # load the image into system memory
# image = cv2.imread('./img1.jpeg', flags=cv2.IMREAD_COLOR)
# # display the image to the screen
# cv2.imshow('AV CV- Winter Wonder', image)
## following kernal is used for sharpening of image
# kernel = np.array([[0, -1, 0],
#                    [-1, 5,-1],
#                    [0, -1, 0]])
# image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel) #ddepth is an integer value representing the expected depth of the output image. We will set this value to be equal to -1. In doing this, we tell the compiler that the output image will have the same depth as the input image.
# display(image_sharp)

# -------------------------------------filter on an  image-------------------------------------
# # load the image into system memory
# image = cv2.imread('./img1.jpeg', flags=cv2.IMREAD_COLOR)
# # display the image to the screen
# cv2.imshow('AV CV- Winter Wonder', image)
# # following kernal is used for sharpening of image
# kernel1 = np.array([[0, 0, 0],
#                     [0, 1, 0],
#                     [0, 0, 0]])
# result_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1) #ddepth is an integer value representing the expected depth of the output image. We will set this value to be equal to -1. In doing this, we tell the compiler that the output image will have the same depth as the input image.
# display(result_image)

# -------------------------------------GaussianBlur-------------------------------------
# # display the image to the screen
# cv2.imshow('AV CV- Winter Wonder', image)
# # following kernal is used for sharpening of image

# result_image = cv2.GaussianBlur(src=image, ksize=(5,5), sigmaX=0, sigmaY=0)
# display(result_image)

# -------------------------------------medianBlur-------------------------------------
# # display the image to the screen
# cv2.imshow('AV CV- Winter Wonder', image)
# # following kernal is used for sharpening of image

# result_image = cv2.medianBlur(src=image, ksize=5)
# display(result_image)

# -------------------------------------Histogram-------------------------------------
# see from: https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
# img = cv2.imread('home.jpg')
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([image],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256]) #to set limit of current axis
# plt.show()

# -------------------------------------Errosion-------------------------------------

def errosion(image):
    # Creating kernel
    kernel = np.ones((5, 5), np.uint8)
    
    # Using cv2.erode() method 
    image = cv2.erode(image, kernel) 
    display(image)

# -------------------------------------display image in full window-------------------------------------
def fullWindow(image):
    # Create a window in full screen mode
 
    cv2.namedWindow("fullscreen", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("fullscreen", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("fullscreen", image)
    cv2.waitKey()

# -------------------------------------BilateralFilter-------------------------------------
# The bilateral filter is a non-linear smoothing filter in OpenCV that reduces the noise in an image while preserving the edges. It works by applying a Gaussian filter
# to the image and weighting the intensity of each pixel based on the similarity of its intensity value to the center pixel. 
def BilateralFilter(image):
    # Apply the bilateral filter to the image
    img_filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Save the filtered image
    cv2.imwrite('filtered_image.jpg', img_filtered)
    display(img_filtered)

# -------------------------------------Box filter-------------------------------------
# In OpenCV, a box filter is a linear filter in which each pixel is replaced by the average value of its neighboring pixels in a square neighborhood
# see about its parameters from : https://www.javatpoint.com/opencv-image-filters#:~:text=Image%20filtering%20is%20the%20process,to%20increase%20brightness%20and%20contrast.
def BoxFilter(image):
    filtered_image = cv2.boxFilter(image, -1, (3, 3))
    display(image)

# -------------------------------------Image pyramid up and down-------------------------------------
# An image pyramid is a collection of images - all arising from a single original image - that are successively downsampled until some desired stopping point is reached.
# An image pyramid is a multi-scale representation of an image, in which the same image is repeatedly downsampled (resized by a factor of 2) to create a set of images with
#  different scales. The downsampled images are called pyramid layers, and the process of creating them is called pyramid construction.
def imagePyramid(image):
      cv2.imshow('Pyramids Demo', image)
      while 1:
        rows, cols, _channels = map(int, image.shape)
        
        cv2.imshow('Pyramids Demo', image)
        
        k = cv2.waitKey(0)
        if k == 27:
            break
            
        elif chr(k) == 'i':
            image = cv2.pyrUp(image, dstsize=(2 * cols, 2 * rows))
            print ('** Zoom In: Image x 2')
            
        elif chr(k) == 'o':
            image = cv2.pyrDown(image, dstsize=(cols // 2, rows // 2))
            print ('** Zoom Out: Image / 2')

imagePyramid(image)
