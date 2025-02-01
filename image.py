import cv2
from matplotlib import pyplot as plt
import numpy as np

def show(name,img):
  cv2.namedWindow(name, cv2.WINDOW_NORMAL)
  cv2.moveWindow(name, 100, 100)
  cv2.imshow(name, img)
  cv2.resizeWindow(name, 1200, 750)
  cv2.waitKey(0)

def convert(img):
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return gray_image

def histo(img):
  hist = cv2.calcHist([img], [0], None, [256], [0, 256])

  plt.figure()
  plt.title("Grayscale Histogram")
  plt.xlabel("Bins (Pixel Intensity)")
  plt.ylabel("# of Pixels")
  plt.plot(hist, color='black')
  plt.xlim([0, 256])
  plt.show()

def equalize(img):
    equalized = cv2.equalizeHist(img)
    hist = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    plt.figure(figsize=(12, 6))  
    plt.subplot(1, 2, 1) 
    plt.title("Equalized Image")
    plt.imshow(equalized, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2) 
    plt.title("Histogram")
    plt.bar(range(256), hist.flatten(), color='black', width=1.0)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.xlim([0, 256])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def bright(img):
    c = np.random.uniform(0, 2)
    print(f"Value of c: {c}")
    scaled = c * img
    modified_image = np.clip(scaled, 0, 255).astype(np.uint8)
    saturated = (scaled > 255)
    s = cv2.cvtColor(modified_image, cv2.COLOR_GRAY2BGR)
    s[saturated] = [0, 0, 255]  
    show("Brightened image", s)

def kernel(img):
  f = np.array([[1, 2, 2], [1, 8, 2],[0, 6, 6]])
  #sum = np.sum(f)
  #f=f/sum
  s = cv2.filter2D(img,-1,f)
  s2 = np.clip(s, 0, 255).astype(np.uint8)

  show("Filtered Image", s2)
  
def mask500(img):
  size = 500
  x = np.random.randint(0, img.shape[1] - size)
  y = np.random.randint(0, img.shape[0] - size)
  mask = np.zeros_like(img)
  mask[y:y+size, x:x+size] = 255
  s = cv2.bitwise_and(img, mask)
  show("Filtered image",s)

def edge_detection(img):
  s = cv2.Canny(img, 100, 100)
  show("Edged image",s)

def grow_R(img):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            seed_point = (x, y)
            print(f"Seed point selected: {seed_point}")

            try:
                seed_value = img[y, x]

                value = int(input("Enter the threshold value (0-255): "))

                lower_bound = max(0, seed_value - value)
                upper_bound = min(255, seed_value + value)

                mask = np.zeros_like(img, dtype=np.uint8)
                stack = [seed_point]
                visited = set()

                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) not in visited:
                        visited.add((cx, cy))

                        if lower_bound <= img[cy, cx] <= upper_bound:
                            mask[cy, cx] = 255
                            for nx, ny in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]:
                                if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0]:
                                    stack.append((nx, ny))
                cv2.namedWindow("Region Growing Result", cv2.WINDOW_NORMAL)
                cv2.moveWindow("Region Growing Result", 100, 100)
                cv2.imshow("Region Growing Result", mask)
                cv2.resizeWindow("Region Growing Result", 800, 750)
                cv2.waitKey(0)
            except ValueError:
                print("Invalid input. Please enter an integer.")

    cv2.namedWindow("Input image", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Input image", 100, 100)
    cv2.imshow("Input image", img)
    cv2.resizeWindow("Input image", 800, 750)
    cv2.setMouseCallback("Input image", mouse_callback)
    cv2.waitKey(0)

choice=-1
path=r"C:\\Users\\HP\Desktop\\image.jpg"
  
image=cv2.imread(path)


if image is None:
  print("Fail to find image")
  exit
gray_image=convert(image)


while (True):
    print("\n1. Read and show the input image.\n\n")
    print("2. Convert the image to grayscale.\n\n")
    print("3. Show the histogram of the grayscale image.\n\n")
    print("4. Perform histogram equalization and display the resulting histogram.\n\n")
    print("5. Modify the brightness of the grayscale image \n\n")
    print("6. Apply a 3x3 kernel to the grayscale image\n\n")
    print("7. Perform a masking operation on the grayscale image\n\n")
    print("8. Display the grayscale image after applying edge detection.\n\n")
    print("9. Implement a region growing algorithm\n\n")
    print("0. For exit.\n\n")
    
    choice=int(input("Enter your choice: "))

    if(choice==1):
     show("Input image",image)
     continue

    if (choice==2):
      show("Gray image",gray_image)
      continue

    elif (choice==3):
      histo(image)
      continue
    
    elif(choice==4):
      equalize(gray_image)
      continue

    elif (choice==5):
      bright(gray_image)

    elif(choice==6):
      kernel(gray_image)
      continue

    elif(choice==7):
      mask500(gray_image)
      continue
    
    elif(choice==8):
      edge_detection(gray_image)
      continue

    elif(choice==9):
      grow_R(gray_image)
      continue

    elif(choice==0):
      print("\nThank you for using.")
      break

    else:
      continue
    