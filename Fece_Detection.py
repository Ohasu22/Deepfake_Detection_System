from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib import patches
from skimage import data, io
from skimage.feature import Cascade

trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade.
detector = Cascade(trained_file)

plt.title("My picture",color='r')


image = mpimg.imread("C:/Users/Ojas Gharde/Downloads/Profile_img.jpg")






plt.imshow(image)
plt.show()