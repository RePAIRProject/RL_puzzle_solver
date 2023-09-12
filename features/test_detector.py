import pdb 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np

# FLD
img = cv2.imread('/media/lucap/big_data/datasets/wikiart/proof_of_concept/aki-kuroda_night-2011.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('/media/lucap/big_data/datasets/wikiart/proof_of_concept/albert-gleizes_on-a-sailboat.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('/media/lucap/big_data/datasets/wikiart/proof_of_concept/albert-gleizes_paysage-1914.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('/media/lucap/big_data/datasets/wikiart/proof_of_concept/albert-gleizes_portrait-de-jacques-nayral-1911.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('/media/lucap/big_data/datasets/wikiart/proof_of_concept/pablo-picasso_still-life-with-guitar-1942.jpg', cv2.IMREAD_GRAYSCALE)

img = cv2.medianBlur(img, 11)
FLD = cv2.ximgproc.createFastLineDetector(length_threshold=30)
lines_fld = FLD.detect((img).astype(np.uint8))
#(seg_img*255).astype(np.uint8)

line_on_image = FLD.drawSegments(img, lines_fld)
plt.imshow(line_on_image, interpolation='nearest', aspect='auto')
plt.show()
pdb.set_trace()