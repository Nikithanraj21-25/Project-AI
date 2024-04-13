#using pillow library to image
try:
       from PIL import Image
except ImportError:
       import Image

#import OCR
import pytesseract

#import Exe. File
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#function
def recText(filename):
       text = pytesseract.image_to_string(Image.open(filename))
       return text

info = recText('test2.png')
print(info)

file = open("result.txt","w")
file.write(info)
file.close()
print("Written Successfully")


