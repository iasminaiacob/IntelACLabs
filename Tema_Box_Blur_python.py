from PIL import Image, ImageFilter
import operator
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#luam imaginea, iar la fiecare pixel, calculam suma pixelilor din jurul sau si inlocuim pixelul curent cu aceasta suma
def box_blur(file, r=1):
    img = Image.open(file).convert('RGB')
    newimg = img.copy()
    width, height = img.size
    area = (2*r+1)**2
    for x in range(r, width-r):
        for y in range(r, height-r):
            sum_pixels = (0, 0, 0)
            for pixel in [(i, j) for j in range(y-r, y+r+1) for i in range(x-r, x+r+1)]:
                sum_pixels = tuple(map(operator.add, sum_pixels, img.getpixel(pixel)))
            newimg.putpixel((x, y), tuple(map(operator.floordiv, sum_pixels, (area, area, area))))
    return newimg

def summed_table(img):
    width, height = img.size
    table = [[(0, 0, 0)]*width for i in range(height)]
    table[0][0] = img.getpixel((0, 0))
    for x in range(1, width):
        table[0][x] = tuple(map(operator.add, img.getpixel((x, 0)), table[0][x-1]))
    for y in range(1, height):
        table[y][0] = tuple(map(operator.add, img.getpixel((0, y)), table[y-1][0]))
    for x in range(1, width-1):
        for y in range(1, height-1):
            table[y][x] = tuple(map(operator.sub,
                                    tuple(map(operator.add,
                                              tuple(map(operator.add,
                                                        img.getpixel((x, y)),
                                                        table[y-1][x])),
                                              table[y][x-1])),
                                    table[y-1][x-1]))
    return table



def box_blur(file, r=1):
    img = Image.open(file).convert('RGB')
    newimg = img.copy()
    width, height = img.size
    area = (2*r + 1)**2
    table = summed_table(img)
    for x in range(r+1, width-r-1):
        for y in range(r+1, height-r-1):
            sum_pixels = tuple(map(operator.add,
                                   tuple(map(operator.sub,
                                             tuple(map(operator.sub,
                                                       table[y+r][x+r],
                                                       table[y+r][x-r-1])),
                                             table[y-r-1][x+r])),
                                   table[y-r-1][x-r-1]))
            newimg.putpixel((x, y), tuple(map(operator.floordiv, sum_pixels, (area, area, area))))
    return newimg

imgplot = plt.imshow(box_blur("lena512.bmp",r=1))
plt.show()