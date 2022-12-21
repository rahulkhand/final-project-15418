# run file from within raytracer directory

from PIL import Image
import random
import subprocess

def randomTest(image, n, minIntensity, incr, textFile):
    myImg = Image.open(image)
    w = myImg.width
    h = myImg.height
    listString = ""
    f = open(textFile, 'w')
    for _ in range(n):
        xcoord = random.random() * w
        ycoord = random.random() * h
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        intensity = minIntensity + (random.randint(0, 10) * incr)
        listString = f"{xcoord} {ycoord} {red} {green} {blue} {intensity}\n"
        f.write(listString)
    
    f.close()

def main():
    imageString = subprocess.check_output("pwd", shell=True).decode().strip() + "/../images/"
    imageString += "terra.png"
    textFile = "lights.txt"
    sourceNum = 256
    minIntensity = 500
    incrIntensity = 500
    randomTest(imageString, sourceNum, minIntensity, incrIntensity, textFile)

if __name__ == "__main__":
    main()