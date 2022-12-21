from cmu_112_graphics import *
from PIL import Image
import subprocess

def incrementLast(n):
    lights = open("lights.txt", "r")
    allSources = lights.readlines()
    print(allSources)
    lights.close()
    lights = open("lights.txt", 'w')
    lastIndex = len(allSources) - 1
    for line in range(lastIndex):
        lights.write(allSources[line])
    last = allSources[lastIndex]
    elems = last.split(" ")
    lastElemIndex = len(elems) - 1
    newSource = ""
    for elemIdx in range(lastElemIndex):
        if (elemIdx == 1):
            oldY = int(elems[elemIdx])
            if (n == 0):
                elems[elemIdx] = "280"
            else:
                elems[elemIdx] = str(oldY - 2)
        newSource += (elems[elemIdx] + " ")
    if (n == 0): 
        prevIntensity = 10
    else: prevIntensity = float(elems[lastElemIndex])
    newSource += str(prevIntensity * 1.1)
    lights.write(newSource)
    lights.close()

def cpf():
    key = input("Reset Images?: ")
    if (not(key == "yes")): return
    subprocess.call("rm -rf demodata", shell=True)
    subprocess.call("mkdir demodata", shell=True)
    frames = int(subprocess.check_output("ls ../data | wc -l", shell=True).decode().strip())
    
    for i in range(frames):
        imN = "frame" + str(i * 20) + ".png"
        imgSrc = f"../data/{imN}"
        txtF = "lights.txt" 
        outLoc = f"demodata/{imN}"
        incrementLast(i)
        subprocess.call(f"make run SRC={imgSrc} TXT={txtF} OUT={outLoc}", shell=True)
        print("done " + str(i))

    return

def appStarted(app):
    app.imgDir = "demodata"
    app.fImg = Image.open(app.imgDir + "/frame0.png")
    app.imx = app.fImg.width / 2
    app.imy = app.fImg.height / 2
    app.imgCount = 0
    app.lastFrameNum = 2760
    app.paused = False

def mousePressed(app, event):
    return

def keyPressed(app, event):
    if event.key == 'r':
        appStarted(app)
    elif event.key == 'p':
        app.paused = not app.paused
    elif event.key == "Left":
        app.imgCount -= 20
    elif event.key == "Right":
        app.imgCount += 20

    if (app.imgCount < 0): app.imgCount = int(app.lastFrameNum)
    elif (app.imgCount > app.lastFrameNum): app.imgCount = 0


def timerFired(app):
    if not app.paused:
        app.imgCount += 20
        if (app.imgCount > app.lastFrameNum):
            app.imgCount = 0
    app.fImg = Image.open(app.imgDir + f"/frame{app.imgCount}.png")

def redrawAll(app, canvas):
    canvas.create_image(app.imx, app.imy, image=ImageTk.PhotoImage(app.fImg))
    return

#cpf()
runApp(width=1066, height=600, title="Raytracer Demonstration")