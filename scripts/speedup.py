import subprocess
import os

def calcSpeedup(image):
    nThreads = int(subprocess.check_output("echo $OMP_NUM_THREADS", shell=True).decode())
    trials = 3
    cmd1 = f"./raytrace-parallel-v1 ../images/{image} ../images/convolution_output.png lights.txt ../images/outputParV1.png"
    cmd2 = f"./raytrace-parallel ../images/{image} ../images/convolution_output.png lights.txt ../images/outputPar.png"
    cmd3 = f"./imageCompare ../images/outputParV1.png ../images/outputPar.png"
    seqTotal = 0
    parTotal = 0
    for i in range(trials):
        output = subprocess.check_output(cmd1, shell=True).decode()
        time = float(output.split(' ')[3].strip()[:-1])
        seqTotal += time
        print(f"Trial {i+1} Sequential done in {time} seconds")

    seqAvg = seqTotal / trials

    for i in range(trials):
        output = subprocess.check_output(cmd2, shell=True).decode()
        time = float(output.split(' ')[3].strip()[:-1])
        parTotal += time
        print(f"Trial {i+1} Parallel done in {time} seconds on {nThreads} threads")
    
    os.system(cmd3)
    parAvg = parTotal / trials

    print(f"Total Speedup: {seqAvg / parAvg}")


# print(subprocess.check_output("echo $OMP_NUM_THREADS", shell=True))
calcSpeedup("terra.png")