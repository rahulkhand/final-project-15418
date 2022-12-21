# Note: This file contains actual measured speedup data. DO NOT MODIFY speedup numbers

from matplotlib import pyplot as plt
import numpy as np

def plotIt(dt, color, title, xLabel, yLabel, sceneLabel, plotMarker):
    x = dt[:, 0].reshape(dt.shape[0], 1)
    y = dt[:, 1].reshape(dt.shape[0], 1)
    plt.plot(x, y, color=color, label=sceneLabel, marker=plotMarker)
    #plt.scatter(x, y, color=color)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

def plotOfTerraSeqVsParOnMacRays():
    oneSourceSpeedup = np.array([
        [1, 1.000552881042825],
        [2, 1.9822300976087563],
        [4, 3.748199261545911],
        [8, 4.878900899090617],
        [16, 4.800114745250956],
        [32, 4.6393907244619585]
    ])

    fiveSourcesSpeedup = np.array([
        [1, 1.041150889145979],
          [2, 2.0541120507830826],
          [4, 3.8764166036903855],
          [8, 5.1293736805785635],
          [16, 5.0108752085048],
          [32, 4.827344669774165]
    ])

    tenSourcesSpeedup = np.array([
        [1, 1.0328410443249183],
        [2, 2.053155692971682],
        [4, 3.76025084567074],
        [8, 5.0717494575146596],
        [16, 5.067638226650999],
        [32, 4.9431755090143925]
    ])

    oneSourceSpeedupGHC = np.array([
        [1, 0.9787300316709009],
        [2, 1.875914018741562],
        [4, 3.5342725361614002],
        [8, 6.313843886409522],
        [16, 6.605571150939321],
        [32, 6.305894450052724]
    ])

    fiveSourcesSpeedupGHC = np.array([
        [1, 0.9754093835671611],
        [2, 1.8961888507749962],
        [4, 3.6132903150220135],
        [8, 6.677453994463555],
        [16, 6.560783896993823],
        [32, 6.600475617267986]
    ])

    tenSourcesSpeedupGHC = np.array([
        [1, 0.9747589050738296],
        [2, 1.9080869393556283],
        [4, 3.6362414659663287],
        [8, 6.743353436167501],
        [16, 6.556414068675876],
        [32, 6.435335035439954]
    ])

    oneSourceSpeedupPSC = np.array([
        [1, 1.005687476040083],
        [2, 1.876116377063807],
        [4, 3.485291524325477],
        [8, 5.935644047702071],
        [16, 9.026599698430566],
        [32, 10.13043114173605],
        [64, 7.10007370060803],
        [128, 5.303234762598565],
        [256, 7.82162585159271],
        [512, 6.2775265740589745]
    ])

    fiveSourcesSpeedupPSC = np.array([
        [1, 1.010847930028542],
        [2, 1.9183311280004054],
        [4, 3.700973688388983],
        [8, 6.657848826786456],
        [16, 10.74694787425308],
        [32, 14.109037551811872],
        [64, 13.81973910168208],
        [128, 10.31316292790972],
        [256, 14.12346289798655],
        [512, 13.325150598092494]
    ])

    tenSourcesSpeedupPSC = np.array([
        [1, 1.0038298467277327],
        [2, 1.9391980823229207],
        [4, 3.7300121366718324],
        [8, 6.905506652018133],
        [16, 10.782005044338774],
        [32, 13.075275970136728],
        [64, 12.811417830879282],
        [128, 9.94417525787498],
        [256, 12.886497468260306],
        [512, 12.047554305149845]
    ])

    plt.rcParams["figure.figsize"] = [9.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    plt.rcParams.update({'text.color': "red",
                        'axes.labelcolor': "green"})
                        
    xLabel = "Number of Threads"
    yLabel = "Speedup"
    title = "Speedup vs. Number of Threads with Varying Light Source Counts on Large Image"

    plotIt(oneSourceSpeedup, "orange", title, xLabel, yLabel, "1 Source Mac", ".")
    plotIt(fiveSourcesSpeedup, "purple", title, xLabel, yLabel, "5 Sources Mac", ".")
    plotIt(tenSourcesSpeedup, "limegreen", title, xLabel, yLabel, "10 Sources Mac", ".")

    plotIt(oneSourceSpeedupGHC, "orange", title, xLabel, yLabel, "1 Source GHC", "*")
    plotIt(fiveSourcesSpeedupGHC, "purple", title, xLabel, yLabel, "5 Sources GHC", "*")
    plotIt(tenSourcesSpeedupGHC, "limegreen", title, xLabel, yLabel, "10 Sources GHC", "*")

    plt.rcParams.update({'text.color': "black",
                     'axes.labelcolor': "black"})
    plt.legend(title="Scene")
    plt.show()

    plotIt(oneSourceSpeedupPSC, "orange", title, xLabel, yLabel, "1 Source PSC", "p")
    plotIt(fiveSourcesSpeedupPSC, "purple", title, xLabel, yLabel, "5 Sources PSC", "p")
    plotIt(tenSourcesSpeedupPSC, "limegreen", title, xLabel, yLabel, "10 Sources PSC", "p")
    plt.legend(title="Scene")
    plt.show()

# lower light source counts chosen to be sparse
def plotOfTerraSeqVsCudaRays():
    sourcesPointsGHC = np.array([
        [1, 14.12701693942004],
        [2, 24.783475262742492],
        [4, 30.606299100535125],
        [8, 46.841582906978665],
        [16, 57.338629948720985],
        [32, 72.48956130098914],
        [64, 91.4222757655837],
        [128, 98.23185189016053],
        [256, 102.67158996038866],
        [512, 104.97753805944215],
        [1024, 107.3367227736005]
    ])

    plt.rcParams["figure.figsize"] = [9.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    xLabel = "Number of Light Sources"
    yLabel = "Speedup"
    title = "Speedup vs. Number of Light Sources on Large Image"

    plotIt(sourcesPointsGHC, "green", title, xLabel, yLabel, "Cuda GHC", "h")


    plt.legend(title="Platform")
    plt.show()

def plotOfPartialScoresSegmentPercentage():
    partialCudaScoresPropGHC = np.array([
        [1, 17.2796],
        [2, 33.826613],
        [4, 43.224870],
        [8, 61.243042],
        [16, 72.885531],
        [32, 83.025245],
        [64, 86.366887],
        [128, 90.623111],
        [256, 92.973094],
        [512, 94.184642],
        [1024, 94.717059]
    ])

    partialOpenMPScoresPropGHC = np.array([
        [1, 89.921623],
        [2, 91.306148],
        [4, 92.251786],
        [8, 94.617732],
        [16, 96.066076],
        [32, 97.279369],
        [64, 97.746962],
        [128, 97.709903],
        [256, 97.878596],
        [512, 97.757605],
        [1024, 97.918235]
    ])

    plt.rcParams["figure.figsize"] = [9.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    xLabel = "Number of Light Sources"
    yLabel = "Percentage of Algorithm Computing Partial Scores"
    title = "Percentage of Algorithm Scoring vs. Number of Light Sources"

    plotIt(partialCudaScoresPropGHC, "turquoise", title, xLabel, yLabel, "Cuda GHC", "h")
    plotIt(partialCudaScoresPropGHC, "gold", title, xLabel, yLabel, "Cuda GHC", "s")

    plt.legend(title="Platform")
    plt.show()

def plotOfTerraSeqVsParConvolution():
    macSpeeds = np.array([
        [1, 0.998737],
        [2, 1.936804],
        [4, 3.626285],
        [8, 4.423789],
        [16, 4.275551]
    ])

    GHCspeeds = np.array([
        [1, 0.956488],
        [2, 1.882255],
        [4, 3.710230],
        [8, 7.287486],
        [16, 7.096235]
    ])

    plt.rcParams["figure.figsize"] = [9.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    xLabel = "Number of Threads"
    yLabel = "Speedup (relative to sequential)"
    title = "Speedup of Convolution with OpenMP Threads"

    plotIt(macSpeeds, "turquoise", title, xLabel, yLabel, "OpenMP Mac", "h")
    plotIt(GHCspeeds, "gold", title, xLabel, yLabel, "OpenMP GHC", "s")

    plt.legend(title="Platform")
    plt.show()


def plotOfISPCTaskSpeedup():
    GHCspeeds_nocopy = np.array([
        [1, 19.296965],
        [2, 38.224314],
        [4, 69.666616],
        [8, 138.597608],
        [16, 49.221255]
    ])
    GHCspeeds = np.array([
        [1, 4.344882],
        [2, 4.717940],
        [4, 4.948327],
        [8, 5.054209],
        [16, 4.900920]
    ])

    localSpeeds_nocopy = np.array([
        [1, 5.797375],
        [2, 11.523137],
        [4, 21.464800],
        [8, 22.958131],
        [16, 24.554258]
    ])

    localSpeeds_withcopy = np.array([
        [1, 2.855783],
        [2, 3.730772],
        [4, 4.399333],
        [8, 4.583366],
        [16, 4.484858]
    ])

    plt.rcParams["figure.figsize"] = [9.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    xLabel = "Number of Tasks"
    yLabel = "Speedup (relative to sequential)"
    title = "Speedup of Convolution with ISPC and OpenMP Threads"

    plotIt(GHCspeeds_nocopy, "turquoise", title, xLabel, yLabel, "Exclude Copy Overhead GHC", "h")
    plotIt(GHCspeeds, "pink", title, xLabel, yLabel, "Include Copy Overhead GHC", "s")
    plotIt(localSpeeds_nocopy, "orange", title, xLabel, yLabel, "Exclude Copy Overhead Mac", "h")
    plotIt(localSpeeds_withcopy, "gray", title, xLabel, yLabel, "Include Copy Overhead Mac", "s")


    plt.legend(title="Modality")
    plt.show()

def plotOfParV2toParV1():
    mac8threads = np.array([
        [1, 5.023387835064516],
        [2, 2.419366011683174],
        [4, 2.3151868083574367],
        [8, 1.1712794573855558],
        [16, 1.12944751645269186],
        [32, 1.2766432442131985],
        [64, 1.0990786431168031],
        [128, 1.0373523307304262],
        [256, 1.0666626286147662]
    ])

    plt.rcParams["figure.figsize"] = [9.50, 5.50]
    plt.rcParams["figure.autolayout"] = True

    xLabel = "Number of Light Sources"
    yLabel = "Inter-parallel Speedup"
    title = "Performance of v2 OpenMP raytracer against v1 OpenMP raytracer"

    plotIt(mac8threads, "blue", title, xLabel, yLabel, "Mac 8 threads", '.')
    plt.legend(title="Modality")
    plt.show()


plotOfParV2toParV1()
#plotOfISPCTaskSpeedup()
#plotOfTerraSeqVsParConvolution()
#plotOfTerraSeqVsParOnMacRays()
#plotOfTerraSeqVsCudaRays()
#plotOfPartialScoresSegmentPercentage()

