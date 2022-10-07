# -*- coding: utf-8 -*-
"""
Created on Fri May 20 07:05:34 2022

@author: Abdullah Alshanqiti
"""

import rouge
import numpy as np
from evaluate import load
SARI = load("sari")
BLEU = load("bleu")
TER = load("ter")
METEOR = load("meteor")



def readTxtFile (strPath):
    with open(strPath, 'r', encoding="utf-8") as file:
        return file.read().replace("\n", " ")


def dLoad(ins=498):    
    #
    sources = []
    hReferences = []    
    predFourier = [] 
    #
    predAraT5 = [] 
    predFacebook= [] 
    predGoogle= [] 
    predArabicT5= [] 
    predArabicT5Small= [] 
    #for i in range(1, ins):
    for i in range(12, 13):
        sources.append(readTxtFile("../resources/Simplification_Datasets/References/{}.txt".format(i)))
        hReferences.append(readTxtFile("../resources/Simplification_Datasets/hSimplification/{}.txt".format(i)))        
        predFourier.append(readTxtFile("../evaluation/Rephrase/Fourier/{}.txt".format(i)))
        
        predAraT5.append(readTxtFile("../evaluation/Rephrase/UBC-NLP-AraT5/{}.txt".format(i)))
        predFacebook.append(readTxtFile("../evaluation/Rephrase/Facebook-mBART-large-50/{}.txt".format(i)))
        predGoogle.append(readTxtFile("../evaluation/Rephrase/google-mt5-base/{}.txt".format(i)))
        predArabicT5.append(readTxtFile("../evaluation/Rephrase/arabic-t5/{}.txt".format(i)))
        predArabicT5Small.append(readTxtFile("../evaluation/Rephrase/arabic-t5-small/{}.txt".format(i)))     
    
    print (sources)
    modelPredData = {}    
    modelPredData ['predFourier'] = predFourier
    modelPredData ['predAraT5'] = predAraT5
    modelPredData ['predFacebook'] = predFacebook    
    modelPredData ['predGoogle'] = predGoogle
    modelPredData ['predArabicT5'] = predArabicT5
    modelPredData ['predArabicT5Small'] = predArabicT5Small
    
               
    return sources, hReferences, modelPredData




def getSARI_score(s, p, r):    
    sources, predictions, references = [],[],[]
    sources.append(s)
    predictions.append(p)    
    inerRef = []
    inerRef.append(r)
    references.append(inerRef)
    sari_score = SARI.compute(sources=sources, predictions=predictions, references=references)    
    return sari_score['sari']/100


def getBLEU_score(p, r): 
    predictions, references = [],[]    
    predictions.append(p)    
    inerRef = []
    inerRef.append(r)
    references.append(inerRef)
    bleu_score = BLEU.compute(predictions=predictions, references=references)    
    return bleu_score['bleu']


def getTER_score(p, r):    
    predictions, references = [],[]    
    predictions.append(p)    
    inerRef = []
    inerRef.append(r)
    references.append(inerRef)
    ter_score = TER.compute(predictions=predictions, references=references, case_sensitive=False)    
    return ter_score['score']/100


def getMETEOR_score(p, r): 
    predictions, references = [],[]    
    predictions.append(p)    
    inerRef = []
    inerRef.append(r)
    references.append(inerRef)
    meteor_score = METEOR.compute(predictions=predictions, references=references)    
    return meteor_score['meteor']


def getROUGE_score(p, r):        
    rScors = rouge.Rouge().get_scores(p, r)[0]
    r1, r2, rL = rScors['rouge-1']['f'], rScors['rouge-2']['f'], rScors['rouge-l']['f']    
    return r1, r2, rL


def getAvgFromList(dList):
    return np.mean(np.array(dList))






sources, hReferences, modelPredData = dLoad()
allMetricsFile = open("../evaluation/Rephrase/allMetrics.txt", "w", encoding="utf-8")

icount=0
for predModelName in modelPredData:
    
    predData = modelPredData[predModelName]

    SARIScore = []
    BLEUScore = []
    TERScore = []
    METEORScore = []
    ROUGEScore1 = []
    ROUGEScore2 = []
    ROUGEScoreL = []
            
    for i in range(len(sources)):
        icount+=1
        print("\r", "Progress {:2.1%}".format(icount / (2982)), end="") 
        SARIScore.append(getSARI_score(sources[i], predData[i], hReferences[i]))
        BLEUScore.append(getBLEU_score(predData[i], hReferences[i]))
        TERScore.append(getTER_score(predData[i], hReferences[i]))
        METEORScore.append(getMETEOR_score(predData[i], hReferences[i]))
        
        r1, r2, rL = getROUGE_score (predData[i], hReferences[i])     
        ROUGEScore1.append(r1)
        ROUGEScore2.append(r2)
        ROUGEScoreL.append(rL)
    
    allMetricsFile.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
        predModelName, 
        getAvgFromList(SARIScore),
        getAvgFromList(BLEUScore),
        getAvgFromList(TERScore),
        getAvgFromList(METEORScore),
        getAvgFromList(ROUGEScore1),
        getAvgFromList(ROUGEScore2),
        getAvgFromList(ROUGEScoreL)))
    

allMetricsFile.close()  

print ("\n\nEval... done.")



