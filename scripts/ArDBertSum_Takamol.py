"""
Created on Sat Aug  8 06:40:54 2020
@author: Abdullah M. Alshanqiti (a.m.alshanqiti@gmail.com)
For Takamol Project (#11-209)
"""

import numpy as np
import rouge
#import sys
import os.path
from statistics import mean
from _preProcessing import TClean
from _ArDocument import Docu
from _DBertSum import ArDistilBert
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score


##
##
##
##
##
##
##
##  Takamol Project
############################################################

class ArDBertSum():


    def __init__(self, TCleanObject):
        #
        #
        self.listDocuments = None
        self.iDocCategory = {}
        self.setDocCategoryIDs()
        self.tClean = TCleanObject
        #
        #
        
    #
    #
    #

    def setDocCategoryIDs(self):  
        #
        self.iDocCategory[1]=1
        self.iDocCategory[2]=1
        self.iDocCategory[3]=1
        self.iDocCategory[4]=1
        self.iDocCategory[5]=1
        self.iDocCategory[6]=1
        self.iDocCategory[7]=2
        self.iDocCategory[8]=2
        self.iDocCategory[9]=2
        self.iDocCategory[10]=2
        self.iDocCategory[11]=3
        self.iDocCategory[12]=3
        self.iDocCategory[13]=3
        self.iDocCategory[14]=3
        self.iDocCategory[15]=3
        self.iDocCategory[16]=3
        self.iDocCategory[17]=3
        self.iDocCategory[18]=3
        self.iDocCategory[19]=3
        self.iDocCategory[20]=3
        self.iDocCategory[21]=3
        self.iDocCategory[22]=3
        self.iDocCategory[23]=3
        self.iDocCategory[24]=3
        self.iDocCategory[25]=3
        self.iDocCategory[26]=3
        self.iDocCategory[27]=3
        self.iDocCategory[28]=3
        self.iDocCategory[29]=3
        self.iDocCategory[30]=3
        self.iDocCategory[31]=3
        self.iDocCategory[32]=3
        self.iDocCategory[33]=3
        self.iDocCategory[34]=3
        self.iDocCategory[35]=3
        self.iDocCategory[36]=3
        self.iDocCategory[37]=3
        self.iDocCategory[38]=3
        self.iDocCategory[39]=3
        self.iDocCategory[40]=3
        self.iDocCategory[41]=3
        self.iDocCategory[42]=3
        self.iDocCategory[43]=4
        self.iDocCategory[44]=4
        self.iDocCategory[45]=5
        self.iDocCategory[46]=5
        self.iDocCategory[47]=5
        self.iDocCategory[48]=5
        self.iDocCategory[49]=5
        self.iDocCategory[50]=5
        self.iDocCategory[51]=5
        self.iDocCategory[52]=5
        self.iDocCategory[53]=5
        self.iDocCategory[54]=5
        self.iDocCategory[55]=5
        self.iDocCategory[56]=6
        self.iDocCategory[57]=6
        self.iDocCategory[58]=6
        self.iDocCategory[59]=6
        self.iDocCategory[60]=6
        self.iDocCategory[61]=6
        self.iDocCategory[62]=6
        self.iDocCategory[63]=6
        self.iDocCategory[64]=6
        self.iDocCategory[65]=7
        self.iDocCategory[66]=7
        self.iDocCategory[67]=7
        self.iDocCategory[68]=7
        self.iDocCategory[69]=7
        self.iDocCategory[70]=7
        self.iDocCategory[71]=7
        self.iDocCategory[72]=8
        self.iDocCategory[73]=8
        self.iDocCategory[74]=8
        self.iDocCategory[75]=8
        self.iDocCategory[76]=8
        self.iDocCategory[77]=8
        self.iDocCategory[78]=8
        self.iDocCategory[79]=8
        self.iDocCategory[80]=8
        self.iDocCategory[81]=8
        self.iDocCategory[82]=8
        self.iDocCategory[83]=9
        self.iDocCategory[84]=9
        self.iDocCategory[85]=9
        self.iDocCategory[86]=9
        self.iDocCategory[87]=9
        self.iDocCategory[88]=9
        self.iDocCategory[89]=9
        self.iDocCategory[90]=9
        self.iDocCategory[91]=9
        self.iDocCategory[92]=9
        self.iDocCategory[93]=10
        self.iDocCategory[94]=10
        self.iDocCategory[95]=10
        self.iDocCategory[96]=10
        self.iDocCategory[97]=10
        self.iDocCategory[98]=10
        self.iDocCategory[99]=10
        self.iDocCategory[100]=10
        self.iDocCategory[101]=10
        self.iDocCategory[102]=10
        self.iDocCategory[103]=10
        self.iDocCategory[104]=10
        self.iDocCategory[105]=10
        self.iDocCategory[106]=10
        self.iDocCategory[107]=1
        self.iDocCategory[108]=2
        self.iDocCategory[109]=3
        self.iDocCategory[110]=5
        self.iDocCategory[111]=5
        self.iDocCategory[112]=5
        self.iDocCategory[113]=6
        self.iDocCategory[114]=6
        self.iDocCategory[115]=6
        self.iDocCategory[116]=6
        self.iDocCategory[117]=6
        self.iDocCategory[118]=6
        self.iDocCategory[119]=6
        self.iDocCategory[120]=1
        self.iDocCategory[121]=1
        self.iDocCategory[122]=1
        self.iDocCategory[123]=2
        self.iDocCategory[124]=2
        self.iDocCategory[125]=4
        self.iDocCategory[126]=4
        self.iDocCategory[127]=4
        self.iDocCategory[128]=4
        self.iDocCategory[129]=4
        self.iDocCategory[130]=4
        self.iDocCategory[131]=4
        self.iDocCategory[132]=4
        self.iDocCategory[133]=4
        self.iDocCategory[134]=4
        self.iDocCategory[135]=4
        self.iDocCategory[136]=4
        self.iDocCategory[137]=4
        self.iDocCategory[138]=4
        self.iDocCategory[139]=4
        self.iDocCategory[140]=5
        self.iDocCategory[141]=5
        self.iDocCategory[142]=5
        self.iDocCategory[143]=6
        self.iDocCategory[144]=6
        self.iDocCategory[145]=6
        self.iDocCategory[146]=6
        self.iDocCategory[147]=6
        self.iDocCategory[148]=7
        self.iDocCategory[149]=8
        self.iDocCategory[150]=8
        self.iDocCategory[151]=8
        self.iDocCategory[152]=8
        self.iDocCategory[153]=8        
        return None
    #
    #
    #
    
    def loadDocuments(self):
    
        sentenceCount = []
        wordCount =[]
        distincWords = {}
        self.listDocuments = []
        strDocPath = "../resources/EASCDataSet/ins{}/x.txt"
                
        #
        #        
        
        for i in range (1, 154):   # (1, 154)
            
            strDoc = Docu(i, 
                     self.tClean.readTxtFile(strDocPath.format(i)),
                     self.iDocCategory,
                     self.tClean)

            #
            #        
            sentenceCount.append(strDoc.getNumberOfSentences())
            wordCount.append(strDoc.getNumberOfWords())
            #
            #        
            for key in strDoc.dWords:
                self.tClean.add_if_key_not_exist(distincWords, key, None)        
            #
            #        
            self.listDocuments.append(strDoc)
            ##### strDoc.orgDocument = None
            #
            #
            print(".", end='')
            if i % 30 == 0 and i>1:
                print(". ", i)
        print(".  153")   
          
        #######################################
        # Printing out sentence/word statistics  
        print ("\nSentences: total({}), min ({}), max ({}), avg({})".format(
            sum(sentenceCount), 
            min(sentenceCount), 
            max(sentenceCount), 
            mean(sentenceCount)))
        
        print ("Words: total({}), min ({}), max ({}), avg({})".format(
            sum(wordCount), 
            min(wordCount), 
            max(wordCount), 
            mean(wordCount)))
        
        print ("Distinct Words: {}".format(len(distincWords)))
            
        return self.listDocuments

############################################################

##
##
##
##
##      End of Class ArDBertSum
##
##
##
##

############################################################
    












##
##
##  Testing functions ...

############################################################

    #    
    #
    #
def checkSentenceSegmentation(xDocuments, tClean):
    #
    #
    print("\n\nTesting sentence segmentaion ...")
    
    sentLenFile = open("../experiments/GeneratedSummaries/allSentences.csv", "w", encoding="utf-8")
    #
    #
    for xDi in xDocuments:        
        #
        sentCounter=0
        xDi.finalSummary = ""
        #
        #        
        for strSent in xDi.nSentences:
            sentCounter += 1
            #
            strSentSegList = tClean.getSegSentTokenization(strSent)
            #
            for strSeg in strSentSegList:
                #
                #       DocID, SentID, SentSize, SentText
                #
                xDi.finalSummary +=strSeg + '. '
                sentLenFile.write("{},{},{},{}\n".format(xDi.iID, sentCounter, len(strSeg), strSeg))
                
    sentLenFile.close()
           
    return None

    #    
    #
    #
    
def preformSummaryForMaxScoreMeasurement(xDocuments, tClean):
    #
    #
    print("\n\nRunning summarization models for measuring the max-scores ...")
    iCount = 1
         
    #
    #    
    for xDi in xDocuments:
        #
        #
        xDi.iniSummary = xDi.getDoc(False, True)
        pBase = getAvgRouge (xDi, xDi.iniSummary, tClean)
        #
        allSentSeq = tClean.getSentTokenization(xDi.iniSummary)
        #
        #
        xDi.finalSummary = ""
        iSeqCount=1
        for strSeg in allSentSeq:
            doKeep = shallKeepIt(xDi, strSeg, pBase, tClean)
            if doKeep:
                xDi.finalSummary += strSeg + ' '
            #pTrace(xDi, iSeqCount, int(doKeep), strSeg)
            iSeqCount+=1
        #
        #
        xDi.iniSummary = xDi.orgDocument
        xDi.saveSummary()
        #
        #        
        print(".", end='')
        iCount+=1
        if iCount % 30 == 0:
            print(". ", iCount)
    print(". ", iCount-1)
    #
    #
    #
    return None    


def shallKeepIt(xDi, strSeg, pBase, tClean):    
    txtUnderTest = xDi.iniSummary.replace(strSeg, ' ')    
    if getAvgRouge(xDi, txtUnderTest, tClean) < pBase:
        return True
    return False

        
def getAvgRouge(xDi, genratedtxt, tClean):
    
    genratedtxt = genratedtxt.replace("\n", "")
    SRouge = rouge.Rouge()    
    allScores =[] 
    for ref in ['A','B','C','D','E']:
        refDoc = tClean.readTxtFile("../resources/EASCDataSet/ins{}/{}.txt".format(xDi.iID, ref))
        pS = SRouge.get_scores(genratedtxt, refDoc.replace("\n", ""))[0]['rouge-2']['p']
        allScores.append(pS)  
        
    return np.mean(allScores)
    
    
def pTrace(xDi, iSeqCount, doKeep, strSeg):
        
    with open("../experiments/tracingDS.csv",'a+', encoding="utf-8") as traceCSV:
        if xDi.iID <2 and iSeqCount <2:
            traceCSV.write("DocID,SentID,DecisionToInclude,docLen,SentLen,SentAVG,TF,IDF,strSeg\n")    
            
        traceCSV.write("{},{},{},{},{}\n".format(
            xDi.iID, 
            iSeqCount, 
            doKeep, 
            xDi.getTotalTF_IDF(strSeg), 
            strSeg))
        
    return None    
    
    
    
def preformSummaryForMaxScoreMeasurementBasedOnDroppingToken(xDocuments, tClean):
    #
    #
    print("\n\nRunning summarization models for measuring the max-scores ...")
    iCount = 1
         
    #
    #    
    for xDi in xDocuments:        
        #
        #
        xDi.iniSummary = xDi.getDoc(False, True)
        #
        # 1 - Put all token form the 5 refs in a list
        # 2 - word-Token (iniSummary) then keep words that exists in List while droping the un-exist
        refDistincList = {}
        for ref in ['A','B','C','D','E']:
            getWTokens = tClean.getWTokens(tClean.readTxtFile("../resources/EASCDataSet/ins{}/{}.txt".format(xDi.iID, ref)))
            
            for wToken in getWTokens:
                tClean.add_if_key_not_exist(refDistincList, wToken, None)
        
        getInISumTokens = tClean.getWTokens(xDi.iniSummary)
        xDi.finalSummary = ""
        for iniWToken in getInISumTokens:
            iniWToken = iniWToken.strip()
            if iniWToken in refDistincList:
                xDi.finalSummary += iniWToken + ' '        
        #
        #
        xDi.saveSummary()       
        #
        #        
        print(".", end='')
        iCount+=1
        if iCount % 30 == 0:
            print(". ", iCount)
    print(". ", iCount-1)
    #
    #
    #
    return None    
    
    #    
    #
    #    
  
    
  
    
  
##
##
##  Summarization functions 

############################################################


    
def preformSummary(xDocuments, tClean, statisticalEquation=False):
        
    #
    #
    print("\n\nRunning summarization models ...")
    iCount = 1
    
    arBert = ArDistilBert()        
    #
    #    
    for xDi in xDocuments:        
        #
        # Generate summaries from original text using DistilBERT
        #xDi.iniSummary = arBert.getSummaryFromOriginal(xDi)
        #xDi.finalSummary = xDi.iniSummary
        #
        #
        loadAndGetSum(arBert, xDi, tClean, True, statisticalEquation)
        #
        #        
        xDi.saveSummary()       
        #
        #        
        print(".", end='')
        iCount+=1
        if iCount % 30 == 0:
            print(". ", iCount)
    print(". ", iCount-1)
    #
    #
    #


def loadAndGetSum(arBert, xDi, tClean, overwriteSumFiles=False, statisticalEquation=False):
        
    #
    # [1]: ini-summary
    #
    # Check if the ini-summary-files have been generated !
    if overwriteSumFiles:
        strPath = "../experiments/GeneratedSummaries/iniSummaries/{}_ini.txt".format(xDi.iID)
        if os.path.isfile(strPath):            
            xDi.iniSummary = tClean.readTxtFile(strPath)
        else:
            xDi.iniSummary = arBert.getSummaryFromCleanedSentences(xDi)
    #
    #
    #
    # [2]: final summary
    #
    # Check if the final-summary-files have been generated !
    if overwriteSumFiles:
        strPath = "../experiments/GeneratedSummaries/finalSummaries/{}_final.txt".format(xDi.iID)
        if os.path.isfile(strPath):            
            xDi.finalSummary = tClean.readTxtFile(strPath)
        elif statisticalEquation:
            xDi.finalSummary = getSummaryBasedOnStatisticalRankings(xDi, tClean)
        else:
            xDi.finalSummary = arBert.getSummaryFromIniSum(xDi)  
    #
    #
    #
    return None

    #
    #
    #

def getSummaryBasedOnStatisticalEquation (xDi, tClean):
   
    statisticalSummary = ""
    allSentSeq = tClean.getSentTokenization(xDi.getDoc(True, True))
    #
    for strSeg in allSentSeq:
        docLen, SentLen, SentAVG, TF, IDF = xDi.getDStatsTF_IDF(strSeg)
        if (SentAVG >= 0.34) and (TF <= 0.19) and (SentLen >= 89) and (docLen <= 270):
            continue
        statisticalSummary += strSeg + ' '
    #
    #
    return statisticalSummary

#
#
#
  

def getSummaryBasedOnStatisticalRankings (xDi, tClean, sumRatio=0.2):
   
    statisticalRankingSummary = ""
    allSentSeq = tClean.getSentTokenization(xDi.getDoc(True, True))
    #
    dRankings = []
    #
    for strSeg in allSentSeq:
        docLen, SentLen, SentAVG, TF, IDF = xDi.getDStatsTF_IDF(strSeg)
        dRankings.append(IDF/TF)
            
    #
    # 
    # normalizing ranks values ... 
    dRankings = np.array(dRankings)
    dMin = dRankings.min()
    dMaxMin = dRankings.max() - dMin    
    dRankings = (dRankings - dMin)  / dMaxMin
    
    #
    #
    #
    iIndex =-1
    sumRatio = 1-sumRatio
    for strSeg in allSentSeq:
        iIndex+=1
        if dRankings[iIndex]>=sumRatio:
            statisticalRankingSummary += strSeg + ' '
    #
    #
    #
    
    return statisticalRankingSummary

#
#
#
    
def measureROUGE(xDocuments, tClean):
    #        
    print("\n\nRunning ROUGE metric ...")
    #
    #
    removeStopWord=False
    applyLemmatize=False
    iCount = 1
    #
    #
    with open("../experiments/allRougeOutputs.csv",'w') as rougeCSV:
        #
        #
        for xDi in xDocuments: 
            #
            #
            sumDocIni = tClean.hardCleaning(xDi.iniSummary, removeStopWord, applyLemmatize)
            sumDocFinal = tClean.hardCleaning(xDi.finalSummary, removeStopWord, applyLemmatize)
            #
            #
            for ref in ['A','B','C','D','E']:
                #
                refDoc = tClean.readTxtFile("../resources/EASCDataSet/ins{}/{}.txt".format(xDi.iID, ref))
                refDoc = tClean.hardCleaning(refDoc, removeStopWord, applyLemmatize)    
                #
                #
                rougeCSV.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    xDi.iID,
                    ref,
                    getRougeScores(sumDocIni, refDoc, tClean),
                    getRougeScores(sumDocFinal, refDoc, tClean),
                    xDi.iCategory, 
                    getBLUEAndMEteroScores(sumDocFinal, refDoc, tClean)))
            #
            #
            print(".", end='')
            iCount+=1
            if iCount % 30 == 0:
                print(". ", iCount)
        print(". ", iCount-1)
    #
    #
    #
    
def getRougeScores(sumDoc, refDoc, tClean):
    #
    #
    sumDoc +=" "
    refDoc +=" "
    #
    #    
    rScores = rouge.Rouge()
    scores = rScores.get_scores(sumDoc, refDoc)[0]        
    #
    #    
    return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
        tClean.toRound(scores['rouge-1']['r']),
        tClean.toRound(scores['rouge-1']['p']),
        tClean.toRound(scores['rouge-1']['f']),
        tClean.toRound(scores['rouge-2']['r']),
        tClean.toRound(scores['rouge-2']['p']),
        tClean.toRound(scores['rouge-2']['f']),
        tClean.toRound(scores['rouge-l']['r']),
        tClean.toRound(scores['rouge-l']['p']),
        tClean.toRound(scores['rouge-l']['f']))


def getBLUEAndMEteroScores(sumDoc, refDoc, tClean):
    #refDoc = tClean.getSentTokenization(refDoc)
    BLUE= sentence_bleu(refDoc, sumDoc)
    MEtero = meteor_score(refDoc, sumDoc)
    #
    #    
    return "{}\t{}".format(
        tClean.toRound(BLUE), 
        tClean.toRound(MEtero))
    
        
############################################################

def main():
    
    #
    # Loading and pre-processing all documents 
    tClean = TClean()
    arSum = ArDBertSum(tClean)       
    xDocuments = arSum.loadDocuments()
    
    #    
    #
    # Testing functions ...
    #checkSentenceSegmentation(xDocuments, tClean)    
    #preformSummaryForMaxScoreMeasurement(xDocuments, tClean)
    
    
    #    
    #
    # Summarization functions
    statisticalEquation=False
    preformSummary(xDocuments, tClean, statisticalEquation)    
    measureROUGE(xDocuments, tClean)
    #
    #  
    #xDocuments[0].toPrintTF_IDF()
    #print(xDocuments[105].nSentences)
    print ("\n\nProcess completed.")
    
    
if __name__ == "__main__":
    main()
    
##
##
##
##
##
##
##
## End
############################################################


