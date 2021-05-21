"""
Created on Sat Aug  8 06:40:54 2020
@author: Abdullah M. Alshanqiti (a.m.alshanqiti@gmail.com)
"""

import numpy as np
import rouge
#import sys
from statistics import mean
from _TxtClean import TClean
from _preProcessing import Docu
from _DBertSum import ArDistilBert
from _RegExSum import RegExp



##
##
##
##
##
##
##
##
############################################################

class RegBERT():


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
##      End of Class RegBERT
##
##
##
##

############################################################
    
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
        xDi.iniSummary = xDi.orgDocument
        #
        # 1 - Put all token form the 5 ref in a list
        # 2 - word-Token (iniSummary) then keep words that exists in List while droping the un-exist
        dNoNExistRefTokens = {}
        for ref in ['A','B','C','D','E']:
            getWTokens = tClean.getWTokenSize(tClean.readTxtFile("../resources/EASCDataSet/ins{}/{}.txt".format(xDi.iID, ref)))
            
            for wToken in getWTokens:
                tClean.add_if_key_not_exist(dNoNExistRefTokens, wToken, None)
        
        getWSumTokens = tClean.getWTokenSize(xDi.iniSummary)
        xDi.finalSummary = ""
        for wToken in getWSumTokens:
            wToken = wToken.strip()
            if wToken in dNoNExistRefTokens:
                xDi.finalSummary += wToken + ' '        
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
    
def preformSummary(xDocuments, tClean):
    
    #XXX To be Removed ...
    #with open("../experiments/tracingDS.csv",'w+') as traceCSV:
    #    traceCSV.write("DocID,SentID,SentPartID,Abase,A,Bbase,B,Cbase,C,Dbase,D,Ebase,E,Avg,avgDecision\n")
            
    #
    #
    print("\n\nRunning summarization models ...")
    iCount = 1
    
    arBert = ArDistilBert()        
    #
    #    
    for xDi in xDocuments:        
        #
        # generate ini-summary using DistilBERT
        xDi.iniSummary = arBert.getInISummary(xDi, tClean)               
        #
        # re-summary (iniSummary) using regExp
        re=RegExp(xDi, tClean)
        re.setFinalSummary()        
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


def measureROUGE(xDocuments, tClean, withStemming=None):
    #        
    print("\n\nRunning ROUGE metric ...")
    #
    iCount = 1
    #     
    with open("../experiments/allRougeOutputs.csv",'w') as rougeCSV:
        #
        #
        for xDi in xDocuments: 
            #
            #
            sumDocIni = xDi.iniSummary 
            sumDocFinal = xDi.finalSummary
            if withStemming is not None: 
                _, sumDocIni = tClean.getWordTokenWithRemovingStopOfWords(sumDocIni, True)
                _, sumDocFinal = tClean.getWordTokenWithRemovingStopOfWords(sumDocFinal, True)
            #
            #            
            for ref in ['A','B','C','D','E']:
                refDoc = tClean.readTxtFile("../resources/EASCDataSet/ins{}/{}.txt".format(xDi.iID, ref))
                    
                #
                #
                #
                if withStemming is not None:        
                    _, refDoc = tClean.getWordTokenWithRemovingStopOfWords(refDoc, True)
                #
                #
                #
                rougeCSV.write("{}\t{}\t{}\t{}\t{}\n".format(
                    xDi.iID,
                    ref,
                    getRougeScores(sumDocIni, refDoc, tClean),
                    getRougeScores(sumDocFinal, refDoc, tClean),
                    xDi.iCategory))
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
    
        
############################################################

def main():
    
    #
    # loading and pre-processing all documents 
    tClean = TClean()
    rB = RegBERT(tClean)       
    xDocuments = rB.loadDocuments()
    #
    #        
    preformSummaryForMaxScoreMeasurement(xDocuments, tClean)
    measureROUGE(xDocuments, tClean, 'x')
    #preformSummary(xDocuments, tClean)
    #measureROUGE(xDocuments, tClean)
    #
    #    
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


