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
        # Health
        self.iDocCategory[1]=1
        self.iDocCategory[2]=1
        self.iDocCategory[3]=1
        # News
        self.iDocCategory[4]=2
        self.iDocCategory[5]=2
        self.iDocCategory[6]=2
        self.iDocCategory[7]=2
        self.iDocCategory[8]=2
        # Religion
        self.iDocCategory[9]=3
        self.iDocCategory[10]=3
        self.iDocCategory[11]=3
        # Technology
        self.iDocCategory[12]=4
        self.iDocCategory[13]=4
        self.iDocCategory[14]=4
        return None
    #
    #
    #
    
    def loadDocuments(self):
    
        sentenceCount = []
        wordCount =[]
        distincWords = {}
        self.listDocuments = []
            
        strDocPath = "../experiments/GeneratedSummaries/HumanEvolution/Original/txt_files/{}.txt"
                
        #
        #        
        
        for i in range (1, 15):   # (1, 154)
            
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
##  Summarization functions 

############################################################


    
def preformSummary(xDocuments, tClean):
        
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
        xDi.iniSummary = arBert.getSummaryFromCleanedSentences(xDi)
        xDi.finalSummary = arBert.getSummaryFromIniSum(xDi)
        #
        #        
        xDi.saveSummary("../experiments/GeneratedSummaries/HumanEvolution/{}/{}_{}.txt")        
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



        
############################################################

def main():
    
    #
    # Loading and pre-processing all documents 
    tClean = TClean()
    arSum = ArDBertSum(tClean)       
    xDocuments = arSum.loadDocuments()       
    #    
    #
    # Summarization functions
    preformSummary(xDocuments, tClean)    
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


