"""
Created on Sat Aug  8 06:40:54 2020
@author: Abdullah M. Alshanqiti (a.m.alshanqiti@gmail.com)
"""
from summarizer import Summarizer as arDBertSummarizer
##
##
##
##
##
##
##
##
############################################################

class ArDistilBert():


    def __init__(self):        
       self.cleanXDocument = ""
       self.arDBS=arDBertSummarizer()
       self.sumRatio=0.75
    
    #
    #
    #
        
    def getSummaryFromOriginal(self, xDi): 

        #
        # get distiBert summary ...
        getDocToSum = xDi.orgDocument
        return self.arDBS(getDocToSum, ratio=self.sumRatio)
    
    #
    #
    #
        
    def getSummaryFromCleanedSentences(self, xDi):

        #
        #        
        txtFromIniSum=False
        applySentSeg=False
        #
        # get distiBert summary ...
        getDocToSum = xDi.getDoc(txtFromIniSum, applySentSeg)        
        return self.arDBS(getDocToSum, ratio=self.sumRatio)
   
    #
    #
    #
    
    def getSummaryFromIniSum(self, xDi):

        #
        #        
        txtFromIniSum=False
        applySentSeg=True
        #
        # get distiBert summary ...
        getDocToSum = xDi.getDoc(txtFromIniSum, applySentSeg)        
        return self.arDBS(getDocToSum, ratio=self.sumRatio)



############################################################

##
##
##
##
##      End of Class DistillBert
##
##
##
##

############################################################

