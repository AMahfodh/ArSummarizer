"""
Created on Sat Aug  8 06:40:54 2020
@author: Abdullah M. Alshanqiti (a.m.alshanqiti@gmail.com)
"""

import rouge
import numpy as np
import re as regEx


##
##
##
##
##
##
##
##
############################################################

class RegExp():


    def __init__(self, XDi, tCleanObject):
        
        self.xDi = XDi
        self.tClean = tCleanObject        
        self.regexPattern = self.tClean.loadSentSegmentationList()
        #
        #
        self.dCurrentFMegure = 1.5

    #
    #
    #

    def setFinalSummary(self):        
        #
        #
        self.xDi.finalSummary = self.xDi.iniSummary
        self.xDi.nSentences.clear() 
        self.xDi.nSentences = self.tClean.getSentTokenization(self.xDi.iniSummary)
        istrSentPosition =0
        concatenateSentences =""
        for strXSent in self.xDi.nSentences:
            concatenateSentences += self.rgRules(istrSentPosition, strXSent)
            istrSentPosition +=1
        #
        #
        self.xDi.finalSummary = concatenateSentences
        return concatenateSentences

    #
    #
    #

    def rgRules(self, istrSentPosition, strXSent):
                
        '''
            - Manage set of rules and their order ..
            - rules to check number of parts, length, and relvants using Frg and score !
            - drop parts that less important
            - drop first part if number of sentences is more than 4 
            
        '''        
        partsOfSent = regEx.split(self.regexPattern, strXSent)
        xPSent = ""
        istrPartSentPosition =0
        
        for strPartSent in partsOfSent:            
            #xSent += self.rule1(iPosition, strPartSent, len(partsOfSent))
            #xPSent += self.FT_IDF_Rule(strPartSent)
                        
            xPSent += self.rogueUpperBound(
                istrSentPosition, 
                istrPartSentPosition, 
                strPartSent)
            istrPartSentPosition+=1
            
            #xPSent += self.isNGramOverlapping(strPartSent)
            
        xPSent = xPSent.strip()
        if len(xPSent)<2:
            return ""
        return xPSent + '. '
 
    #
    #
    #
       
    
    
    def FT_IDF_Rule(self, strPartSent):
        
        return strPartSent
        getWTokens = self.tClean.getWordTokenWithRemovingStopOfWords(strPartSent)
        
        if (len(getWTokens)<3):
            return ""
        
        ################################################
        iCounts = 0
        for wToken in getWTokens:
            if wToken in self.xDi.dWords:
                xT = self.xDi.dWords[wToken]
                if xT['TF']== self.xDi.minTF and xT['IDF']==self.xDi.maxIDF:
                    iCounts +=1
            #else:
                #print (wToken)
                #iCounts +=1
        ################################################    
        
        if (iCounts/len(getWTokens))>.6:
            return ""
        return strPartSent
        
    #
    #
    #
    
    def isNGramOverlapping(self, strPartSent):
        
        strPartSent = strPartSent.strip()
        #
        # TOFIX Print for tracing ...
        self.printToTrace(strPartSent)
        #
        # check if it exits in Refs
        getWTokens, _ = self.tClean.getWordTokenWithRemovingStopOfWords(strPartSent)
        
        if len(getWTokens)<1:
            return ""
                
        nGramOverLap =0
        for wToken in getWTokens:                
            if wToken in self.xDi.dWords:
                if self.xDi.dWords[wToken]['overlap'] is not None:
                    nGramOverLap +=1
        
        if nGramOverLap>=2:
            return strPartSent
        
        return ""
    
    #
    #
    #    



    def printToTrace(self, strTxt):
        
        with open("../experiments/sent_segmentation_list.csv",'a+', encoding="utf-8") as traceCSV:
            traceCSV.write("{}\t{}\n".format(
                len(self.tClean.getWTokenSize(strTxt)), 
                strTxt))
        return None



    # hard-code to identify the upper bound metric-values ..
    def rogueUpperBound(self, istrSentPosition, istrPartSentPosition, strPartSent):
        
        strAllRouge, isDroppingImprove = self.getAllRouge(strPartSent)
        '''
        with open("../experiments/tracingDS.csv",'a+', encoding="utf-8") as traceCSV:
            #traceCSV.write("DocID,SentID,SentPartID,Abase,A,Bbase,B,Cbase,C,Dbase,D,Ebase,E,Avg,avgDecision\n")
            traceCSV.write("{},{},{},{},{}\n".format(
                self.xDi.iID, 
                istrSentPosition, 
                istrPartSentPosition, 
                strAllRouge, 
                strPartSent))
        '''    
        if isDroppingImprove:
            return ""
        return strPartSent
     
    #
    #
    #
           
    def getAllRouge(self, strPartSent) :
                        
        allRouge =[]        
        strAllRouge =""
        for ref in ['A','B','C','D','E']:
            refDoc = self.xDi.tClean.readTxtFile("../resources/EASCDataSet/ins{}/{}.txt".format(self.xDi.iID, ref))
            Fbase = self.getRougeScores(self.xDi.finalSummary, refDoc)
            Fchange = self.getRougeScores(self.xDi.finalSummary.replace(strPartSent, " ", 1), refDoc)
            Fdiff = Fchange-Fbase
            
            strAllRouge += str(self.xDi.tClean.toRound(Fbase)) + ',' + str(self.xDi.tClean.toRound(Fdiff)) + ','
            allRouge.append(Fdiff)
        #
        #        
        finalMean = np.mean(allRouge)
        strAllRouge += str(self.xDi.tClean.toRound(finalMean))
        if (finalMean>0):
            self.xDi.finalSummary = self.xDi.finalSummary.replace(strPartSent, " ", 1)
            strAllRouge += ',1'
        else:
            strAllRouge += ',0'
        
        # Abase,A,Bbase,B,Cbase,C,Dbase,D,Ebase,E,Avg,avgDecision
        return strAllRouge, (finalMean>0)
 
    #
    #
    #

    def getRougeScores(self, orgDoc, refDoc):
    
        rScores = rouge.Rouge()
        scores = rScores.get_scores(orgDoc.replace("\n", ""), refDoc.replace("\n", ""))[0]  
        return scores['rouge-2']['f']
    



############################################################

##
##
##
##
##      End of Class RegExp
##
##
##
##

############################################################