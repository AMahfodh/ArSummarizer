"""
Created on Sat Aug  8 06:40:54 2020
@author: Abdullah M. Alshanqiti (a.m.alshanqiti@gmail.com)
"""

import math 

##
##
##
##
##
##
##
##
###############################################################################    
   
class Docu():
    
    def __init__(self, 
                 IID, 
                 OrgDocument, 
                 iDocCategory,
                 TXTClean):
        #
        #
        self.iID = IID
        self.orgDocument = OrgDocument
        self.iCategory = iDocCategory[IID]
        self.tClean = TXTClean
        self.nSentences  = self.tClean.getSentTokenization(
            self.tClean.softCleaning(
                self.orgDocument))        
        #
        #
        self.iniSummary = None
        self.finalSummary = None
        self.nWords = 0
        self.dWords = {}
        self.dNoNExistRefTokens = {}
        self.minTF=1.1
        self.maxIDF =.0
        self.calculateWordsStats()

    #
    #
    #
    
    def calculateWordsStats(self):
             
        self.nWords = 0
        self.dWords = {}
        self.dNoNExistRefTokens = {}
        #
        #        
        for strSent in self.nSentences:
            #
            getTokens = self.tClean.getWTokens(strSent)
            #
            for sToken in getTokens:
                #
                if len(sToken)<1:
                    continue
                #
                self.nWords+=1
                #
                # compute the token counts for (IF= count/len(x)) for each token ...     
                #                       
                TF_IDF = 1
                if sToken in self.dWords:
                    TF_IDF = self.dWords[sToken] + 1
                            
                self.dWords[sToken] = TF_IDF
            #
            #
        '''    
        #
        #
        #
        # Add all tokens from ref-docs for computing the upper bound nGrameOverlapping
        for ref in ['A','B','C','D','E']:
            getWTokens, _ = self.tClean.getWordTokenWithRemovingStopOfWords(
                self.tClean.readTxtFile("../resources/EASCDataSet/ins{}/{}.txt".format(self.iID, ref)), True)
            
            for wToken in getWTokens:
                self.tClean.add_if_key_not_exist(self.dNoNExistRefTokens, wToken, None)
        '''
        #
        #        
        # Compute Freq and Score for each token ..
        for wTok in self.dWords:
            self.dWords[wTok] = self.getTF_IDF(wTok, self.dWords[wTok])
    
    #
    #
    #

    def getTF_IDF(self, wTok, wTokCount):
        
        TF = self.tClean.toRound(wTokCount/self.nWords, 8)
        
        IDF=1
        for strSent in self.nSentences:
            if wTok in strSent:
                IDF+=1
        
        IDF = self.tClean.toRound(math.log(len(self.nSentences)/IDF), 8)      
        
        #
        #
        
        if self.minTF > TF:
            self.minTF=TF
        if self.maxIDF < IDF:
            self.maxIDF = IDF        
        #
        #
        
        if wTok in self.dNoNExistRefTokens:
            self.dNoNExistRefTokens.pop(wTok)
            return {"TF" : TF, "IDF": IDF, "overlap" : 1}
                        
        return {"TF" : TF, "IDF": IDF, "overlap" : None}
    
    #
    #
    #

    def toPrintTF_IDF(self):
        for wTokKey, wTokValues in self.dWords.items():
            print ("Token-{} \t TF-{} \t IDF-{}".format(wTokKey, wTokValues['TF'], wTokValues['IDF']))           
    
    #
    #
    #

    def getTotalTF_IDF(self, strText):
        totalTF = 0
        totalIDF = 0
        wTokens = self.tClean.getWTokens(strText)
        for wT in wTokens:
            if wT not in self.dWords:
                print ('Err: token {} not found!'.format(wT))
                continue
            totalTF += self.dWords[wT]['TF']
            totalIDF +=self.dWords[wT]['IDF']
        return "{},{},{},{},{}".format(
            len(self.dWords), 
            len(strText),
            self.tClean.toRound(len(strText)/len(self.dWords),2),
            self.tClean.toRound(totalTF,2), 
            self.tClean.toRound(totalIDF,2))


    #
    #
    #

    def getDStatsTF_IDF(self, strText):
        totalTF = 0
        totalIDF = 0
        wTokens = self.tClean.getWTokens(strText)
        for wT in wTokens:
            if wT not in self.dWords:
                print ('Err: token {} not found!'.format(wT))
                continue
            totalTF += self.dWords[wT]['TF']
            totalIDF +=self.dWords[wT]['IDF']
            
        return len(self.dWords), len(strText), (len(strText)/len(self.dWords)), totalTF, totalIDF
    
    #
    #
    #

    def getNumberOfWords(self):
        return self.nWords
    
    #
    #
    #

    def getNumberOfDistinctWords(self):
        return len(self.dWords)
    
    #
    #
    #

    def getNumberOfSentences(self):  
        return len(self.nSentences)
    
    #
    #
    #

    def getDoc(self, isFromIniSum=False, applySentSeg=False, minSeqSentLen=20):
        
        #
        #        
        tSentences = None
        if not isFromIniSum:
            tSentences = self.nSentences
        else:
            tSentences = self.tClean.getSentTokenization(self.iniSummary)
        #
        #
        completeDoc =""        
        #
        #        
        if not applySentSeg:
            #
            #
            for strSent in tSentences:
                completeDoc += strSent + '\n'
            #   
            #
        else:
            #
            #            
            for strSent in tSentences:                
                strSentSegList = self.tClean.getSegSentTokenization(strSent)                
                #
                #
                for strSeg in strSentSegList:
                    strSeg = ' ' + strSeg.strip().replace(".", "")
                    if len(strSeg)>minSeqSentLen:
                        strSeg = strSeg + ' ' + '.' #'.\n'
                    #else:
                    #    strSeg = '' # needs to be tested ..
                        
                    completeDoc += strSeg
                #
                #
                if len(completeDoc)>0 and completeDoc[-1]!='.':
                    completeDoc += '.'

                
        return completeDoc.strip()
    #
    #
    #
    
    def printAllDocuments(self):
        print ("\n\tOrignal Document:\n", self.orgDocument)
        
        txtToPrint = self.getDoc(False, False)
        print ("\n\tsoftClean Document | isFromIniSum=None, applySentSeg=None:\n", txtToPrint, '\n')
        print ('\n', self.tClean.getSentTokenization(txtToPrint))
        
        print ("\n\thardClean Document | removeStopWord=false | applyLemmatize=false :\n", self.tClean.hardCleaning(txtToPrint, False, False), '\n')
        print ("\n\thardClean Document | removeStopWord=True | applyLemmatize=false :\n", self.tClean.hardCleaning(txtToPrint, True, False), '\n')        
        print ("\n\thardClean Document | removeStopWord=True | applyLemmatize=True :\n", self.tClean.hardCleaning(txtToPrint, True, True), '\n')        
        print ("\n\thardClean Document | removeStopWord=False | applyLemmatize=True :\n", self.tClean.hardCleaning(txtToPrint, False, True), '\n')        
        
        
        txtToPrint = self.getDoc(False, True)
        print ("\n\tsoftClean Document | isFromIniSum=None, applySentSeg=True:\n", txtToPrint, '\n')       
        print ('\n', self.tClean.getSentTokenization(txtToPrint))
        
        if self.iniSummary is not False:
            txtToPrint = self.getDoc(True, False)
            print ("\n\tsoftClean Document | isFromIniSum=Yes, applySentSeg=None:\n", txtToPrint, '\n')
            print ('\n', self.tClean.getSentTokenization(txtToPrint))
            
            txtToPrint = self.getDoc(True, True)
            print ("\n\tsoftClean Document | isFromIniSum=Yes, applySentSeg=True:\n", txtToPrint, '\n')       
            print ('\n', self.tClean.getSentTokenization(txtToPrint))
        
        print ("\n\tiniSummary Document:\n", self.iniSummary)
        print ("\n\tfinalSummary Document:\n", self.finalSummary)
    
    #
    #
    #

    def saveSummary(self, strDefPath = "../experiments/GeneratedSummaries/{}/{}_{}.txt"):
        
        # saving ini summary ...
        sumFile = open(strDefPath.format("iniSummaries", self.iID, "ini"), "w", encoding="utf-8")
        sumFile.write(self.iniSummary)
        sumFile.close()
        
        # saving final summary
        sumFile = open(strDefPath.format("finalSummaries", self.iID, "final"), "w", encoding="utf-8")
        sumFile.write(self.finalSummary)
        sumFile.close()
        


###############################################################################

#
#
#
#
#
#
## End