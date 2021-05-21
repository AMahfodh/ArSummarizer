"""
Created on Sat Aug  8 06:40:54 2020
@author: Abdullah M. Alshanqiti (a.m.alshanqiti@gmail.com)
"""

import numpy as np
import re as regEx
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer


##
##
##
##
##
##
##
##
############################################################

class TClean():


    def __init__(self):
        
        self.CuratedList = self.loadCuratedList()
        self.stop_words = set(stopwords.words('arabic'))
        self.arStemmer = Analyzer(MorphologyDB.builtin_db())
        self.sentSegRegexPattern = self.loadSentSegmentationList()        
        self.DotChar = '_'
        
    #
    #
    #

    def loadCuratedList(self):
          
        curatedFile = open('../resources/CuratedList.txt', 'r', encoding="utf-8") 
        cList = {}
        while True:       
            strLine = curatedFile.readline()             
            if not strLine: 
                break
            strKeyVal = strLine.replace('\n', '').split(":::")
            self.add_if_key_not_exist(cList, strKeyVal[0], strKeyVal[1])  
        curatedFile.close()   
        
        return cList
    
    #
    #
    #
      
    def loadSentSegmentationList(self):      
        sent_segmentationFile = open('../resources/sent_segmentation_list.txt', 'r', encoding="utf-8") 
        delimiterList = []    
        while True:       
            strLine = sent_segmentationFile.readline()             
            if not strLine: 
                break
            strLine = ' ' + strLine.replace('\n', '').strip() + ' '
            delimiterList.append(strLine) 
        sent_segmentationFile.close()
        
        return '(' +  '|'.join(map(regEx.escape, delimiterList)) + ')'         

    #
    #
    #
       
    def getSentTokenization (self, strDoc):
        return sent_tokenize(strDoc)
    #
    #
    #
    def getWTokens(self, strTxt):        
        return word_tokenize(strTxt)
    #
    #
    #
    
    def getSegSentTokenization(self, strSentence, minSeqSentLen=30):
        
        if len(strSentence)<=minSeqSentLen:
            strSent =[]
            strSent.append(strSentence)
            return strSent
        return regEx.split(self.sentSegRegexPattern, strSentence)
    
    #
    #
    #
        
    def softCleaning (self, strText):
        
        #
        # Remove newline
        strText = strText.replace('\n', ' ')
        
        #
        # Remove Tashkeel
        strText = dediac_ar(strText)
        
        #
        # Clean by replacing any matched token with any item in the curated list .. 
        for incorrectToken, correctedToken in self.CuratedList.items():            
            strText = strText.replace(incorrectToken, correctedToken)
        
        #
        # fix coma and semicolon ..
        strText = self.replaceWrongComa(strText)
        
        #
        # remove extra spaces 
        strText = regEx.sub(" +", " ", strText)
        
        return strText

    #
    #
    #
    
    def hardCleaning (self, strText, removeStopWord=False, applyLemmatize=False):
        
        #
        #
        # Apply soft cleaning first
        strText = self.softCleaning(strText)
        
        #
        # Normailse 
        strText = normalize_teh_marbuta_ar(strText)   # for Alha
        strText = normalize_alef_ar(strText)          # for Alhamza
        strText = normalize_alef_maksura_ar(strText) 
        
        #
        #
        strText = self.removeNonArabicChar(strText)
        
        #
        #
        strText = self.lemmatizeAndRemoveDotFromToken(strText, removeStopWord, applyLemmatize)
        
        # Remove final sentence-dots
        #strText = strText.replace('.', ' ')
        return strText

    #
    #
    #
  

    def replaceWrongComa(self, strText):
        
        # to keep coma and semicolon 
        strText = strText.replace(",", "،").replace(";", "؛").replace("?", "؟")
        #
        # to add space for correct sepration ..
        strText = strText.replace("،", " ، ").replace("؛ ","؛ ").replace("؟ ","؟ ").replace(":", " : ").replace(".", " . ")

        return strText
    #
    #
    #
    
    def removeNonArabicChar(self, strText):
        
        # 
        # remove english and non-arabic (including special) characters 
        strText = regEx.compile('([^\n\u060C-\u064A\.:؟?])').sub(' ', strText)
        #
        # remove extra spaces 
        return regEx.sub(" +", " ", strText)
    #
    #
    #

    def lemmatizeAndRemoveDotFromToken (
            self, strDoc,
            removeStopWord=False,
            applyLemmatize=False):
        
        getTokens = word_tokenize(strDoc)        
        strDoc = ""
        
        for strToken in getTokens:
            #
            sT= strToken.strip()
            #
            # skip if it's a stop word
            if removeStopWord and sT in self.stop_words:
                continue
            #
            #
            if applyLemmatize:
                sT = self.getStemWToken(sT)                
            #
            # check Dots
            if '.' in sT and len(sT)>1:
                sT = sT.replace(".", self.DotChar)                 
            #
            #
            if len(sT)<2 and '.' not in sT:
                continue
               
            strDoc += sT + ' ' 
            
        return strDoc.strip()

    #
    #
    #

    def getStemWToken(self, wToken):
        #
        try:            
            stemObject = self.arStemmer.analyze(wToken)
            
            # Remove Tashkeel and Normailse
            strText = dediac_ar(stemObject[0]['stem'])
            strText = normalize_teh_marbuta_ar(strText)   # for Alha
            strText = normalize_alef_ar(strText)          # for Alhamza
            strText = normalize_alef_maksura_ar(strText)         
            return strText
        except:
            return wToken
       
 
        
    #
    #
    #

    def add_if_key_not_exist(self, dict_obj, key, value):
        if key not in dict_obj:
            dict_obj.update({key: value})
    
    #
    #
    #
    
    def toRound(self, dVal, iDigits =2):
        return np.round(dVal, iDigits)

    #
    #
    #

    def readTxtFile (self, strPath):
        with open(strPath, 'r', encoding="utf-8") as file:
            return file.read().replace("\n", " ")
        
        
        
        
        