
"""
Created on Tue Jan 11 21:54:12 2022

@author: Abdullah Alshanqiti
"""

from os import listdir
from os.path import isfile, join
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForMaskedLM, FillMaskPipeline
from _PunSelection import PunSelection
import re as regEx
from time import perf_counter

##
##
##
##
##
##
##
##  Main PDSS Class
############################################################

class PDSS():

    def __init__(self):
        #
        self.pun = [":", "،", "؛", "؟", "!", "."]
        self.pipeline = None
        self.dAlpha = .06           # [.1,1]
        self.dTheta = 0.01
        self.strModel='mBERT'       # mBERT | ArBERT | AraBERT | CAMeLBERT | XLM-RoBERTa 
        self.punSelection = PunSelection()

    #
    # 
    #############################################################################################################
    #############################################################################################################
    #
    #

    def readTxtFileLineByLine(self, strNameTxT):
        #
        strF = open(strNameTxT, 'r', encoding="utf-8")     
        strDoc=""
        #
        while True:
            strLine = strF.readline()
            if not strLine:
                break
            strDoc += strLine + ' '
        strF.close()
        #
        #
        return strDoc.strip()
    
    #
    #
    #

    def load_ptModel(self):
        #  
        tokenizer = AutoTokenizer.from_pretrained("..\ptModels\{}".format(self.strModel))
        model = AutoModelForMaskedLM.from_pretrained("..\ptModels\{}".format(self.strModel))
        model.eval()
        pipeline = FillMaskPipeline(model, tokenizer)
        return pipeline
    
    #
    # 
    #############################################################################################################
    #############################################################################################################
    #
    #

    def genInputText(self, refDoc):
        # Remove pun for experiment purpose!
        for getPun in self.pun:
            refDoc = refDoc.replace(getPun, ' ')
        return regEx.sub(" +", " ", refDoc)
    
    #
    #
    #

    def addPunToText(self, strDoc, genPun):
        #    
        tok = word_tokenize(strDoc)
        #iRuntimeIndex=0
        for iKeyIndex, punValue in genPun.items():
            #tok[iKeyIndex-iRuntimeIndex]= "{} {}".format(tok[iKeyIndex-iRuntimeIndex], punValue)
            tok[iKeyIndex]= "{} {}".format(tok[iKeyIndex], punValue)
            #iRuntimeIndex+=1
        #
        #
        return regEx.sub(" +", " ", ' '.join(tok))
    
    #
    #
    #
        
    def getTextQueries(self, refDoc):
        #
        refDocList = []
        tok = word_tokenize(refDoc)
        #
        #
        iMax = int(self.dAlpha * len(tok))
        #
        if iMax > len(tok):
            iMax = len(tok)         
        #
        sIndex=0
        for i in range(len(tok)-1):
            #
            # update sIndex ..
            if i >= float(iMax/2.0) and (sIndex + iMax< len(tok)):
                sIndex+=1
            eIndex = sIndex + iMax
            #
            strQuery=""
            for j in range(sIndex, eIndex):            
                strQuery += tok[j] + ' '
                if i==j:
                    strQuery += ' [MASK] '
            refDocList.append(strQuery)
        #
        #
        return refDocList, tok
    
    #
    #
    #
    
    def getPrediction(self, strInput):
        #
        if self.pipeline is None:
            self.pipeline = self.load_ptModel()
        #        
        results = self.pipeline(strInput)
        #
        #
        token_str_List = [r['token_str'] for r in results]
        score_List = [r['score'] for r in results]
        #
        #
        for i in range(len(token_str_List)):
            if score_List[i]>= self.dTheta and any(x in token_str_List[i] for x in self.pun):
                return token_str_List[i], score_List[i]
        return None, None
    
    
    #
    #
    #
    #  
    ## 1 - Generate text queries with max alph size 512 for spaces between two sequential tokens: xQueries
    ## 2-  In a loop, get predictions/scores from a passed pre-trained model : ptModel
    ## 3-  Verify all predictions using linguistic rules, defined in the class greedy selection    
    def getPDSS(self, refToInputDoc):
        #
        refToInputDoc = self.genInputText(refToInputDoc)
        xQueries, tok = self.getTextQueries(refToInputDoc)
        #
        #
        predPun = {}
        iIndex=0        
        for xQuery in xQueries:
            #
            pu, puScore = self.getPrediction(xQuery)
            if pu!=None:
                # -1 Not verified | 0 Exclude  | 1 Add                
                predPun[iIndex]=[pu, puScore, -1]
            iIndex+=1
        #
        predPun[iIndex]=['.', 0.99, -1]
        #
        #
        genPun = self.punSelection.getPun(refToInputDoc, tok, predPun) 
        #print ('predPun=', predPun, '\n\t genPun =', genPun)
        return genPun, self.addPunToText(refToInputDoc, genPun)
    
    
    
    

    #
    # Evaluation
    #############################################################################################################
    #############################################################################################################
    #
    #
    
    def getPunctuationIndices (self, strDoc):
        # index is id, and val is punctuation         
        #
        punIndices={}    
        tok = word_tokenize(strDoc)
        #
        #
        iKey =0    
        for i in range(len(tok)):
            strToken = tok[i].strip()
            if len(strToken)==1 and any(x in strToken for x in self.pun):
                punIndices[(iKey-1)+.5]=strToken
            else:
                iKey+=1
        #
        #   
        return punIndices
    
    #
    #
    #
    
    def getScores (self, refDoc, genDoc):
        #
        #        
        refPun = self.getPunctuationIndices (refDoc)          
        genPun = self.getPunctuationIndices (genDoc)
        #
        TP=0
        FP=0
        ## Match indices for positive cases
        for keyRef, valueRef in refPun.items():        
            if keyRef in genPun.keys():
                if valueRef==genPun[keyRef]:
                    TP+=1
                else:
                    FP+=1
        #
        #
        # Negative cases: total matched indices = (TP + FP)
        FN = len(refPun) - TP - FP
        TN = len(genPun) - TP - FP
        #
        #
        return TP, FP, TN, FN
    
    
    #
    #
    #############################################################################################################
    #############################################################################################################
    #
    #
    
    def mainPDSS(self):
        #    
        #
        #  Note: we've to options to experiment: Sorted vs. Normalized | or 'BenchmarkingDS'
        strFolderDocName = "../evaluation/BenchmarkingDS/"
        docListNames = [f for f in listdir(strFolderDocName) if isfile(join(strFolderDocName, f))]
        #
        #
        iCount=0
        with open("../evaluation/eval.csv",'w') as statsCSV:
            #
            #
            for sDocName in docListNames:                        
                #
                refDoc = self.readTxtFileLineByLine ("{}{}".format(strFolderDocName, sDocName))            
                genPun, genText = self.getPDSS(refDoc)
                #
                #
                genF = open("../evaluation/GeneratedDocuments/{}".format(sDocName), 'w', encoding="utf-8")
                genF.write(genText)
                genF.close()
                #
                #
                TP, FP, TN, FN = self.getScores (refDoc, genText)
                docID = sDocName.replace('Doc_', '').replace('.txt', '')
                statsCSV.write("{}\t{}\t{}\t{}\t{}\n".format(docID, TP, FP, TN, FN))
                #
                #
                print (".", end='')
                iCount+=1
                if iCount % 30 == 0:
                    print(". ", iCount)
            print(". ", iCount-1)
                
        
        return "mainPDSS .. done\n"
    




# test call ..
############################################################


    
def main():
    #
    t1_start = perf_counter()
    pdd = PDSS()
    
    print (pdd.mainPDSS())
    
    '''
    refDoc = pdd.readTxtFileLineByLine ("../evaluation/BenchmarkingDS/Doc_10859.txt")
    #refDoc = pdd.readTxtFileLineByLine ("../resources/PDSS_Corpora/Normalized/Doc_1.txt")        
    genPun, genText = pdd.getPDSS(refDoc)
    print ("refDoc", '\n', refDoc)
    print ("genPun", '\n', genPun)
    print ("genText", '\n', genText)
    '''
    t1_stop = perf_counter()
    print ("Done .. exe time: \t", (t1_stop-t1_start))
    
if __name__ == "__main__":
    main()   
    
    
    
    
    
    
    
    
    
    
    
    
