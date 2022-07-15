# -*- coding: utf-8 -*-
"""
Created on Fri May 20 07:05:34 2022

@author: Abdullah Alshanqiti
"""
from os import listdir
from os.path import isfile, join
from nltk.tokenize import word_tokenize

#
#
#
pun = [":", "،", "؛", "؟", "!"]
#
#
#


def getPunctuationIndices (strDoc):
    # index is id, and val is punctuation         
    #
    punIndices={}    
    tok = word_tokenize(strDoc)
    #
    #
    for i in range(len(tok)):
        strToken = tok[i].strip()
        if len(strToken)==1 and any(x in strToken for x in pun):
            punIndices[i]=tok[i].strip()
    #
    #
    print (punIndices)
    return punIndices




def getScores (refDoc, genDoc, genPun=None):
    #
    #
    refPun = getPunctuationIndices (refDoc)
    if genPun==None:
        genPun = getPunctuationIndices (genDoc)
    #
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
#


def readTxtFileLineByLine(strNameTxT):
    #
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
    return strDoc


#
#
#


def evaluateAllDocs():
    #    
    strFolderDocName = "../evaluation/GeneratedDocuments/"
    docListNames = [f for f in listdir(strFolderDocName) if isfile(join(strFolderDocName, f))]
    #
    #
    iCount=0
    with open("../evaluation/eval.csv",'w') as statsCSV:
        #
        #
        for sDoc in docListNames:                        
            #
            # We have to options: Sorted vs. Normalized 
            refDoc = readTxtFileLineByLine ("../resources/PDSS_Corpora/Normalized/{}".format(sDoc))
            genDoc = readTxtFileLineByLine ("{}{}".format(strFolderDocName, sDoc))
            #
            #
            docID = sDoc.replace('Doc_', '').replace('.txt', '')
            TP, FP, TN, FN = getScores (refDoc, genDoc)
            #
            #
            statsCSV.write("{}\t{}\t{}\t{}\t{}\n".format(docID, TP, FP, TN, FN))
            #
            #
            print (".", end='')
            iCount+=1
            if iCount % 30 == 0:
                print(". ", iCount)
        print(". ", iCount-1)
            
    
    return "evaluateAllDocs_Done\n"
    



print (evaluateAllDocs())





print ("Eval... done.")