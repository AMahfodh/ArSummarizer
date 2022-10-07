# -*- coding: utf-8 -*-
"""
Created on Fri May 20 07:05:34 2022

@author: Abdullah Alshanqiti
"""

from nltk.tokenize import word_tokenize, sent_tokenize
from os import listdir
from os.path import isfile, join
import shutil
import re as regEx
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar


#
#
#
pun = [":", "،", "؛", "؟", "!", "."]
#
#
#


def replaceWrongComa(strText):
    
    # to keep coma and semicolon 
    strText = strText.replace(",", "،").replace(";", "؛").replace("?", "؟")
    #
    # to add space for correct sepration ..
    strText = strText.replace("،", " ، ").replace("؛ ","؛ ").replace("؟ ","؟ ").replace(":", " : ").replace(".", " . ")

    return strText

 

def removeNonArabicChar(strText):
    
    # 
    # remove english and non-arabic (including special) characters 
    strText = regEx.compile('([^\n\u060C-\u064A\.:؟?])').sub(' ', strText)
    #
    # remove extra spaces 
    return regEx.sub(" +", " ", strText)

        
def softCleaning (strText):
    
    #
    # Remove newline and brackets
    strText = strText.replace('\n', ' ').replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ')
    strText = strText.replace('\'', ' ').replace('\\', ' ').replace('n', ' ')
    
    #
    # Remove Tashkeel and Normailse
    strText = dediac_ar(strText)
    strText = normalize_teh_marbuta_ar(strText)   # for Alha
    strText = normalize_alef_ar(strText)          # for Alhamza
    strText = normalize_alef_maksura_ar(strText) 
    #
    # fix coma and semicolon ..
    strText = replaceWrongComa(strText)
    
    #
    # remove extra spaces 
    strText = regEx.sub(" +", " ", strText)
    
    #
    # add pun to fix word tokenization
    for getPun in pun:
        strText = strText.replace(getPun, " {} ".format(getPun))
        
        
    return strText

    

def readTFile(strNameTxT, doesIncludePun=True):        
    strF = open(strNameTxT, 'r', encoding="utf-8") 
    #
    strDoc=""
    while True:
        strLine = strF.readline()
        if not strLine:
            break
        strDoc += strLine + ' '
    strF.close()
    #
    tokinzed= word_tokenize(strDoc)
    if doesIncludePun:        
        doesIncludePun = any(x in strDoc for x in pun)
    #
    return len(tokinzed), strDoc, doesIncludePun

















#
#
#
# Read from original corpora and then generate sorted-normalized versions
#
# Just call readFromOriginalCorpora()
#

def readFromOriginalCorpora():
    #
    #
    OrgDir = "../resources/PDSS_Corpora/Original/"
    docListNames = [f for f in listdir(OrgDir) if isfile(join(OrgDir, f))]
    
    
    iDocCategory = {}
    
    for sDoc in docListNames:    
        fileSize, _ , doesIncludePun = readTFile("{}{}".format(OrgDir, sDoc))
        #
        # Exclude (small) file names
        #
        if fileSize>50 and doesIncludePun:
            iDocCategory[sDoc]= fileSize    
    
    
    # sorting .. 
    iDocCategory = dict(sorted(iDocCategory.items(), key=lambda item: item[1]))
    
    
    iConter=0
    for k in iDocCategory:
        
        iConter +=1
        strSource = "{}{}".format(OrgDir, k)    
        shutil.copyfile(strSource, "../resources/PDSS_Corpora/Sorted/Doc_{}.txt".format(iConter))    
        with open("../resources/PDSS_Corpora/Normalized/Doc_{}.txt".format(iConter), 'w', encoding="utf-8") as f:
            _, strSource, _ = readTFile(strSource, False)
            f.write(softCleaning(str(strSource)))
                        
        print (iConter)
        

#### readFromOriginalCorpora()



    




#
#
#
# Generate stats. from sorted documnets to result.xsl:
#
#   Call print (generateStats('Sorted'))
#
def readStatFromFile(strNameTxT):        
    strF = open(strNameTxT, 'r', encoding="utf-8") 
    #
    strDoc=""
    while True:
        strLine = strF.readline()
        if not strLine:
            break
        strDoc += strLine + ' '
    strF.close()
    #
    #
    wTokinzed= word_tokenize(strDoc)
    sTokinzed= sent_tokenize(strDoc)
    #
    #    
    return len(wTokinzed), len(sTokinzed)


def generateStats(strFolderDocName):
    
    strFolderDocName = "../resources/PDSS_Corpora/{}/".format(strFolderDocName)
    docListNames = [f for f in listdir(strFolderDocName) if isfile(join(strFolderDocName, f))]
    #
    #
    iCount=0
    with open("../evaluation/stats.csv",'w') as statsCSV:
        #
        #
        for sDoc in docListNames:            
            docID = sDoc.replace('Doc_', '').replace('.txt', '')
            wTokinzed, sTokinzed = readStatFromFile("{}{}".format(strFolderDocName, sDoc))
            #
            # 16281 total
            L_M_H = 'H'
            if int(docID) <5427:
                L_M_H = 'L'
            elif int(docID) <10854:
                L_M_H = 'M'
            #
            #
            statsCSV.write("{}\t{}\t{}\t{}\t{}\n".format(docID, sDoc, wTokinzed, sTokinzed, L_M_H))
            #
            #
            print (".", end='')
            iCount+=1
            if iCount % 30 == 0:
                print(". ", iCount)
        print(". ", iCount-1)            
            
    
    return "generateStats .. Done"
    








print ("Done.")












'''
Not used anymore (only once)
def readTFileLineByLine(strNameTxT, prx):
    strF = open(strNameTxT, 'r', encoding="utf-8") 
    iConter=0
    while True:        
        strLine = str(strF.readline())
        if not strLine: 
                break
        
        iConter+=1
        #print (strLine)
        with open("../resources/PDSS_Corpora/genDS/{}_{}.txt".format(prx, iConter), 'w', encoding="utf-8") as f:
            f.write(strLine)
    
    strF.close()
    return None


readTFileLineByLine("../resources/PDSS_Corpora/test.txt", "test")
readTFileLineByLine("../resources/PDSS_Corpora/train.txt", "train")
readTFileLineByLine("../resources/PDSS_Corpora/val.txt", "val")
'''