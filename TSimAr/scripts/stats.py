# -*- coding: utf-8 -*-
"""
Created on Fri May 20 07:05:34 2022

@author: Abdullah Alshanqiti
"""

from nltk.tokenize import word_tokenize, sent_tokenize



def readTxtFile (strPath):
    with open(strPath, 'r', encoding="utf-8") as file:
        return file.read().replace("\n", " ")
    
def confToString (strList):
    return " . ".join(strList)


def dsLoad(ins=498):    
    refTexts, hSimTexts = [], []    
    for i in range(1, ins):
        refTexts.append(readTxtFile("../resources/Simplification_Datasets/References/{}.txt".format(i)))
        hSimTexts.append(readTxtFile("../resources/Simplification_Datasets/hSimplification/{}.txt".format(i)))          
    return confToString(refTexts), confToString(hSimTexts)




def add_if_key_not_exist(dict_obj, key, value):
    if key not in dict_obj:
        dict_obj.update({key: value})
        
        
def getDis(strList1, strList2=['']):
    strDis={}
    for s in strList1:
        add_if_key_not_exist(strDis, s, None)    
    for s in strList2:
        add_if_key_not_exist(strDis, s, None)           
    return len(strDis)
        
    



def printStat():
    
    refTexts, hSimTexts = dsLoad()
    
    print ("Input Complex: sent len={}".format(len(sent_tokenize(refTexts))))
    print ("Simplified Rewrites: sent len={}".format(len(sent_tokenize(hSimTexts))))
    
    wtRef = word_tokenize(refTexts)
    wtSim = word_tokenize(hSimTexts)
    
    
    print ("Input Complex: Voc len={} and dis={}".format(len(wtRef), getDis(wtRef)))
    print ("Simplified Rewrites: Voc len={} and dis={}".format(len(wtSim), getDis(wtSim)))
    
    print ("Sharid voc={}".format(getDis(wtRef, wtSim)))
    
    
    

printStat()
