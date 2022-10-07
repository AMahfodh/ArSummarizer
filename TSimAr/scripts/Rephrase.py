# -*- coding: utf-8 -*-
"""
Created on Sat May 28 07:34:16 2022

@author: Amma
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline
from time import perf_counter





#pModels = "UBC-NLP-AraT5"
#pModel="Facebook-mBART-large-50"
#pModel="google-mt5-base"
#pModel="arabic-t5"
#pModel="arabic-t5-small"

pModels = ['UBC-NLP-AraT5','Facebook-mBART-large-50','google-mt5-base','arabic-t5','arabic-t5-small']






def readTxtFile (strPath):
    with open(strPath, 'r', encoding="utf-8") as file:
        return file.read().replace("\n", " ")


def dsLoad(ins=498):    
    refTexts, hSimTexts = [],[]    
    for i in range(1, ins):
        refTexts.append(readTxtFile("../resources/Simplification_Datasets/References/{}.txt".format(i)))
        hSimTexts.append(readTxtFile("../resources/Simplification_Datasets/hSimplification/{}.txt".format(i)))          
    return refTexts, hSimTexts
    



def load_ptModel(strModelName):
    #  
    tokenizer = AutoTokenizer.from_pretrained("..\ptModels\{}".format(strModelName))
    model = AutoModelForSeq2SeqLM.from_pretrained("..\ptModels\{}".format(strModelName))
    model.eval()
    pipeline = Text2TextGenerationPipeline(model, tokenizer)
    return pipeline






refTexts, hSimTexts =   dsLoad()


for pM in pModels:

    t1_start = perf_counter()
    pipeline = load_ptModel(pM)
    
    icount=0
    
    for i in range(len(refTexts)):    
        #
        #
        resultFile = open("../evaluation/Rephrase/{}/{}.txt".format(pM, i+1), "w", encoding="utf-8")
        resultFile.write("{}".format(pipeline(refTexts[i])[0]['generated_text']))
        resultFile.close()  
        #
        icount+=1
        print("\r", "Progress {:2.1%}".format(icount / len(refTexts)), end="") 
    
    
    t1_stop = perf_counter()
    print ("\t ", pM, "\t e-time: ", (t1_stop-t1_start), )


print ("Done")







