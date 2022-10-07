"""
Created on Fri May 20 07:05:34 2022

@author: Abdullah Alshanqiti
"""

from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tagger.default import DefaultTagger

##
##
##
##
##
##
##
##  Best Subsect Selection Class
############################################################

class PunSelection():

    def __init__(self):
        #
        #
        self.posTagger = DefaultTagger(MLEDisambiguator.pretrained(), 'pos')
        return

    #
    # 
    #############################################################################################################
    #############################################################################################################
    #
    #
    
    def rule1(self, strSeq):
        if len(strSeq)>=2:
            return True
        return False
    
    #
    #
    #
    
    def rule2(self, SeqNounCount):
        # check number of pos-noun ..
        if SeqNounCount>=2:
            return True
        return False
    
    #
    #
    #
        
    def rule3(self, SeqNounCount, SeqVerbCount):
        if SeqNounCount>=1 and SeqVerbCount >=1:
            return True
        return False
    
    #
    #
    #
        
    def rule4(self, predPunV, strSA):
        if predPunV=='.' and len(strSA) == 0:            
            return True
        return False
    
    #
    #
    #
        
    def linguisticValidation (self, SB, iBestPun, SA, posTokens, predPunV):
        #
        strSB =[]
        strSA =[]
        #
        for i in range(SB, SA):
            if i<=iBestPun:
                strSB.append(posTokens[i]) 
            else:
                strSA.append(posTokens[i])
        #
        #
        # Check SB: segment before
        SB_noun = strSB.count('noun')
        SB_verb = strSB.count('verb')
        #
        segmentValidation = (self.rule1(strSB) and (self.rule2(SB_noun) or self.rule3(SB_noun, SB_verb)))
        if not segmentValidation:
            return False
        #
        #
        #
        # Check SA: segment before
        SA_noun = strSA.count('noun')
        SA_verb = strSA.count('verb')
        #
        segmentValidation = ((self.rule1(strSA) and (self.rule2(SA_noun) or self.rule3(SA_noun, SA_verb))) or self.rule4(predPunV, strSA))
        if not segmentValidation:
            return False
        #
        return True
    #
    # 
    #############################################################################################################
    #############################################################################################################
    #
    #
    
    def get_SB_SA(self, PredPun):
        
        SBIndex=0
        SAIndex=0
        #
        # Find the best to pick
        bestKeyPun=None
        dMaxScore=-1
        for keyPun, valuePun in PredPun.items():
            SAIndex = keyPun
            if valuePun[2]==-1 and dMaxScore<valuePun[1]:
                bestKeyPun = keyPun
                dMaxScore=valuePun[1]
        #
        #
        if bestKeyPun==None:
            return None, None, None
        #
        #
        # Surrounding segments
        for keyPun, valuePun in PredPun.items():
            if valuePun[2]!=1:
                continue
                        
            if keyPun<bestKeyPun:
                SBIndex = keyPun
            
            # valid only once
            if keyPun>bestKeyPun and SAIndex>keyPun:
                SAIndex=keyPun

            
        return bestKeyPun, SBIndex, SAIndex
        
    
    
    def getPun(self, sDoc, wTokens, PredPun):
        
        posTokens = self.posTagger.tag(wTokens)
            
        
        for iTime in range (len(wTokens)):
            
            bestKeyPun, SB, SA = self.get_SB_SA(PredPun)            
            #
            #
            #
            if bestKeyPun == None:
                break
            #
            #
            PredPun[bestKeyPun][2]=1
            if self.linguisticValidation(SB, bestKeyPun, SA, posTokens, PredPun[bestKeyPun][0]):
                PredPun[bestKeyPun][2]=1
            else:
                PredPun[bestKeyPun][2]=0
            #
            #
        
        #
        #
        #
        # Predications after applying our linguistic validation...
        genPun={}
        for keyPun, valuePun in PredPun.items():
            if valuePun[2]==1:
                genPun[keyPun]=valuePun[0]               
        
        return genPun
    
    
    
    
    
    