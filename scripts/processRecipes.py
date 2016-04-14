# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:06:33 2016

@author: jmf
"""

import re
from os import listdir

spaces  = re.compile(" +")
newlines= re.compile(r"\n+")

disallowed  = ['\x89', '\xa9', '!', '#', '"', '$', "'", '&',
                '+', '*', '=', '?', '\xc3', '~','[',']','<']

#ld  = listdir("../recipes/")
#with open("../allrecipes.txt",'wb') as ar:
#    for fi in ld:
#        fn  = "../recipes/"+fi
#        with open(fn,'rb') as f:
#            text    = f.read()
#        for d in disallowed:
#            text    = text.replace(d,' ')
#        temp    = text.split("\n\n")
#        text    = "name:\n"+temp[0]+"\n\ningredients:\n"+temp[1]+"\n\ndirections:\n"+'\n'.join(temp[2:])
#        text    = text.replace(";",'').replace('@','at').replace('\t','').replace("  ",' ')
#        text    = re.sub(spaces,' ',text).replace('\n ','\n')
#        text    = text.lower()
#
#        
#        if text.find("could not open recipe") <  0:
#            ar.write(text+"$$$$\n")

pageReg     = re.compile("<page>(.+?)</page>",re.DOTALL)
with open("../Wikibooks-20160329223834.xml",'rb') as f:
    text     = f.read().decode("utf-8","replace")

pages       = re.findall(pageReg,text)
recipes     = []
for p in pages:
    if p.lower().find("procedure") > -1:
        #print p
        recipes.append(p)
        #stop=raw_input("") 

print len(p)
stop=raw_input("")

with open("../allrecipes2.txt",'wb') as ar:
    for fi in ld:
        fn  = "../recipes/"+fi
        with open(fn,'rb') as f:
            text    = f.read()
        for d in disallowed:
            text    = text.replace(d,' ')
        temp    = text.split("\n\n")
        text    = "name:\n"+temp[0]+"\n\ningredients:\n"+temp[1]+"\n\ndirections:\n"+'\n'.join(temp[2:])
        text    = text.replace(";",'').replace('@','at').replace('\t','').replace("  ",' ')
        text    = re.sub(spaces,' ',text).replace('\n ','\n')
        text    = text.lower()

        
        if text.find("could not open recipe") <  0:
            ar.write(text+"$$$$\n")
