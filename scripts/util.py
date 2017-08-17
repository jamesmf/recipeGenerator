# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 23:04:50 2017

@author: jmf
"""
import re

NAMEREGEX = "name:\n(.+?)\n"


class Recipe:
    
    def __init__(self, inp):
        self.ingredientStrings = []
        self.buildFromText(inp)


    def buildFromText(self, txt):
        """
        parse a recipe into its parts
        """
        ind = txt.find("name:")+6
        txt = txt[ind:]
        self.name = txt[:txt.find("\n\n")]
        ingInd = txt.find("ingredients:\n\n")+14
        txt = txt[ingInd:]
        dirInd = txt.find("directions:\n")
        ingSec = txt[:dirInd]
        for line in ingSec.split("\n"):
            if len(line) > 1:
                self.ingredientStrings.append(line.strip())
        self.directionsString = txt[dirInd+12:].strip()


class Preprocessor:
    
    def __init__(self):
        self.recipes = []


    def recipesFromFile(self, fname):
        """
        read in recipes from .txt, create Recipes from them
        """
        with open(fname, 'r') as f:
            recipeStrings = f.read().split("$$$$")

        for recipeString in recipeStrings:
            recipe = Recipe(recipeString)
            self.recipes.append(recipe)