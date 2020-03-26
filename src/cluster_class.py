import numpy as np

def EUdist(v1, v2):
    s = 0
    for i, j in zip(v1,v2):
        s += abs(i-j)**2
    return s**0.5

def Consine_dissimilarity(v1, v2):
    upp = np.dot(v1,v2)
    lower = (np.sum(np.square(v1)) * np.sum(np.square(v1)))**0.5
    return 1-upp/lower

def dissimilarity(clusters):
    """Assumes clusters a list of clusters
       Returns a measure of the total dissimilarity of the
       clusters in the list"""
    totDist = 0
    for c in clusters:
        totDist += c.variability()/(len(c.examples))
    return totDist/len(clusters)


class Example(object):
    
    def __init__(self, name, features, label = None):
        #Assumes features is an array of floats
        self.name = name
        self.features = features
        self.label = label
        
    def dimensionality(self):
        return len(self.features)
    
    def getFeatures(self):
        return self.features[:]
    
    def getLabel(self):
        return self.label
    
    def getName(self):
        return self.name
    
    def distance(self, other):
        return Consine_dissimilarity(self.features, other.getFeatures())
    
    def __str__(self):
        return str(self.name) + ':' + str(self.features) + ':'+ str(self.label)

class Cluster(object):
    
    def __init__(self, examples):
        """Assumes examples a non-empty list of Examples"""
        self.examples = examples
        self.centroid = self.computeCentroid()
        
    def update(self, examples):
        """Assume examples is a non-empty list of Examples
           Replace examples; return amount centroid has changed"""
        oldCentroid = self.centroid
        self.examples = examples
        self.centroid = self.computeCentroid()
        return oldCentroid.distance(self.centroid)
    
    def computeCentroid(self):
        vals = np.array([0.0]*self.examples[0].dimensionality())
        for e in self.examples:      #compute mean
            vals += e.getFeatures()
        lower = sum(np.square(vals))**0.5
        centroid = Example('Center', vals/lower)
        return centroid

    def getCentroid(self):
        return self.centroid

    def variability(self):
        totDist = 0
        for e in self.examples:
            totDist += e.distance(self.centroid)
        return totDist
        
    def members(self):
        for e in self.examples:
            yield e

    def __str__(self):
        names = []
        for e in self.examples:
            names.append(e.getName())
        names.sort()
        result = 'Cluster with centroid '\
               + str(self.centroid.getFeatures()) + ' contains:\n  '
        for e in names:
            result = result + str(e) + ', '
        return result[:-2] #remove trailing comma and space