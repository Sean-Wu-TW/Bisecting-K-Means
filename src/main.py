import cluster_class
import random
import datetime
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class Doc(cluster_class.Example):
    pass

def Doc_of_words():
    context = []
    d = {}
    doc_dict_count = {}
    with open('train.clabel') as f:
        for index, word in enumerate(f, 1):
            d[index] = word.strip()
    with open('train.dat') as f:
        row = 0
        doc_lines = []
        for line in f:
            context = line.split()
            temp = []
            for i in range(0, len(context), 2):
                temp += [d[int(context[i])]]*(int(context[i+1]))
            de = {}
            for i in temp:
                if i not in de:
                    de[i] = 1
                else:
                    de[i] += 1
            doc_dict_count[row] = [(i, de[i])for i in de]
            # doc[row] = temp
            row += 1
            doc_lines.append(temp)
    return doc_dict_count, doc_lines

def buildSVD(doc_lines):
    inp = []
    for i in doc_lines:
        temp = ''
        for j in i:
            temp += j+' '
        inp.append(temp[:-1])
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(inp)
    svd = TruncatedSVD(n_components=100, n_iter=7)
    k = svd.fit_transform(features)
    ex = []
    c = 0
    for j in k:
        ex.append(Doc(c, j))
        c += 1
    return ex

def kmeans(examples, k=2, verbose = False, maxIter = 100):
    #Get k randomly chosen initial centroids, create cluster for each
    initialCentroids = random.sample(examples, k)
    clusters = []
    for e in initialCentroids:
        clusters.append(cluster_class.Cluster([e]))
        
    #Iterate until centroids do not change
    converged = False
    numIterations = 0
    while not converged and numIterations < maxIter:
        numIterations += 1
        #Create a list containing k distinct empty lists
        newClusters = []
        for i in range(k):
            newClusters.append([])
            
        #Associate each example with closest centroid
        for e in examples:
            #Find the centroid closest to e
            smallestDistance = e.distance(clusters[0].getCentroid())
            index = 0
            for i in range(1, k):
                distance = e.distance(clusters[i].getCentroid())
                if distance < smallestDistance:
                    smallestDistance = distance
                    index = i
            #Add e to the list of examples for appropriate cluster
            newClusters[index].append(e)
            
        for c in range(len(newClusters)): #Avoid having empty clusters
            if len(newClusters[c]) == 0:
                C = cluster_class.Cluster(newClusters[0])
                largestVar = float('Inf')
                ex_i = 0
                for j in range(len(newClusters[0].examples)):
                    if j.distance(C.getCentroid()) < largestVar:
                        ex_i = j
                        largestVar = j.distance(C.getCentroid())
                newClusters[c].append(newClusters[0].pop(ex_i))
                    
        #Update each cluster; check if a centroid has changed
        converged = True
        for i in range(k):
            if clusters[i].update(newClusters[i]) > 0.0:
                converged = False
        if verbose:
            print('Iteration #' + str(numIterations))
            for c in clusters:
                print(c)
            print('') #add blank line
    return clusters


def bisecting_kmeans(clusters, v = False):
    print("Start bisecting_kmeans:")
    # start with one big cluster
    clusterCount = 1
    # loop until 7 clusters
    while clusterCount < 7:
        # find the cluster with maximum variability
        li = [c.variability() for c in clusters]
        print(cluster_class.dissimilarity(clusters))
        index =  li.index(max(li))
        BigC = clusters.pop(index)
        # divide the cluster with maximum variability into two subClusters
        print('Working kmeans with',str(clusterCount),'cluster(s)...')
        subClusters = kmeans(BigC.examples, verbose = v)
        clusters += subClusters
        clusterCount += 1
    return clusters

def write_file(clusters):
    print('Saving results..')
    dict_ = {}
    c = 1
    for cluster in clusters:
        for example in cluster.examples:
            dict_[example.name] = c
        c += 1
    dirName = 'output'
    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    file_name = "answer" + "_" + str(month) + "-" + \
                 str(day) + "-" + str(hour) + "-" + str(minute) + ".csv"
    path = os.path.join("output",file_name)
    with open(path, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['itemid','clusterid'])
        for i in sorted(dict_.keys()):
            writer.writerow((i+1,dict_[i]))

if __name__ == '__main__':
    
    doc_dict, doc_lines = Doc_of_words()
    examples = buildSVD(doc_lines)
    clusters = bisecting_kmeans([cluster_class.Cluster(examples)])
    write_file(clusters)
