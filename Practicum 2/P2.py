import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
***********K-Means algorithm implementation concept***********

- Set number of clusters i.e. k
- Initialize k centroids randomly 
- Optimize centroid location in the cluster (data assigned to the closest centroid)
- Repeat last step until the centroid location no longer updates

"""

fields = ["R","G","B"]
observations = list(range(0, 250000))
    
def pre_processor(img):
    img = img.astype(np.float32)
    #rescaling to reduce the data size and speed up the algorithm
    scale_percent = 5 # adjust percentage of original image size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dimension = (width, height)  
    return img, dimension

def convert_2d(img):
    vectorised = img.reshape((-1,3))
    #Convert the array to a dataframe
    print(vectorised)
    img_df = pd.DataFrame(vectorised)
    img_df.rename(columns={0:'R', 1:'G', 2: 'B'}, inplace =True)
    return img_df


def get_random_centroids(data, k ):
    
    #return random samples from the dataset
    centroids = data.sample(k)
    print(centroids)
    return centroids
    

def k_means_fit(data,centroids, k):
    #get a copy of the original data
    
    diff = 1
    j=0

    while(abs(diff)>0.05):
        data_cpy=data
        i=1
        
        #iterate over each centroid point 
        for index1,row_c in centroids.iterrows():
            ED=[]
            #iterate over each data point
            print("Calculating distance")
            for index2,row_d in tqdm(data_cpy.iterrows()):
                #calculate distance between current point and centroid
                d1=(row_c["R"]-row_d["R"])**2
                d2=(row_c["G"]-row_d["G"])**2
                d3=(row_c["B"]-row_d["B"])**2
                d=np.sqrt(d1+d2+d3)
                #append disstance in a list 'ED'
                ED.append(d)
            #append distace for a centroid in original data frame
            data[i]=ED
            i=i+1

        C=[]
        print("Getting Centroid")
        for index,row in tqdm(data.iterrows()):
            #get distance from centroid of current data point
            min_dist=row[1]
            pos=1
            #loop to locate the closest centroid to current point
            for i in range(k):
                #if current distance is greater than that of other centroids
                if row[i+1] < min_dist:
                    #the smaller distanc becomes the minimum distance 
                    min_dist = row[i+1]
                    pos=i+1
            C.append(pos)
        #assigning the closest cluster to each data point
        data["Cluster"]=C
        #grouping each cluster by their mean value to create new centroids
        centroids_new = data.groupby(["Cluster"]).mean()[["R","G", "B"]]
        if j == 0:
            diff=1
            j=j+1
        else:
            #check if there is a difference between old and new centroids
            diff = (centroids_new['R'] - centroids['R']).sum() + (centroids_new['G'] - centroids['G']).sum() + (centroids_new['B'] - centroids['B']).sum()
            print(diff.sum())
        centroids = data.groupby(["Cluster"]).mean()[["R","G","B"]]
        return (data, centroids)
        

k = [3,5]
image = plt.imread("E:\\CSCI 4261\\Practicum 2\\sample.jpg")/255
segmented_image = [None,None]
cnt =0
img, dimension = pre_processor(image)


img_2d = convert_2d(img)

while cnt < 2:
    centroids = get_random_centroids(img_2d, k[cnt])
    cluster, centroids = k_means_fit(img_2d, centroids,k[cnt])
    centroids = centroids.to_numpy()
    labels = img_2d["Cluster"].to_numpy()
    print(centroids)
    
    segmented_image[cnt] = centroids[labels-1]
    segmented_image[cnt] = segmented_image[cnt].reshape(img.shape)
    cnt+=1



#plotting the image
plt.figure()
f, axarr = plt.subplots(1,3)
plt.imshow(image)
plt.imshow(segmented_image[0])
plt.imshow(segmented_image[1])

plt.show()

