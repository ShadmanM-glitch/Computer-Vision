import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)*(a - b)))

def k_means(image,K):
    n_samples, n_features = image.shape
    
    random_sample = np.random.choice(n_samples, K, replace=False)
    centroids = [image[id] for id in random_sample]
    # Create clusters and then optimize as necessary
    for _ in range(iterations):
        # Assign to the closest centroids and create clusters
        clusters = [[] for _ in range(K)]
        for id, sample in enumerate(image):
            distances = [euclidean_distance(sample, point) for point in centroids]
            centroid_id = np.argmin(distances)
            clusters[centroid_id].append(id)       
        # Find centroids from the clusters
        centroids_old = centroids
        centroids = np.zeros((K, n_features))
        for cluster_id, cluster in enumerate(clusters):
            cluster_mean = np.mean(image[cluster], axis=0)
            centroids[cluster_id] = cluster_mean        
        # Check if clusters have changed position
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(K)]
        if (sum(distances) == 0):
            break
            
    # Distinguish data based on the cluster index
    labels = np.empty(n_samples)
    for cluster_id, cluster in enumerate(clusters):
        for sample_index in cluster:
            labels[sample_index] = cluster_id
            
    return centroids, labels

#########################################################################
iterations = 10 #maximum iterations set to reduce output time
segmented_image = [0]*8 #defining empty list
image = plt.imread("E:\\CSCI 4261\\Practicum 2\\sample2.jpg")

pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
count = 0

centroids = []
# This for loop acts like a driver fn to create the final output mosaic
for K in range(5,13):
    centroids, labels = k_means(pixel_values,K) 
    centers = np.uint8(centroids)
    labels = labels.astype(int)
    labels = labels.flatten()
    segmented_image[count] = centers[labels.flatten()]
    segmented_image[count] = segmented_image[count].reshape(image.shape)
    count += 1
# Output subplots to form a mosaic
f, axarr = plt.subplots(3,3)

axarr[0, 0].imshow(image)
axarr[0, 1].imshow(segmented_image[0])
axarr[0, 2].imshow(segmented_image[1])
axarr[1, 0].imshow(segmented_image[2])
axarr[1, 1].imshow(segmented_image[3])
axarr[1, 2].imshow(segmented_image[4])
axarr[2, 0].imshow(segmented_image[5])
axarr[2, 1].imshow(segmented_image[6])
axarr[2, 2].imshow(segmented_image[7])

axarr[0,0].set_title("Original")
axarr[0,0].axis('off')
axarr[0,1].set_title("K=5")
axarr[0,1].axis('off')
axarr[0,2].set_title("K=6")
axarr[0,2].axis('off')
axarr[1,0].set_title("K=7")
axarr[1,0].axis('off')
axarr[1,1].set_title("K=8")
axarr[1,1].axis('off')
axarr[1,2].set_title("K=9")
axarr[1,2].axis('off')
axarr[2,0].set_title("K=10")
axarr[2,0].axis('off')
axarr[2,1].set_title("K=11")
axarr[2,1].axis('off')
axarr[2,2].set_title("K=12")
axarr[2,2].axis('off')
f.tight_layout()
plt.show()