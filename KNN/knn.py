import numpy as np 
import heapq

class KNN:
    
    def __init__(self, dataset, labels, k, distType):

        if(callable(dataset)):
            self.dataset = np.array(dataset())
        else:
            self.dataset = np.array(dataset)
        self.labels = np.array(labels)
        self.k = k
        self.distType = distType

    def get_top1(self, k_neighbors):

        stats = {}

        for neighbor in k_neighbors:

            label = neighbor['label']
            
            dist = neighbor['dist']

            if label not in stats:
                stats[label] = { 'category_dist': 0, 'freq': 0 }

            stats[label]['category_dist'] += dist
            stats[label]['freq'] += 1
            
        freqs = [class_stats['freq'] for class_stats in stats.values() ]
        
        max_freq = max(freqs)

        top1_categories_stats = []
        top1_labels = []
        top1_category_distances = []

        # Just in case there is a draw

        for label, category_stats in stats.items():

            freqs.append(category_stats['freq'])

            if category_stats['freq'] == max_freq:
                top1_categories_stats.append(category_stats)
                top1_labels.append(label)
                top1_category_distances.append(category_stats['category_dist'])
        

        if len(top1_categories_stats) > 1:

            top1_neighbors_distances_sum = sum(top1_category_distances)

            if top1_neighbors_distances_sum > 0:
                weights = top1_category_distances / top1_neighbors_distances_sum
                weights = weights[::-1]

                for weight, top1_category, top1_label in zip(weights, top1_categories_stats, top1_labels):
                    print(f"The category {top1_label} is in 1st place and has a dist sum of: {top1_category['category_dist']:.2f},\n so we assigned the weight {(weight * 100):.2f}%\n")

            highest_class = np.random.choice(top1_labels, p=weights)
            prob = max_freq / len(k_neighbors)

        else:
            highest_class = top1_labels[0]

        return highest_class, max_freq, prob, weights
            
    
    def classify(self, input):


        match self.distType:
            case "l1":
                dists = np.sum(np.abs(self.dataset - input), axis=1)
            case "l2":
                dists = np.linalg.norm(self.dataset - input, axis=1)
            case _:
                raise ValueError("Invalid distance type! It should be l1 or l2!")
            
        # top_k_indices = np.argsort(dists)[0:self.k]

        # print(top_k_indices)

        # unique_labels, freqs = np.unique(self.labels[top_k_indices], return_counts=True)

        # max_freq = np.max(freqs)

        # max_freq_idx = np.argmax(freqs)

        # prob = max_freq / np.sum(freqs)

        # best_label = unique_labels[max_freq_idx]

        k_neighbors = [{'dist': dists[i], 'label': self.labels[i]} for i in range(self.k)]

        highest_class, max_freq, prob, weights = self.get_top1(k_neighbors)

        top1_dict = {
            'highest_class': highest_class,
            'max_freq': max_freq,
            'prob': prob, 
            'weights': weights

        }

        return {
            'top1': top1_dict,
            'k_neighbors': k_neighbors
        }

        

