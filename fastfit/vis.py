#!/usr/bin/env python
# coding: utf-8

# In[108]:


import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import seaborn as sns
from matplotlib.colors import to_rgb, to_hex, rgb_to_hsv, hsv_to_rgb
from collections import defaultdict
import matplotlib.colors as mcolors
from fire import Fire

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    sentences = [item['text'] for item in data]
    labels = [','.join(item['label'].split(',')[:2]) for item in data]
    return sentences, labels

def encode_sentences(sentences, model_name):
    model = SentenceTransformer(model_name)
    return model.encode(sentences)

def reduce_dimensions(embeddings, perplexity=100):
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    pca = PCA(n_components=2)
    return tsne.fit_transform(embeddings), pca.fit_transform(embeddings), pca.explained_variance_ratio_

def 

def create_color_map(labels):
    unique_labels = list(set(labels))
    return dict(zip(unique_labels, sns.color_palette("husl", len(unique_labels))))

def plot_embeddings(tsne_results, pca_results, labels, color_map):
    plt.style.use('fivethirtyeight')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Sentence Embeddings Visualization', fontsize=16)

    for ax, results, title in [(ax1, tsne_results, 't-SNE Visualization'), 
                               (ax2, pca_results, 'PCA Visualization')]:
        for label in color_map:
            mask = np.array(labels) == label
            ax.scatter(results[mask, 0], results[mask, 1], c=[color_map[label]], label=label, alpha=0.7)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f'{title.split()[0]} 1', fontsize=12)
        ax.set_ylabel(f'{title.split()[0]} 2', fontsize=12)
        ax.legend(title='Classes', title_fontsize='13', fontsize='10', loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

def show_embeddings(dataset_path):
    sentences, labels = load_data(dataset_path)
    embeddings = encode_sentences(sentences, 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    tsne_results, pca_results, explained_variance_ratio = reduce_dimensions(embeddings)
    color_map = create_color_map(labels)
    plot_embeddings(tsne_results, pca_results, labels, color_map)
    print(f"Explained variance ratio: {explained_variance_ratio}")


# In[91]:





# In[92]:


def generate_hierarchical_colors(labels):
    color_map = {}
    primary_types = set(label.split(',')[0] for label in labels)
    secondary_types = defaultdict(set)
    tertiary_types = defaultdict(set)
    
    for label in labels:
        parts = label.split(',')
        if len(parts) > 1:
            secondary_types[parts[0]].add(parts[1])
        if len(parts) > 2:
            tertiary_types[tuple(parts[:2])].add(parts[2])

    # Generate primary colors
    primary_colors = plt.cm.hsv(np.linspace(0, 1, len(primary_types)))
    primary_color_map = dict(zip(primary_types, primary_colors))
    
    for label in set(labels):
        parts = label.split(',')
        base_color = primary_color_map[parts[0]]
        
        if len(parts) == 1:
            color_map[label] = base_color
        elif len(parts) == 2:
            h, s, v = rgb_to_hsv((base_color[:3]))
            color_map[label] = hsv_to_rgb((h, s * 0.8, v * 0.9))
        else:
            h, s, v = rgb_to_hsv((base_color[:3]))
            secondary_count = len(secondary_types[parts[0]])
            secondary_index = list(secondary_types[parts[0]]).index(parts[1])
            tertiary_count = len(tertiary_types[tuple(parts[:2])])
            tertiary_index = list(tertiary_types[tuple(parts[:2])]).index(parts[2])
            
            # Adjust hue based on secondary and tertiary levels
            hue_shift = (secondary_index / secondary_count) * 0.2
            hue_shift += (tertiary_index / tertiary_count) * 0.1
            new_h = (h + hue_shift) % 1.0
            
            # Adjust saturation and value
            new_s = s * (0.6 + 0.4 * (tertiary_index / tertiary_count))
            new_v = v * (0.7 + 0.3 * ((tertiary_count - tertiary_index) / tertiary_count))
            
            color_map[label] = hsv_to_rgb((new_h, new_s, new_v))
    
    return {k: to_hex(v) for k, v in color_map.items()}


# In[93]:





# In[94]:


def print_color_map(color_map):
    def rgb_to_ansi(r, g, b):
        if r == g == b:
            if r < 8:
                return 16
            if r > 248:
                return 231
            return round(((r - 8) / 247) * 24) + 232
        
        ansi = 16 + (36 * round(r / 255 * 5)) + (6 * round(g / 255 * 5)) + round(b / 255 * 5)
        return ansi

    for class_name, color_hex in color_map.items():
        class_name = class_name.strip('"')
        
        # hex to RGB
        r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        
        # ANSI color code
        ansi_color = rgb_to_ansi(r, g, b)
        
        # ANSI escape sequence
        bg_color = f"\033[48;5;{ansi_color}m"
        reset = "\033[0m"
        print(f"{class_name}: {bg_color}  {reset}")


# In[97]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA

def visualize_embedding_progression(data, color_map, output_gif="embedding_progression.gif", interval=1000):
    labels = [item["label"] for item in data]
    num_embedding_sets = len(data[0]["vectors"])

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter([], [], c=[], alpha=0.7)
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    ax.legend(title="Classes", title_fontsize='10', fontsize='8', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    pca_results = []
    for i in range(num_embedding_sets):
        embeddings = [item["vectors"][i] for item in data]
        embeddings = np.array(embeddings)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings)
        pca_results.append(pca_result)

    def update(frame):
        ax.clear()
        for label in set(labels):
            mask = np.array(labels) == label
            ax.scatter(pca_results[frame][mask, 0], pca_results[frame][mask, 1], 
                       c=[color_map[label]], label=label, alpha=0.7)
        ax.set_title(f"PCA of Embeddings (Frame {frame+1}/{num_embedding_sets})")
        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")
        ax.legend(title="Classes", title_fontsize='10', fontsize='8', loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlim(min(r[:, 0].min() for r in pca_results), max(r[:, 0].max() for r in pca_results))
        ax.set_ylim(min(r[:, 1].min() for r in pca_results), max(r[:, 1].max() for r in pca_results))
        return scatter,

    anim = FuncAnimation(fig, update, frames=num_embedding_sets, interval=interval, blit=True)

    anim.save(output_gif, writer='pillow')
    plt.close(fig)

    print(f"Animation saved as {output_gif}")


# In[110]:


def test():
    data = [
        {"text": "input 1", "label": "label1", "vectors": [[1,2,3,1], [1,3,4,1], [3,4,5,1],[1,2,3,1], [1,3,4,1], [3,4,5,1]]},
        {"text": "input 2", "label": "label2", "vectors": [[2,3,4,1], [2,4,5,1], [4,5,6,1],[2,3,4,1], [2,4,5,1], [4,5,6,1]]},
        {"text": "input 3", "label": "label3", "vectors": [[2,3,4,1], [2,4,5,1], [4,5,6,1],[2,3,4,1], [2,4,5,1], [4,5,6,1]]}
    ]
    for i in range(3):
        data[i]["vectors"] = np.array([np.random.randn(36) for _ in range(10)])
    
    color_map = {
        "label1": "#FF0000",
        "label2": "#00FF00",
        "label3": "#0FF0FF",
    }
    #color_map = generate_hierarchical_colors(labels)
    print_color_map(color_map)
    visualize_embedding_progression(data, color_map)

    labels = [l.strip() for l in open("../../labels.txt").readlines()][:30]
    plot_embeddings("../../dataset.txt")


# In[111]:


def generate_random_color_map(labels):
    """
    Generate a random color map for a given set of labels.
    
    Args:
    labels (list): A list of unique labels.
    
    Returns:
    dict: A dictionary mapping each label to a unique color in hex format.
    """
    unique_labels = list(set(labels))
    n_colors = len(unique_labels)
    
    hsv_colors = [(i / n_colors, 1, 1) for i in range(n_colors)]
    rgb_colors = [mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors]
    
    hex_colors = [mcolors.to_hex(rgb) for rgb in rgb_colors]
    
    np.random.shuffle(hex_colors)
    
    return dict(zip(unique_labels, hex_colors))


# In[112]:

def animate(embedding_history_data):
    with open(embedding_history_data) as f:
        data = [json.loads(l) for l in f.readlines()]
    labels = set([l["label"] for l in data])
    color_map = generate_random_color_map(labels)
    visualize_embedding_progression(data, color_map)


# In[113]:


if __name__ == "__main__":
    Fire(animate)


# In[ ]:




