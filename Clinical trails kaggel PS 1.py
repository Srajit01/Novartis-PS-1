import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sentence_transformers import SentenceTransformer
import faiss
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load the Data
def load_data(file_path):
    """
    Load clinical trials dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    
    # Strip any leading/trailing spaces from column names
    data.columns = data.columns.str.strip()
    
    # Print success message and dataset info
    print("Data Loaded Successfully!")
    print(data.columns)  # Check the column names
    
    return data

# Step 2: Data Preprocessing
def preprocess_text(text):
    """
    Basic preprocessing for text: lowercase, remove unnecessary spaces.
    """
    if pd.isna(text):
        return ""
    return text.lower().strip()

def preprocess_data(data):
    """
    Apply preprocessing to relevant text fields and combine them.
    """
    fields_to_process = ["Study Title", "Primary Outcome Measures", "Secondary Outcome Measures", "Brief Summary"]
    
    missing_fields = [field for field in fields_to_process if field not in data.columns]
    if missing_fields:
        print(f"Warning: Missing columns: {missing_fields}")
    
    for field in fields_to_process:
        if field in data.columns:
            data[field] = data[field].apply(preprocess_text)
    
    data["Combined_Text"] = data["Study Title"] + " " + \
                            data["Primary Outcome Measures"].fillna('') + " " + \
                            data["Secondary Outcome Measures"].fillna('') + " " + \
                            data["Brief Summary"].fillna('')
    
    print("Data Preprocessing Completed!")
    return data

def generate_embeddings(data, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate text embeddings using a pre-trained SentenceTransformer model.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(data["Combined_Text"].tolist(), show_progress_bar=True)
    print("Embeddings Generated!")
    return embeddings, model

# Step 4: FAISS Indexing for Similarity Search
def create_faiss_index(embeddings):
    """
    Create and populate a FAISS index with the generated embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    print("FAISS Index Created and Populated!")
    return index

# Step 5: Query Similar Trials
def query_similar_trials(query_text, model, index, data, top_k=10):
    """
    Retrieve top K similar clinical trials for a given query.
    """
    query_embedding = model.encode([query_text])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = data.iloc[indices[0]].copy()
    results["Similarity_Score"] = distances[0]
    print(f"Top {top_k} Similar Trials Retrieved!")
    return results

# Step 6: Evaluation
def compute_precision_recall(predicted_indices, actual_indices, k):
    """
    Compute Precision@K and Recall@K for a query.
    """
    predicted = set(predicted_indices[:k])
    actual = set(actual_indices)
    precision = len(predicted & actual) / len(predicted) if predicted else 0
    recall = len(predicted & actual) / len(actual) if actual else 0
    return precision, recall

# Step 7: Visualization
def plot_similarity_scores(results):
    """
    Visualize similarity scores of the retrieved trials.
    """
    sns.barplot(x=results["Similarity_Score"], y=results["Study Title"], palette="viridis")
    plt.title("Similarity Scores of Retrieved Trials")
    plt.xlabel("Similarity Score")
    plt.ylabel("Study Title")
    plt.show()

# Main Function: End-to-End Workflow
def main(file_path, output_file_path="similar_trials_output.csv"):
    # Step 1: Load the Data
    data = load_data(file_path)  # Pass file_path to load_data function
    
    # Step 2: Preprocess the Data
    data = preprocess_data(data)
    
    # Step 3: Generate Embeddings
    embeddings, model = generate_embeddings(data)
    
    # Step 4: Create FAISS Index
    index = create_faiss_index(embeddings)
    
    # Step 5: Query Example
    query = "A study about diabetes and cardiovascular outcomes"
    similar_trials = query_similar_trials(query, model, index, data, top_k=10)
    print(similar_trials[["Study Title", "Similarity_Score"]])
    
    # Step 6: Evaluate the Model (Example Placeholder)
    # Replace actual_indices with true relevant indices for your test case.
    predicted_indices = index.search(np.array(model.encode([query])), 10)[1][0]
    actual_indices = [0, 5, 10]  # Example relevant indices
    precision, recall = compute_precision_recall(predicted_indices, actual_indices, k=10)
    print(f"Precision@10: {precision}, Recall@10: {recall}")
    
    # Step 7: Visualization
    plot_similarity_scores(similar_trials)
    
    # Save the similar trials to a CSV file
    similar_trials.to_csv(output_file_path, index=False)
    print(f"Output saved to {output_file_path}")

# Run the Pipeline
if __name__ == "__main__":
    # Replace the file path with the actual path to your CSV file in Kaggle
    file_path = "/kaggle/input/clinical-trials-novartis-nit/usecase_1_.csv"  # Adjust the file path
    main(file_path, output_file_path="similar_trials_output.csv")
