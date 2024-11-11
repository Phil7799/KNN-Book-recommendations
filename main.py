# Import libraries
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import requests, zipfile, io

# Download and unzip the data
url = "https://cdn.freecodecamp.org/project-data/books/book-crossings.zip"
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall()

# Define filenames
books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# Import CSV data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding="ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# Check the number of unique users and books before filtering
print(f"Number of unique users: {df_ratings['user'].nunique()}")
print(f"Number of unique books (ISBNs): {df_ratings['isbn'].nunique()}")

# Step 1: Apply stricter filtering to reduce data size

# Filter users who have rated at least 100 books
user_ratings_count = df_ratings.groupby('user').size()
filtered_users = user_ratings_count[user_ratings_count >= 100].index

# Filter books that have received at least 50 ratings
book_ratings_count = df_ratings.groupby('isbn').size()
filtered_books = book_ratings_count[book_ratings_count >= 50].index

# Apply filtering
df_filtered = df_ratings[(df_ratings['user'].isin(filtered_users)) & (df_ratings['isbn'].isin(filtered_books))]

# Merge with book data to get titles and authors
df_filtered = pd.merge(df_filtered, df_books, on='isbn', how='inner')

# Check the shape of the merged dataset after filtering
print(f"Shape of filtered dataset: {df_filtered.shape}")

# Step 2: Sample the data (e.g., 10% of the rows to reduce size for quick testing)
df_filtered_sampled = df_filtered.sample(frac=0.1, random_state=42)

# Check the shape of the sampled dataset
print(f"Shape of sampled dataset: {df_filtered_sampled.shape}")

# Step 3: Remove duplicates by keeping the highest rating for each (user, title) pair
df_filtered_sampled = df_filtered_sampled.groupby(['title', 'user'], as_index=False).agg({'rating': 'max'})

# Create the pivot table for book-user matrix
book_user_matrix = df_filtered_sampled.pivot(index='title', columns='user', values='rating').fillna(0)

# Check if the matrix is empty after filtering
if book_user_matrix.shape[0] == 0 or book_user_matrix.shape[1] == 0:
    raise ValueError("The matrix is empty after filtering. Try adjusting the filtering thresholds.")

# Convert to sparse matrix format
book_user_sparse_matrix = csr_matrix(book_user_matrix.values)

# Step 4: Fit the KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
model_knn.fit(book_user_sparse_matrix)

# Function to get book recommendations
def get_recommends(book_title):
    book_title_lower = book_title.lower()
    book_title_mapping = {title.lower(): title for title in book_user_matrix.index}

    if book_title_lower not in book_title_mapping:
        return f"Book '{book_title}' not found in the dataset. Please check the title."

    original_title = book_title_mapping[book_title_lower]
    book_idx = book_user_matrix.index.get_loc(original_title)

    # Find nearest neighbors with adjusted n_neighbors to get closer recommendations
    distances, indices = model_knn.kneighbors(book_user_matrix.iloc[book_idx, :].values.reshape(1, -1), n_neighbors=10)

    # Gather recommendations
    recommended_books = []
    for i in range(1, len(distances.flatten())):
        recommended_books.append([book_user_matrix.index[indices.flatten()[i]], round(distances.flatten()[i], 2)])

    return [original_title, recommended_books]

# Testing the recommendation system
def test_book_recommendation():
    test_pass = True
    recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
    
    # Expected recommendations
    expected_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
    expected_distances = [0.8, 0.77, 0.77, 0.77]
    
    # Check the main book title
    if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
        test_pass = False
    
    # Check each recommended book and its distance
    for i in range(4):
        if recommends[1][i][0] not in expected_books:
            print(f"Mismatch in recommended book: {recommends[1][i][0]} not in expected list.")
            test_pass = False
        if abs(recommends[1][i][1] - expected_distances[i]) >= 0.05:
            print(f"Mismatch in distance: Expected {expected_distances[i]}, but got {recommends[1][i][1]}")
            test_pass = False
    
    if test_pass:
        print("You passed the challenge! 🎉🎉🎉🎉🎉")
    else:
        print("You haven't passed yet. Keep trying!")

# Run the test case
test_book_recommendation()
