from sklearn.model_selection import train_test_split

# Split Data into Training and Test Sets
def split_data(data):
    X = data.drop(columns=['churned'])
    y = data['churned']
    return train_test_split(X, y, test_size=0.2, random_state=42)