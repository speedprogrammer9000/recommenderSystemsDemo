import torch
import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#Definiere ein einfaches Recommender-Modell
class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=10):
        super(RecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.fc = nn.Linear(embedding_size * 2, 1)

    def forward(self, user, movie):
        user_embed = self.user_embedding(user)
        movie_embed = self.movie_embedding(movie)
        x = torch.cat([user_embed, movie_embed], dim=1)
        x = x.view(x.size(0), -1)  # Ändere die Form zu (Batch_Size, embedding_size * 2)
        return self.fc(x)


#Erstelle den Datensatz und den DataLoader
class RatingDataset(Dataset):
    def __init__(self, user_ratings):
        self.user_ratings = user_ratings

    def __len__(self):
        return len(self.user_ratings)

    def __getitem__(self, idx):
        return torch.LongTensor([self.user_ratings[idx]['user']]), torch.LongTensor([self.user_ratings[idx]['movie']]), torch.FloatTensor([self.user_ratings[idx]['rating']])

#Trainingsprozess
def train(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        for user, movie, rating in train_loader:
            optimizer.zero_grad()
            output = model(user, movie)
            loss = criterion(output, rating.view(-1, 1))
            loss.backward()
            optimizer.step()


#Daten für das Beispiel
def read_csv(filename):
    user_ratings = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            user_ratings.append({
                'user': int(row['user']),
                'movie': int(row['movie']),
                'rating': int(row['rating'])
            })
    return user_ratings

# Dateiname der CSV-Datei
csv_filename = 'data.csv'

#Daten für das Beispiel
#user_ratings = [
#    {'user': 1, 'movie': 1, 'rating': 1}, # für demo hier die werte von 0-5 verschieben, ergebniss ändert sich entsprechend
#    {'user': 2, 'movie': 1, 'rating': 1},
#    {'user': 3, 'movie': 1, 'rating': 1},
#    {'user': 1, 'movie': 2, 'rating': 5},
#    {'user': 2, 'movie': 2, 'rating': 5},
#    {'user': 3, 'movie': 2, 'rating': 5},
#]

# Initialisierung der Variable user_ratings
user_ratings = read_csv(csv_filename)

#Cold Start: Neue Benutzer oder Filme hinzufügen
new_user_id = max([rating['user'] for rating in user_ratings]) + 1

new_movie_id = 1 # / 2, je nach beispiel halt, bzw für welchen film man die recommendation will


#Modell, Optimizer und DataLoader initialisieren
model = RecommenderModel(num_users=6, num_movies=5) # num_users muss > anzahl user in user_rating sein, analog num_movies

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
dataset = RatingDataset(user_ratings)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Überprüfe die Größe der Embedding-Matrizen, vgl paar zeilen oben
print(f"Größe der Benutzer-Embedding-Matrix: {model.user_embedding.weight.size()}")
print(f"Größe der Film-Embedding-Matrix: {model.movie_embedding.weight.size()}")

#Training durchführen

# Führe das Training für das Modell durch
train(model, train_loader, criterion, optimizer)

# Vorhersage für einen neuen Benutzer und Film
new_user = torch.LongTensor([new_user_id])
new_movie = torch.LongTensor([new_movie_id])
prediction = model(new_user, new_movie)

# Ergebnis ausgeben
print(f"Die Vorhersage für Benutzer {new_user_id} und Film {new_movie_id} ist {prediction.item()}")
