import torch
import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_size=10):
        super(RecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(5, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.genre_embedding = nn.Embedding(31, embedding_size)
        self.year_embedding = nn.Embedding(100 + 1, embedding_size)  # Assuming a range of 100 years
        self.fc = nn.Linear(embedding_size * 4, 1)

    def forward(self, user, movie, genre, year):
        user_embed = self.user_embedding(user)
        movie_embed = self.movie_embedding(movie)
        genre_embed = self.genre_embedding(genre)

        year_embed = self.year_embedding(year)
        x = torch.cat([user_embed, movie_embed, genre_embed, year_embed], dim=1)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class RatingDataset(Dataset):
    def __init__(self, user_ratings):
        self.user_ratings = user_ratings

    def __len__(self):
        return len(self.user_ratings)

    def __getitem__(self, idx):
        return (
            torch.LongTensor([self.user_ratings[idx]['user']]),
            torch.LongTensor([self.user_ratings[idx]['movie']]),
            torch.LongTensor([self.user_ratings[idx]['genre']]),
            torch.LongTensor([self.user_ratings[idx]['year']]),
            torch.LongTensor([self.user_ratings[idx]['rating']])
        )

# Trainingsprozess
def train(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        for user, movie, rating, genre, year in train_loader:
            optimizer.zero_grad()
            output = model(user, movie, genre, year)
            loss = criterion(output, rating.view(-1, 1))
            loss.backward()
            optimizer.step()

# Daten für das Beispiel
def read_csv(filename):
    user_ratings = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            user_ratings.append({
                'user': int(row['user']),
                'movie': int(row['movie']),
                'genre': int(row['genre']),# Genre als Integer codiert, 1-n durchnummerieren
                'year': int(row['year']),    #Erscheinungsjahr als Integer codiert; vor X Jahren erschienen
                'rating': int(row['rating'])
            })
    return user_ratings

# Dateiname der CSV-Datei
csv_filename = 'data.csv'

# Daten für das Beispiel
user_ratings = read_csv(csv_filename)

# Cold Start: Neue Benutzer oder Filme hinzufügen
new_user_id = max([rating['user'] for rating in user_ratings]) + 1
new_movie_id = 4  # / 2, je nach Beispiel halt, bzw. für welchen Film man die Empfehlung möchte
new_genre = 1     # Beispielgenre
new_year = 0    # Beispieljahr

# Modell, Optimizer und DataLoader initialisieren
num_users = max([rating['user'] for rating in user_ratings]) + 1
num_movies = max([rating['movie'] for rating in user_ratings]) + 1
num_genres = 2

model = RecommenderModel(num_users, num_movies, num_genres)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
dataset = RatingDataset(user_ratings)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Überprüfe die Größe der Embedding-Matrizen
print(f"Größe der Benutzer-Embedding-Matrix: {model.user_embedding.weight.size()}")
print(f"Größe der Film-Embedding-Matrix: {model.movie_embedding.weight.size()}")
print(f"Größe der Genre-Embedding-Matrix: {model.genre_embedding.weight.size()}")
print(f"Größe der Jahr-Embedding-Matrix: {model.year_embedding.weight.size()}")

unique_genres = set([rating['genre'] for rating in user_ratings])
unique_years = set([rating['year'] for rating in user_ratings])

print("Einzigartige Genre-IDs:", unique_genres)
print("Einzigartige Jahr-IDs:", unique_years)
# Training durchführen
for name, param in model.named_parameters():
    print(name, param.dtype)
train(model, train_loader, criterion, optimizer)

# Vorhersage für einen neuen Benutzer und Film
new_user = torch.LongTensor([new_user_id])
new_movie = torch.LongTensor([new_movie_id])
new_genre = torch.LongTensor([new_genre])
new_year = torch.LongTensor([new_year])
prediction = model(new_user, new_movie, new_genre, new_year)

# Ergebnis ausgeben
print(f"Die Vorhersage für Benutzer {new_user.item()}, Film {new_movie.item()}, "
      f"Genre {new_genre.item()} und Jahr {new_year.item()} ist {prediction.item()}")