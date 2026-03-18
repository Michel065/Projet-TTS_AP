doskey projet=cd C:\Users\Maste\"OneDrive - Aix-Marseille Université"\Bureau\ecole_pc\5A\T2\"Apprentissage Profond"\projet\Projet-TTS_AP
projet
prompt %CONDA_PROMPT_MODIFIER%$g
cls


#### rac projet
doskey projet=cd C:\Users\Maste\"OneDrive - Aix-Marseille Université"\Bureau\ecole_pc\5A\T2\"Apprentissage Profond"\projet\Projet-TTS_AP

#### pour redruire le chemin : 
prompt $g

prompt %CONDA_PROMPT_MODIFIER%$g

#### retablire ou presque:
prompt $p$g


# Projet-TTS_AP

Exemple: 
from tensorflow.keras import layers, models

input_shape = (80, 100, 1)  # mel spectrogram

encoder = models.Sequential([
    layers.Conv2D(16, 3, activation="relu", padding="same"),
    layers.MaxPool2D(2),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPool2D(2)
])

decoder = models.Sequential([
    layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same"),
    layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same"),
    layers.Conv2D(1, 3, activation="sigmoid", padding="same")
])

input_layer = layers.Input(shape=input_shape)
encoded = encoder(input_layer)
decoded = decoder(encoded)

autoencoder = models.Model(input_layer, decoded)
autoencoder.summary()

### ce qu'on veux

Model model;

model.add(Conv2D(...));
model.add(MaxPool(...));
model.add(Conv2D(...));

model.add(ConvTranspose(...));
model.add(ConvTranspose(...));

Tensor out = model.forward(x);