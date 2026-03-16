#### rac projet
doskey projet=cd C:\Users\Maste\"OneDrive - Aix-Marseille Université"\Bureau\ecole_pc\5A\T2\"Apprentissage Profond"\projet\Projet-TTS_AP

#### pour redruire le chemin : 
prompt $g

prompt %CONDA_PROMPT_MODIFIER%$g

#### retablire ou presque:
prompt $p$g



# Projet-TTS_AP
├── src/
│   ├── main.cpp
│   ├── audio/
│   │   ├── audio_loader.cpp
│   │   ├── audio_saver.cpp
│   │   └── wav_reader.cpp
│   ├── features/
│   │   ├── spectrogram.cpp
│   │   ├── mel_spectrogram.cpp
│   │   └── phoneme_converter.cpp
│   ├── dataset/
│   │   ├── dataset_loader.cpp
│   │   ├── dataset_parser.cpp
│   │   └── sample_builder.cpp
│   ├── model/
│   │   ├── tensor.cpp
│   │   ├── layer_dense.cpp
│   │   ├── activation.cpp
│   │   ├── autoencoder.cpp
│   │   ├── encoder.cpp
│   │   ├── decoder.cpp
│   │   ├── quantizer.cpp
│   │   └── sequential_model.cpp
│   ├── train/
│   │   ├── train_autoencoder.cpp
│   │   ├── train_sequential.cpp
│   │   └── loss.cpp
│   ├── inference/
│   │   ├── encode_audio.cpp
│   │   ├── decode_latent.cpp
│   │   ├── text_to_phoneme.cpp
│   │   └── generate_speech.cpp
│   └── utils/
│       ├── config.cpp
│       ├── logger.cpp
│       └── file_utils.cpp
│
├── include/
│   ├── audio/
│   │   ├── audio_loader.h
│   │   ├── audio_saver.h
│   │   └── wav_reader.h
│   ├── features/
│   │   ├── spectrogram.h
│   │   ├── mel_spectrogram.h
│   │   └── phoneme_converter.h
│   ├── dataset/
│   │   ├── dataset_loader.h
│   │   ├── dataset_parser.h
│   │   └── sample_builder.h
│   ├── model/
│   │   ├── tensor.h
│   │   ├── layer_dense.h
│   │   ├── activation.h
│   │   ├── autoencoder.h
│   │   ├── encoder.h
│   │   ├── decoder.h
│   │   ├── quantizer.h
│   │   └── sequential_model.h
│   ├── train/
│   │   ├── train_autoencoder.h
│   │   ├── train_sequential.h
│   │   └── loss.h
│   ├── inference/
│   │   ├── encode_audio.h
│   │   ├── decode_latent.h
│   │   ├── text_to_phoneme.h
│   │   └── generate_speech.h
│   └── utils/
│       ├── config.h
│       ├── logger.h
│       └── file_utils.h
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── phonemes/
│   └── latents/
│
├── models/
│   ├── autoencoder/
│   └── sequential/
│
├── tests/
│   ├── test_spectrogram.cpp
│   ├── test_autoencoder.cpp
│   └── test_sequential.cpp
│
├── Makefile
└── README.md