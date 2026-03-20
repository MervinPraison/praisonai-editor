# INA Speech Segmenter (`--detector ina`)

Uses a CNN (Convolutional Neural Network) trained on broadcast media to detect speech vs music vs noise vs silence.

## Install

```bash
pip install "praisonai-editor[detect]"
```

## Usage

```bash
praisonai-editor edit file.mp3 --preset speech_only --detector ina
```

## Strengths

- High accuracy for **speech vs music** boundary detection
- Handles noisy environments well (broadcast quality)
- Good at detecting short speech segments within music

## Weaknesses

- Slower (CNN inference)
- Cannot distinguish **singing** from music (both labeled as `music`)
- Use with `--demix` to get singing classification

## Content types returned

| INA label | Mapped to |
|-----------|-----------|
| `speech` | `speech` |
| `music` | `music` |
| `noEnergy` | `silence` |
