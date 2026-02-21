# Deep Learning Projects

A collection of deep learning experiments and tools.

## Folders

### Binary Classification
CNN-based binary image classification in both PyTorch and TensorFlow, adapted from a Kaggle notebook.

### CarRecognition
Transfer-learning pipeline (EfficientNet via `timm`) to identify a car's make/model/year from images. Includes dataset downloading, training, CLI inference, and a Tkinter GUI for labeling.

### Contact Classification
Whisker-pole contact detection classifier using EfficientNet-B3. Extracts frames from video based on annotated contact/no-contact intervals, builds a binary dataset, and trains a model to classify whether a whisker is contacting a pole in a given frame.

### ConvNeXT
Exploration and analysis of the ConvNeXt architecture (Small and Large variants) with pretrained ImageNet weights. Includes inference notebooks and model summary exports.

### FigureSummarizer
Extracts figures and captions from scientific PDFs, then generates natural-language summaries of each figure using a vision-language model (BLIP-2). Runnable via CLI or Jupyter notebook.

### Keypoints
Custom keypoint detection using PyTorch's Keypoint R-CNN. Includes COCO-format annotation conversion and the torchvision reference training utilities for detecting keypoints on glue tubes.

### Testing
Hierarchical summarization pipeline using BART (`facebook/bart-large-cnn`). Extracts text from PDFs, splits into chunks, and recursively summarizes. Includes fine-tuning experiments.
