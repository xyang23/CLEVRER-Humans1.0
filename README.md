# CLEVRER-Humans1.0

Building machines that can reason about physical events and their causal relationships is crucial for flexible interaction with the physical world. However, most existing physical and causal reasoning benchmarks are exclusively based on synthetically generated events and synthetic natural language descriptions of the causal relationships. This design brings up two issues. First, there is a lack of diversity in both event types and natural language descriptions; second, causal relationships based on manually-defined heuristics are different from human judgments. To address both shortcomings, we present the CLEVRER-Humans benchmark, a video reasoning dataset for causal judgment of physical events with human labels. We employ two techniques to improve data collection efficiency: first, a novel iterative event cloze task to elicit a new representation of events in videos, which we term Causal Event Graphs (CEGs); second, a data augmentation technique based on neural language generative models. We convert the collected CEGs into questions and answers to be consistent with prior work. Finally, we study a collection of baseline approaches for CLEVRER-Humans question-answering, highlighting great challenges set forth by our benchmark.

# Usage

`captioning.py`:  Captioning model for description generation.

`dataset_pipeline.py`: Pipeline to generate CLEVRER-Humans, including trajectory filtering, object and event grounding, and bounding box visualization.

`dataloader.py`: Dataloader for CEGs and QA pairs.

`models`: baseline models.

# Data
Data and more info is on our project website: https://sites.google.com/stanford.edu/clevrer-humans/home
