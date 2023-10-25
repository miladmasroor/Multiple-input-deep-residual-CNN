# Multiple-input-deep-residual-CNN
A multiple-input deep residual convolutional neural network for reservoir permeability prediction

# Highlights

  - A novel Multiple-Input deep Residual Convolutional Neural Network (named MIRes CNN) is introduced for permeability prediction from conventional well logs.

  - MIRes CNN is simultaneously fed by two different types of datasets, Numerical Well Logs (NWLs) and Graphical Feature Images (GFIs).

  - We integrated a Single-Input deep Residual 1D-CNN to handle the NWL dataset and a Single-Input deep Residual 2D-CNN to treat with GFIs.

  - An innovative approach for image generation from conventional well logs is also introduced, which offers some physical insight.

# Abstract

Permeability plays an essential role in reservoir-related studies, including fluid flow characterization, reservoir modeling/simulation, and management. However, operational constraints and high costs limit the wide access to the direct measurements of reservoir permeability. Over the years, various machine learning (ML) techniques have been utilized to predict reservoir permeability from widely-available reservoir-related datasets (such as well logs). Nevertheless, experience in real case studies reveals that there is still room to improve the process of permeability prediction in both well and field scales. In this study, an innovative Deep Learning (DL)-based approach is developed for fast and accurate reservoir permeability prediction from conventional well logs. By integrating a multiple-input module, convolutional network, deeper bottleneck, and residual structures, we developed a novel Multiple-Input deep Residual Convolutional Neural Network (named MIRes CNN) for this purpose. It is simultaneously fed by two different types of datasets, Numerical Well Logs (NWLs) and Graphical Feature Images (GFIs). To construct this MIRes CNN model, we integrated a Single-Input deep Residual one Dimensional CNN (SIRes 1D-CNN) to handle the NWL dataset and a Single-Input deep Residual two Dimensional CNN (SIRes 2D-CNN) to treat with GFIs. The significant advantages of the proposed architecture are its capability to efficiently exploit mixed data, reduce both overfitting problem and computational cost, and add more flexibility to the process compared to the classical single-input deep neural networks. The other novelty of this work lies in its generation of an image from geophysical well logs, which offers some physical insight. We compared the proposed MIRes CNN model with two Single-Input deep Residual CNN methods (i.e., SIRes 1D-CNN and SIRes 2D-CNN) and two baseline methods. The statistical results concluded that the proposed multiple-input method outperforms single-modality-based techniques and the baseline methods for all employed performance indicators. The study also demonstrates the importance of using physically meaningful log-derived images for CNN model training and the crucial role played by the modified architectures (e.g., residual and deeper bottleneck architectures) in enhancing the accuracy and performance of the model.

# Keywords

Permeability; Conventional well logs; Deep learning; Multiple-input Convolutional neural network; Graphical feature images


For a detailed explanation of the methodology and results, please refer to the associated research paper:

Title: A multiple-input deep residual convolutional neural network for reservoir permeability prediction

Published in: Geoenergy Science and Engineering, 2023

DOI: https://doi.org/10.1016/j.geoen.2023.211420
