# Jump Attentive Graph Neural Networks (JA-GNN)

This repository provides a base implementation of **Jump Attentive Graph Neural Networks (JA-GNN)**, as described in [this paper](https://arxiv.org/pdf/2411.05857).

## Key Notes
Due to proprietary constraints, certain components of the original production model are not included in this repository. The following features have been excluded:

1. **Temporal Node Properties:**  
   - Temporal features used as node properties in the production model are not provided.

2. **Hyperparameter Tuning Techniques:**  
   - Advanced hyperparameter tuning methodologies used in the production model are not shared.

3. **Custom Edge Weights:**  
   - Pre-calculated edge weight files for datasets have been excluded.

4. **Custom Implementations:**  
   - Proprietary customizations and optimizations from the production model are not included.

## Overview
As the availability of financial services online continues to grow, the incidence of fraud has surged correspondingly. Fraudsters continually seek new and innovative ways to circumvent the detection algorithms in place. Traditionally,fraud detection relied on rule-based methods, where rules were manually created based on transaction data features. However, these techniques soon became ineffective due to their reliance on manual rule creation and their inability to detect complex data patterns. Today, a significant portion of the financial services sector employs various machine learning algorithms, such as XGBoost, Random Forest, and neural networks, to model transaction data. While these techniques have proven more efficient than rule-based methods, they still fail to capture interactions between different transactions and their interrelationships. Recently, graph-based techniques have been adopted for financial fraud detection, leveraging graph topology to aggregate neighborhood information of transaction data using Graph Neural Networks (GNNs). Despite showing improvements over previous methods, these techniques still struggle to keep pace with the evolving camouflaging tactics of fraudsters and suffer from information loss due to over-smoothing. In this paper, we propose a novel algorithm that employs an efficient neighborhood sampling method, effective for camouflage detection and preserving crucial feature information from non-similar nodes. Additionally, we introduce a novel GNN architecture that utilizes attention mechanisms and preserves holistic neighborhood information to prevent information loss. We test our algorithm on financial data to show that our method outperforms other state-of-the-art graph algorithms.

This repository serves as a foundation for exploring the concepts presented in the paper and can be extended to suit specific datasets or tasks.
