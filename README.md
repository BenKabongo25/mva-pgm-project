# MAGMA Project

We worked on the article **MAGMA: inference and prediction using multi-task Gaussian processes with common mean** written by Arthur Leroy, Pierre Latouche, Benjamin Guedj, and Servane Gey.

The model proposed in this article is a multi-task Gaussian process, applied to time series forecasting, where processes don't share a common covariance matrix and zero mean, as in most previous work, but share a common mean. The model is trained with the EM algorithm to calculate its parameters.
