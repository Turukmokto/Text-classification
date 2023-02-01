# Text-classification

Data set

1. The dataset contains many messages (files).
2. If the file name contains the legit substring, then this is a good message.
3. If the file name contains the spmsg substring, then it is spam.
4. Messages consist of a header and a message body.
5. Each word has been replaced with a specific number.

Exercise

1. Come up with two ways to take into account the header and body of the message when vectorizing. For example: consider equally or consider separately.
2. Think of two ways to vectorize text. For example, for each word, create a sign: met or not, or how many times met.
3. The tabular dataset must be in sparse form.
4. Plot the Naive Bayes classifier on the dataset.
5. Plot the ROC curve. Calculate the AUC for it. Choose the best combination of the first two points in relation to the AUC metric.
6. Change the prior distribution for the selected model so that the number of good messages classified as spam is zero. In this case, the accuracy should be as high as possible.
7. Plot classification accuracy versus change in the prior distribution in logarithmic space.
