        model_probabilities.append(probabilities.cpu().numpy() * model_weights[i])

#         # Weighted Voting
#         ensemble_probabilities = np.sum(model_probabilities, axis=0)
#         ensemble_probabilities /= np.sum(ensemble_probabilities, axis=1, keepdims=True)

#         # Select the label with the highest weighted probability for each sample
#         ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)

#         # Track predictions and true labels
#         predictions.extend(ensemble_predictions)
#         true_labels.extend(labels.cpu().numpy())

# # Calculate test accuracy and F1 score
# test_accuracy = accuracy_score(true_labels, predictions)
# f1_macro = f1_score(true_labels, predictions, average='macro')
# print