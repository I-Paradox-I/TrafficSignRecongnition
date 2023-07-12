# Train the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()  # Set the model to train mode
#     running_loss = 0.0
#     correct_predictions = 0

#     progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

#     for images, labels in progress_bar:
#         images = images.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         # Track training loss and accuracy
#         running_loss += loss.item() * images.size(0)
#         _, predicted = torch.max(outputs, 1)
#         correct_predictions += (predicted == labels).sum().item()

#         progress_bar.set_postfix(loss=loss.item())

#     # Calculate average training loss and accuracy
#     epoch_loss = running_loss / len(train_dataset)
#     epoch_accuracy = correct_predictions / len(train_dataset)

#     tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")
