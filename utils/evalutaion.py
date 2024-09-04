# utils/evaluation.py
import torch
import torch.nn.functional as F

def calculate_loss(outputs, targets, vocab_size):
    # Calculate cross-entropy loss
    loss = F.cross_entropy(outputs.view(-1, vocab_size), targets.view(-1))
    return loss

def evaluate_model(model, dataloader, vocab_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions in dataloader:
            outputs = model(images, captions[:, :-1])
            loss = calculate_loss(outputs, captions[:, 1:], vocab_size)
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {average_loss:.4f}")
    return average_loss
