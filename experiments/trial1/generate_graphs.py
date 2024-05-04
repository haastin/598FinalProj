from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

experi_epochs = []
experi_accuracy = []

for i in range(4):
    log_dir = f'runs/resnet18_experiment{(i+1)*25}_logs/'

    # Load TensorBoard logs
    event_acc = EventAccumulator(log_dir)
    try:
        event_acc.Reload()
    except Exception as e:
        print("Error loading event files:", e)

    # Get the scalar data for accuracy over epochs
    accuracy_data = event_acc.Scalars('Accuracy/test')

    # Extract epochs and accuracy values
    epochs = [scalar.step for scalar in accuracy_data]
    accuracies = [scalar.value for scalar in accuracy_data]
    experi_epochs.append(epochs)
    experi_accuracy.append(accuracies)

# Plot the data using Matplotlib
for i in range(len(experi_epochs)):
    plt.plot(experi_epochs[i], experi_accuracy[i], label=f'Trained On {(i+1)*25}% of Data')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('trial5_accuracy_comparison_plot.png')
plt.show()