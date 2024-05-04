from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

experi_epochs = []
experi_accuracy = []

all_top_10_f1_scores = [[] for _ in range(4)]
all_top_10_class_names = []

for i in range(4):
    
    log_dir = f'runs/resnet18_experiment{(i+1)*25}_logs/'

    # Load TensorBoard logs
    event_acc = EventAccumulator(log_dir)
    try:
        event_acc.Reload()
    except Exception as e:
        print("Error loading event files:", e)
    
    # Get the scalar data for accuracy over epochs
    # accuracy_data = event_acc.Scalars('Accuracy/test')
    # # Extract epochs and accuracy values
    # epochs = [scalar.step for scalar in accuracy_data]
    # accuracies = [scalar.value for scalar in accuracy_data]
    # experi_epochs.append(epochs)
    # experi_accuracy.append(accuracies)

    # Retrieve precision and recall values for each class
    
    class_names = [f'class_{i}' for i in range(62)]  # Replace with your class names
    class_ids = {
    "shipyard": 0,
    "parking_lot_or_garage": 1,
    "crop_field": 2,
    "flooded_road": 3,
    "race_track": 4,
    "electric_substation": 5,
    "recreational_facility": 6,
    "impoverished_settlement": 7,
    "smokestack": 8,
    "zoo": 9,
    "nuclear_powerplant": 10,
    "office_building": 11,
    "port": 12,
    "tunnel_opening": 13,
    "police_station": 14,
    "ground_transportation_station": 15,
    "park": 16,
    "wind_farm": 17,
    "fire_station": 18,
    "storage_tank": 19,
    "tower": 20,
    "place_of_worship": 21,
    "toll_booth": 22,
    "multi-unit_residential": 23,
    "stadium": 24,
    "barn": 25,
    "construction_site": 26,
    "railway_bridge": 27,
    "helipad": 28,
    "debris_or_rubble": 29,
    "fountain": 30,
    "surface_mine": 31,
    "lake_or_pond": 32,
    "runway": 33,
    "car_dealership": 34,
    "lighthouse": 35,
    "burial_site": 36,
    "oil_or_gas_facility": 37,
    "amusement_park": 38,
    "aquaculture": 39,
    "road_bridge": 40,
    "factory_or_powerplant": 41,
    "waste_disposal": 42,
    "airport_hangar": 43,
    "airport": 44,
    "single-unit_residential": 45,
    "educational_institution": 46,
    "gas_station": 47,
    "golf_course": 48,
    "military_facility": 49,
    "swimming_pool": 50,
    "shopping_mall": 51,
    "interchange": 52,
    "hospital": 53,
    "border_checkpoint": 54,
    "space_facility": 55,
    "prison": 56,
    "archaeological_site": 57,
    "dam": 58,
    "solar_farm": 59,
    "water_treatment_facility": 60,
    "airport_terminal": 61
}
    flipped_ids = {value: key for key, value in class_ids.items()}
    # Calculate mean F1 score for each class
    class_metrics = {}
    for idx,class_name in enumerate(class_names):
        f1_score_tag = f'F1-Score/{class_name}'
        f1_score_values = [scalar.value for scalar in event_acc.Scalars(f1_score_tag)]
        class_metrics[flipped_ids[idx]] = f1_score_values[-1]
       
    #Sort classes based on mean F1 score
    sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1], reverse=False)
    #print(sorted_classes)
    top_10_classes = [class_name for class_name, _ in sorted_classes[:10]]
    top_10_f1_scores = [f1_score for _, f1_score in sorted_classes[:10]]

    all_top_10_f1_scores[i] = top_10_f1_scores
    all_top_10_class_names.extend(top_10_classes)

    # # Plot bar chart for top 10 classes
    # plt.figure(figsize=(10, 6))
colors = ['skyblue', 'purple', 'orange', 'green']
    # plt.bar(top_10_classes, top_10_f1_scores, color=colors[i])
    # plt.xlabel('Class')
    # plt.ylabel('F1 Score')
    # plt.title(f'F1 Score for Top 10 Classes Trained On {(i+1)*25}% of Data')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.show()

    # Plot combined results
plt.figure(figsize=(12, 8))

# Plot bars for each experiment
bar_width = 0.2
tick_positions = []
for i in range(4):
    x = np.arange(10) + i * bar_width  # Adjust x positions for each experiment
    plt.bar(x, all_top_10_f1_scores[i], width=bar_width, color=colors[i], label=f'Experiment {(i+1)*25}%')

    tick_positions.extend(x + bar_width/2)

all_top_10_classes = []
for i in range(4):
    all_top_10_classes.extend(top_10_classes)
# Customize the plot
plt.xlabel('Class')
plt.ylabel('F1 Score')
plt.title('F1 Score for Bottom 10 Classes Across Experiments')
plt.xticks(tick_positions, all_top_10_class_names, rotation=90, ha='right')
plt.legend(ncol=4)
plt.tight_layout()
plt.savefig('trial5_f1score_comparison_plot.png')
plt.show()

# # Plot the data using Matplotlib
# for i in range(len(experi_epochs)):
#     plt.plot(experi_epochs[i], experi_accuracy[i], label=f'Trained On {(i+1)*25}% of Data')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Test Accuracy Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.savefig('trial5_accuracy_comparison_plot.png')
# plt.show()