from config import *
from utils.data_handling import load_dataset
from parallel.processor import process_dataset_parallel
from evaluation.metrics import save_results, evaluate_accuracy

RunningMode = "evaluate" # "inference" or "evaluate"

def main():
    
    if RunningMode == "inference":
        # Load dataset
        dataset = load_dataset(inference_file, sample_size=300)
        
        # Process dataset in parallel
        results = process_dataset_parallel(dataset)
        
        # Generate submission files
        save_results(results, "output")
        
    elif RunningMode == "evaluate":
        # Load dataset
        dataset = load_dataset(evaluation_file, sample_size=3)
        
        # Process dataset in parallel
        results = process_dataset_parallel(dataset)
        
        # Generate submission files
        save_results(results, "output")
        
        # Evaluate accuracy
        ground_truth = [item for item in dataset]
        predictions = [item for item in results]
        accuracy = evaluate_accuracy(predictions, ground_truth)["accuracy"]
        print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()