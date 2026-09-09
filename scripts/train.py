import argparse

def main():
    parser = argparse.ArgumentParser(description="Train the music recommender model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to processed data")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "wrmf"], help="Which model to train")
    args = parser.parse_args()

    # TODO: load data, train model, save results
    print(f"Training {args.model} model using data from {args.data_path}")

if __name__ == "__main__":
    main()