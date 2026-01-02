from plot_results import plot_learning_curve

if __name__ == "__main__":
    monitor_csv_path = "models/model_170_125K/monitor55K.csv"
    plot_learning_curve(monitor_csv_path)
