import ultralytics

def main():
    # Load a model
    model = ultralytics.YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(data='./num_classificator_data/data', epochs=60, imgsz=640)

if __name__ == "__main__":
    main()