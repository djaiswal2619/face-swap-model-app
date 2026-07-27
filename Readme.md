Dataset:
   Link: https://www.kaggle.com/datasets/atulanandjha/lfwpeople

Models:
   Link: https://huggingface.co/ezioruan/inswapper_128.onnx/blob/main/inswapper_128.onnx

Results Location:
   Graphs: outputs/graphs/*.png

        Graph 1: accuracy_pairsDevTrain.png
        Graph 2: confusion_matrix.png
        Graph 3: embedding_separation.png
        Graph 4: matrics_breakdown.png
        Graph 5: overall_performance.png

   Images: outputs/images/swapped_*.png

   Videos: outputs/videos/swapped_*.mp4

   Trained models: models/

Run commands:
1. pip install -r requirements.txt
2. python update_dataset.py
3. python train_classifier.py
4. python generate_graphs.py
5. streamlit run app.py