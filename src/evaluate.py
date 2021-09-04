import sys
sys.path.insert(1, '/src')
import train
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import mlflow
from urllib.parse import urlparse


def eval_metrics(actual, pred):
    acc_score = accuracy_score(actual, pred)
    f1Score = f1_score(actual, pred, average='macro')
    tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()
    return acc_score, f1Score, tn, fp, fn, tp

def main():
    with mlflow.start_run():
        model, test_X, test_y = train.main()
        pred_y = model.predict(test_X)

        acc_score, f1Score, tn, fp, fn, tp = eval_metrics(test_y, pred_y)

        print(acc_score)
        print(f1Score)
        print(tn, fp, fn, tp)

        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("accuracy", acc_score)
        mlflow.log_param("f1 Score", f1Score)
        mlflow.log_metric("TN", tn)
        mlflow.log_metric("TP", tp)
        mlflow.log_metric("FP", fp)
        mlflow.log_metric("FN", fn)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForest")
        else:
            mlflow.sklearn.log_model(model, "model")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    mlflow.end_run()

if __name__ == "__main__":
    main()  






