import joblib
import os


def load_cls_model(classifier_path,
                   scaler_path,
                   selector_path=None):
    """
    Load the classifier model, scaler, and optional selector.

    Args:
        classifier_path (str): Path to the classifier model.
        scaler_path (str): Path to the scaler.
        selector_path (str, optional): Path to the selector model. Defaults to None.

    Returns:
        model: The loaded classifier model.
        scaler: The loaded scaler.
        selector: The loaded selector, or None if not provided or failed to load.
    """
    # Load the model
    model = joblib.load(classifier_path)
    print(f"Classifier loaded successfully from {classifier_path}")

    # Load the scaler
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded successfully from {scaler_path}")

    selector = None  # Default to None

    # Load the selector if provided
    if selector_path and os.path.exists(selector_path):
        try:
            selector = joblib.load(selector_path)
            print(f"Selector loaded successfully from {selector_path}")
        except Exception as e:
            print(f"Failed to load selector from {selector_path}: {e}")
    else:
        print(f"No selector path provided or file does not exist at {selector_path}. Skipping selector loading.")

    return model, scaler, selector
