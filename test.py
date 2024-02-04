import torch
from torch.utils.data import DataLoader
from src.utils.config_loader import Config
from src.data.dataset import BuildingDataset 
from src.models.model import UNet 
from src.utils.transformations import get_transformations
from src.utils.metrics import compute_metrics
from src.utils.logger import setup_logger


def test(model, dataloader, device):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            metrics = compute_metrics(outputs, masks)
            all_metrics.append(metrics)
  
    return all_metrics

def main():
    config = Config.get_instance().config
    logger = setup_logger('test', 'logs/test.log')
    logger.info("Starting testing...")

    # Load the test dataset
    transform = get_transformations()
    test_dataset = BuildingDataset(config['data']['test_path'], 'images', 'masks', transform )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Load the model
    model = UNet(n_channels=3, n_classes=1).to(config["device"])
    model.load_state_dict(torch.load(config["checkpoint_path"], map_location=config["device"]))

    # Perform testing
    test_metrics = test(model, test_loader, config["device"])
    for metric_name in test_metrics[0]: 
        avg_metric = sum(d[metric_name] for d in test_metrics) / len(test_metrics)
        logger.info(f"{metric_name}: {avg_metric}")

